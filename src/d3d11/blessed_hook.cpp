// blessed: generic post-draw hook, fired after a game draw whose pixel shader matches BLESSED_HOOK_PS
#include "blessed_hook.h"
#include "blessed_shadow.h"
#include "blessed_scene_capture.h" // blessed: skin-v2, mask pass for skinned capture

#include <exception>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d11_context_imm.h"
#include "d3d11_texture.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    enum class Mode : uint8_t {
      Off,
      Fill,
      Log,
      // blessed: a consumer supplies its own SetCallback() and just wants
      // OnDraw's match/logging bookkeeping; no implicit `fill` behavior.
      Custom,
    };

    // one token from BLESSED_HOOK_PS: either a full "fs.<hash>" name
    // (exact match) or a bare hex prefix of 8+ chars (prefix match against
    // whatever follows "fs." in the shader's debug name)
    struct Token {
      std::string text;
      bool        fullName = false;
    };

    struct State {
      bool                 initDone = false;
      bool                 enabled  = false;
      Mode                 mode     = Mode::Off;
      std::vector<Token>   tokens;
      bool                 once     = false;
      FLOAT                fillColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

      // per-pixel-shader-pointer match cache; shaders are effectively
      // immutable for their lifetime, so a pointer->bool memo is safe for
      // as long as that pointer stays alive (see blessed_hook.h caveat)
      std::unordered_map<const void*, bool> matchCache;

      BlessedHookCallback  callback;

      // log accounting, reset every 120 presents
      dxvk::high_resolution_clock::time_point processStart;
      std::ofstream        file;
      uint64_t             framesInWindow  = 0;
      uint64_t             matchesInWindow = 0;
      bool                 matchedThisFrame = false;
      BlessedHookTargets   lastTargets;
    };

    State g_state;

    std::vector<Token> ParseTokens(const std::string& csv) {
      std::vector<Token> out;
      size_t pos = 0;

      while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        std::string tok = csv.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? csv.size() : comma + 1;

        if (tok.empty())
          continue;

        Token t;
        if (tok.rfind("fs.", 0) == 0) {
          t.text     = tok;
          t.fullName = true;
        } else {
          t.text     = tok;
          t.fullName = false;
        }
        out.push_back(std::move(t));
      }

      return out;
    }

    bool MatchesName(const std::string& name) {
      for (const auto& tok : g_state.tokens) {
        if (tok.fullName) {
          if (name == tok.text)
            return true;
        } else if (name.rfind("fs.", 0) == 0 && tok.text.size() >= 8) {
          if (name.compare(3, tok.text.size(), tok.text) == 0)
            return true;
        }
      }
      return false;
    }

    // blessed: pure function of env vars alone -- no touch of g_state, no
    // cross-TU reach -- so it is safe to run at static-init time (unlike the
    // rest of EnsureInit()'s work, which stays lazy; see the comment on
    // blessed_hook_detail::g_enabled in blessed_hook.h). Mirrors the same
    // verdict EnsureInit() reaches for g_state.enabled below.
    bool ComputeHookEnabled() {
      std::string psList = env::getEnvVar("BLESSED_HOOK_PS");
      if (psList.empty() || ParseTokens(psList).empty())
        return false;

      std::string modeStr = env::getEnvVar("BLESSED_HOOK_MODE");
      return modeStr == "fill" || modeStr == "log" || modeStr == "rtshadow";
    }

    void EnsureInit() {
      if (g_state.initDone)
        return;

      g_state.initDone = true;
      g_state.processStart = dxvk::high_resolution_clock::now();

      std::string psList = env::getEnvVar("BLESSED_HOOK_PS");
      if (psList.empty())
        return;

      g_state.tokens = ParseTokens(psList);
      if (g_state.tokens.empty())
        return;

      std::string modeStr = env::getEnvVar("BLESSED_HOOK_MODE");
      if (modeStr == "fill")
        g_state.mode = Mode::Fill;
      else if (modeStr == "log")
        g_state.mode = Mode::Log;
      else if (modeStr == "rtshadow") {
        // blessed: installed here, not from a global constructor in
        // blessed_shadow.cpp: static init order across files isn't defined,
        // and g_state's own init wiped that callback (firstlight-2).
        g_state.mode = Mode::Custom;
        g_state.callback = &BlessedShadow::OnDraw;
      }
      else
        return; // "off", unset, or unrecognized: stay disabled

      g_state.once = env::getEnvVar("BLESSED_HOOK_ONCE") == "1";

      std::string fillStr = env::getEnvVar("BLESSED_HOOK_FILL");
      if (!fillStr.empty()) {
        std::vector<Token> parts = ParseTokens(fillStr);
        float parsed[4];
        bool ok = parts.size() == 4;

        for (size_t i = 0; ok && i < 4; i++) {
          try {
            parsed[i] = std::stof(parts[i].text);
          } catch (const std::exception&) {
            ok = false;
          }
        }

        if (ok) {
          for (int i = 0; i < 4; i++)
            g_state.fillColor[i] = parsed[i];
        } else {
          Logger::warn(str::format("BlessedHook: could not parse BLESSED_HOOK_FILL '", fillStr, "', using default"));
        }
      }

      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
      if (dir.empty()) {
        Logger::warn("BlessedHook: BLESSED_HOOK_PS is set but BLESSED_PROBE_DIR is not; log output disabled");
      } else {
        std::string path = dir + env::PlatformDirSlash + "hook.jsonl";
        g_state.file.open(str::topath(path.c_str()), std::ios::out | std::ios::trunc);
      }

      g_state.enabled = true;

      Logger::info(str::format("BlessedHook: enabled, mode=", modeStr,
        ", ", g_state.tokens.size(), " shader token(s)"));
    }

    BlessedHookRtv DescribeRtv0(D3D11RenderTargetView* rtv) {
      BlessedHookRtv info;
      if (!rtv)
        return info;

      D3D11_RENDER_TARGET_VIEW_DESC desc = { };
      rtv->GetDesc(&desc);
      info.fmt = desc.Format;

      const D3D11_VK_VIEW_INFO& vi = rtv->GetViewInfo();
      if (D3D11CommonTexture* tex = GetCommonTexture(vi.pResource)) {
        VkExtent3D ext = tex->MipLevelExtent(vi.Image.MinLevel);
        info.w = ext.width;
        info.h = ext.height;
      }

      return info;
    }

    void WriteLogLine() {
      if (!g_state.file.is_open() || g_state.framesInWindow == 0)
        return;

      double t = std::chrono::duration<double>(
        dxvk::high_resolution_clock::now() - g_state.processStart).count();
      double matchesPerFrame = double(g_state.matchesInWindow) / double(g_state.framesInWindow);

      std::string line = str::format("{\"t\":", t,
        ",\"frames\":", g_state.framesInWindow,
        ",\"matches_per_frame\":", matchesPerFrame,
        ",\"rtv\":[");

      if (g_state.lastTargets.hasRtv0) {
        line += str::format("{\"fmt\":", int32_t(g_state.lastTargets.rtv0.fmt),
          ",\"w\":", g_state.lastTargets.rtv0.w,
          ",\"h\":", g_state.lastTargets.rtv0.h, "}");
      }

      line += "]}\n";

      g_state.file << line;
      g_state.file.flush();

      g_state.framesInWindow  = 0;
      g_state.matchesInWindow = 0;
    }

  }


  // blessed: definition for the header's inline IsEnabled() -- see
  // blessed_hook.h. EnsureInit() (the rest of g_state -- mode, tokens,
  // callback) still runs lazily, from OnDraw()/OnPresent() below, the first
  // time either is reached with the feature enabled.
  namespace blessed_hook_detail {
    extern const bool g_enabled = ComputeHookEnabled();
  }


  void BlessedHook::OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    if (!IsEnabled())
      return;

    EnsureInit();

    if (g_state.once && g_state.matchedThisFrame)
      return;

    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps)
      return;

    const void* key = ps;
    auto it = g_state.matchCache.find(key);
    bool matches;

    if (it != g_state.matchCache.end()) {
      matches = it->second;
    } else {
      matches = MatchesName(ps->GetCommonShader()->GetName());
      g_state.matchCache.emplace(key, matches);
    }

    if (!matches)
      return;

    g_state.matchedThisFrame = true;
    g_state.matchesInWindow++;

    // blessed: skin-v2 -- the hooked draw is the mask draw; skinned capture
    // keeps only the depth-only pass right before it
    if (BlessedSceneCapture::IsEnabled())
      BlessedSceneCapture::NoteMaskDraw();

    BlessedHookTargets targets;
    D3D11RenderTargetView* rtv0 = state.om.maxRtv > 0 ? state.om.rtvs[0].ptr() : nullptr;
    if (rtv0) {
      targets.hasRtv0 = true;
      targets.rtv0    = DescribeRtv0(rtv0);
      targets.rtv0View = rtv0->GetImageView();
    }
    D3D11DepthStencilView* dsv = state.om.dsv.ptr();
    targets.hasDsv = dsv != nullptr;
    if (dsv)
      targets.dsvView = dsv->GetImageView();
    targets.state = &state;

    g_state.lastTargets = targets;

    if (g_state.callback) {
      g_state.callback(ctx, targets);
    } else if (g_state.mode == Mode::Fill && targets.hasRtv0) {
      ctx->ClearRenderTargetView(rtv0, g_state.fillColor);
    }
  }


  void BlessedHook::OnPresent() {
    if (!IsEnabled())
      return;

    EnsureInit();

    g_state.framesInWindow++;
    g_state.matchedThisFrame = false;

    if (g_state.framesInWindow >= 120)
      WriteLogLine();
  }


  void BlessedHook::SetCallback(BlessedHookCallback cb) {
    g_state.callback = std::move(cb);
  }

}
