// blessed: BLESSED_LOOK -- d3d11 side of the bless colour chain: finds skyrim's tonemap and final copy, patches and dispatches
#include "blessed_look.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d11_buffer.h"
#include "d3d11_context_imm.h"
#include "d3d11_view_rtv.h"

#include "../dxvk/blessed/blessed_look.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

// blessed: env, with defaults (profile bless unless noted)
//
//   BLESSED_LOOK=bless                 on; anything else is off
//   BLESSED_LOOK_PROFILE=bless|rori    rori: bloom strength 0.65, haze 0.22,
//                                      threshold 0.73 (her R_EDIT bloom), same grade
//   BLESSED_LOOK_PS_TONEMAP=716590ec   pass 175, ISHDR blend
//   BLESSED_LOOK_PS_FINAL=831de5eb     pass 181, final copy
//   BLESSED_LOOK_FINAL_MATCH=0         which match of the final ps per frame to run after
//   BLESSED_LOOK_VANILLA_BLOOM=0       0: zero Param.x in pass 175; 1: keep vanilla's bloom
//   BLESSED_LOOK_VANILLA_GRADE=1       0: neutral Cinematic/Tint in pass 175 (Fade kept)
//   BLESSED_LOOK_TIMING=0              1: gpu timestamps around the pass, gpu_ms in look.jsonl
//   BLESSED_LOOK_DEBUG=off             see Debug below
//   BLESSED_LOOK_SPLIT=0.5             split view: fraction of the width left vanilla
//
//   numbers (float): BLESSED_LOOK_ + BLOOM_THRESHOLD 0.80, BLOOM_KNEE 0.35,
//   BLOOM_SURFACE 0.10, BLOOM_HAZE 0.10, BLOOM_RADIUS 3.5, BLOOM_STRENGTH 0.32,
//   SPILL_THRESHOLD 0.55, SPILL_KNEE 0.25, SPILL_RADIUS 6.0, SPILL_STRENGTH 0.35,
//   EXPOSURE 1.0, GRAIN 0.03, GRAIN_SCALE 0.60, GRAIN_RATE 24, SHOULDER_KNEE 0.78,
//   SHOULDER_BEND 0.80, WARMTH 0.35, PASTEL 0.22, SPLIT_TONE 0.60,
//   SATURATION 1.06, CONTRAST 0.92, LIFT 0.035, GRADE_STRENGTH 1.0, VIGNETTE 0
//
// Debug (BLESSED_LOOK_DEBUG), and what the frame should show:
//
//   final    the look pass writes solid magenta after the final copy. The
//            world is magenta, the hud (compass, crosshair, menus) draws on
//            top of it. Proves pass 181 is the last world pass before the hud.
//   tonemap  no look pass; pass 175's Fade is forced to (0,1,0) at weight 1.
//            The world is solid green (hud normal). Proves pass 175 is the
//            tonemap and that b2 c5 is Fade, i.e. the whole b2 layout.
//   vbloom   no look pass; pass 175's Param.x forced to 16. Bright bloom
//            glow everywhere, much stronger than vanilla. Proves Param.x.
//   vgrade   no look pass; pass 175's Cinematic.x (saturation) forced to 0.
//            The world is black and white, the hud keeps its colour.
//   bloom    the bless bloom layer alone (screened haze over black)
//   spill    the colour spill layer alone (dark except around saturated lights)
//   grade    the grade and exposure only, no bloom, spill or grain
//   split    left of BLESSED_LOOK_SPLIT vanilla, right the full look, 2 px seam

namespace dxvk {

  namespace {

    enum class LookDebug : int32_t {
      Off     = 0,
      Final   = 1,   // composite debugMode 1
      Bloom   = 2,   // composite debugMode 2
      Spill   = 3,   // composite debugMode 3
      Split   = 4,   // composite debugMode 4
      Grade   = 5,   // composite debugMode 5
      Tonemap = 100, // tonemap-stage modes: no look pass
      VBloom  = 101,
      VGrade  = 102,
    };

    enum class Kind : uint8_t {
      None,
      Tonemap,
      Final,
    };

    struct Token {
      std::string text;
      bool        fullName = false;
    };

    struct LookConfig {
      std::vector<Token> tonemapTokens;
      std::vector<Token> finalTokens;
      uint32_t           finalMatch   = 0;
      bool               vanillaBloom = false;
      bool               vanillaGrade = true;
      bool               timing       = false;
      LookDebug          debug        = LookDebug::Off;
      float              split        = 0.5f;
      float              grainRate    = 24.0f;
      BlessedLookArgs    args;        // the numbers; target and size filled per run
      std::string        profile;
    };

    struct LookState {
      bool                 initDone = false;
      LookConfig               cfg;

      std::unordered_map<const void*, Kind> kindCache;
      const void*          lastPs   = nullptr;
      Kind                 lastKind = Kind::None;

      Rc<BlessedLookPass>  pass;
      bool                 passFailed = false;

      dxvk::high_resolution_clock::time_point processStart;
      std::ofstream        file;

      // per frame
      uint32_t             finalMatchesThisFrame = 0;

      // per 120-present window
      uint64_t             frames         = 0;
      uint64_t             tonemapMatches = 0;
      uint64_t             finalMatches   = 0;
      uint64_t             b2Patched      = 0;
      uint64_t             looksRun       = 0;
      const char*          b2Fail         = nullptr;
      uint32_t             rtFmt = 0, rtW = 0, rtH = 0;

      uint64_t             tonemapTotal   = 0;
      bool                 b2Dumped       = false;
    };

    LookState g_state;

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
        t.text     = tok;
        t.fullName = tok.rfind("fs.", 0) == 0;
        out.push_back(std::move(t));
      }

      return out;
    }

    bool MatchesName(const std::vector<Token>& tokens, const std::string& name) {
      for (const auto& tok : tokens) {
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

    void EnvFloat(const char* name, float& value) {
      std::string s = env::getEnvVar(name);
      if (s.empty())
        return;

      char* end = nullptr;
      float v = std::strtof(s.c_str(), &end);

      if (end != s.c_str() && std::isfinite(v))
        value = v;
      else
        Logger::warn(str::format("BlessedLook: could not parse ", name, "='", s, "', keeping ", value));
    }

    void EnvBool(const char* name, bool& value) {
      std::string s = env::getEnvVar(name);
      if (s == "1")
        value = true;
      else if (s == "0")
        value = false;
    }

    std::vector<Token> EnvTokens(const char* name, const char* fallback) {
      std::string s = env::getEnvVar(name);
      return ParseTokens(s.empty() ? std::string(fallback) : s);
    }

    LookDebug ParseDebug(const std::string& s) {
      if (s == "final")   return LookDebug::Final;
      if (s == "bloom")   return LookDebug::Bloom;
      if (s == "spill")   return LookDebug::Spill;
      if (s == "split")   return LookDebug::Split;
      if (s == "grade")   return LookDebug::Grade;
      if (s == "tonemap") return LookDebug::Tonemap;
      if (s == "vbloom")  return LookDebug::VBloom;
      if (s == "vgrade")  return LookDebug::VGrade;
      if (!s.empty() && s != "off")
        Logger::warn(str::format("BlessedLook: unknown BLESSED_LOOK_DEBUG '", s, "', off"));
      return LookDebug::Off;
    }

    void EnsureInit() {
      if (g_state.initDone)
        return;

      g_state.initDone = true;
      g_state.processStart = dxvk::high_resolution_clock::now();

      LookConfig& c = g_state.cfg;
      BlessedLookArgs& a = c.args;
      BlessedLookCompositePush& p = a.composite;

      c.profile = env::getEnvVar("BLESSED_LOOK_PROFILE");
      if (c.profile.empty())
        c.profile = "blessed"; // blessed: rori picked the blend as the default (2026-09-23)

      // blessed: bless = the shipped default-config.json; rori = her R_EDIT
      // bloom (rmls-irl.settings.txt, 2026-09-09) with bless's grade
      if (c.profile == "rori") {
        p.bloomStrength  = 0.65f;
        a.bloomHaze      = 0.22f;
        a.bloomThreshold = 0.73f;
      } else if (c.profile == "blessed") {
        // blessed: rori 2026-09-23, "go with bless but move some of rori into
        // it": bless's grade, the bloom halfway to her r_edit, her grain and
        // exposure, and less haze than either (skyrim's daylight is already
        // hazy; bless's 0.10 read milky on it, look-bless-1)
        p.bloomStrength  = 0.48f;
        a.bloomHaze      = 0.07f;
        a.bloomThreshold = 0.77f;
        p.grainStrength  = 0.024f;
        p.grainScale     = 0.57f;
        p.exposure       = 0.95f;
      } else if (c.profile != "bless") {
        Logger::warn(str::format("BlessedLook: unknown BLESSED_LOOK_PROFILE '", c.profile, "', using bless"));
        c.profile = "bless";
      }

      c.tonemapTokens = EnvTokens("BLESSED_LOOK_PS_TONEMAP", "716590ec");
      c.finalTokens   = EnvTokens("BLESSED_LOOK_PS_FINAL",   "98e1bd7b,831de5eb"); // blessed: 98e1bd7b is the final pass on the live frame (look-proof2)

      float finalMatch = 0.0f;
      EnvFloat("BLESSED_LOOK_FINAL_MATCH", finalMatch);
      c.finalMatch = uint32_t(std::max(finalMatch, 0.0f));

      EnvBool("BLESSED_LOOK_VANILLA_BLOOM", c.vanillaBloom);
      EnvBool("BLESSED_LOOK_VANILLA_GRADE", c.vanillaGrade);
      EnvBool("BLESSED_LOOK_TIMING",        c.timing);
      c.debug = ParseDebug(env::getEnvVar("BLESSED_LOOK_DEBUG"));
      EnvFloat("BLESSED_LOOK_SPLIT", c.split);

      EnvFloat("BLESSED_LOOK_BLOOM_THRESHOLD", a.bloomThreshold);
      EnvFloat("BLESSED_LOOK_BLOOM_KNEE",      a.bloomKnee);
      EnvFloat("BLESSED_LOOK_BLOOM_SURFACE",   a.bloomSurface);
      EnvFloat("BLESSED_LOOK_BLOOM_HAZE",      a.bloomHaze);
      EnvFloat("BLESSED_LOOK_BLOOM_RADIUS",    a.bloomRadius);
      EnvFloat("BLESSED_LOOK_BLOOM_STRENGTH",  p.bloomStrength);
      EnvFloat("BLESSED_LOOK_SPILL_THRESHOLD", a.spillThreshold);
      EnvFloat("BLESSED_LOOK_SPILL_KNEE",      a.spillKnee);
      EnvFloat("BLESSED_LOOK_SPILL_RADIUS",    a.spillRadius);
      EnvFloat("BLESSED_LOOK_SPILL_STRENGTH",  p.spillStrength);
      EnvFloat("BLESSED_LOOK_EXPOSURE",        p.exposure);
      EnvFloat("BLESSED_LOOK_GRAIN",           p.grainStrength);
      EnvFloat("BLESSED_LOOK_GRAIN_SCALE",     p.grainScale);
      EnvFloat("BLESSED_LOOK_GRAIN_RATE",      c.grainRate);
      EnvFloat("BLESSED_LOOK_SHOULDER_KNEE",   p.shoulderKnee);
      EnvFloat("BLESSED_LOOK_SHOULDER_BEND",   p.shoulderBend);
      EnvFloat("BLESSED_LOOK_WARMTH",          p.warmth);
      EnvFloat("BLESSED_LOOK_PASTEL",          p.pastel);
      EnvFloat("BLESSED_LOOK_SPLIT_TONE",      p.splitTone);
      EnvFloat("BLESSED_LOOK_SATURATION",      p.saturation);
      EnvFloat("BLESSED_LOOK_CONTRAST",        p.contrast);
      EnvFloat("BLESSED_LOOK_LIFT",            p.lift);
      EnvFloat("BLESSED_LOOK_GRADE_STRENGTH",  p.gradeStrength);
      EnvFloat("BLESSED_LOOK_VIGNETTE",        p.vignette);

      a.timing = c.timing;
      p.debugMode = int32_t(c.debug) < 100 ? int32_t(c.debug) : 0;

      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
      if (!dir.empty()) {
        std::string path = dir + env::PlatformDirSlash + "look.jsonl";
        g_state.file.open(str::topath(path.c_str()), std::ios::out | std::ios::trunc);
      }

      Logger::info(str::format("BlessedLook: enabled, profile=", c.profile,
        ", bloom ", p.bloomStrength, "/", a.bloomHaze, "/", a.bloomThreshold,
        ", spill ", p.spillStrength, "/", a.spillThreshold,
        ", grain ", p.grainStrength,
        ", vanilla bloom ", c.vanillaBloom ? "kept" : "zeroed",
        ", vanilla grade ", c.vanillaGrade ? "kept" : "neutral",
        ", debug ", int32_t(c.debug)));
    }

    Kind Classify(const D3D11ContextState& state) {
      D3D11PixelShader* ps = state.ps.ptr();
      if (!ps)
        return Kind::None;

      const void* key = ps;
      if (key == g_state.lastPs)
        return g_state.lastKind;

      Kind kind;
      auto it = g_state.kindCache.find(key);

      if (it != g_state.kindCache.end()) {
        kind = it->second;
      } else {
        const std::string& name = ps->GetCommonShader()->GetName();
        kind = MatchesName(g_state.cfg.tonemapTokens, name) ? Kind::Tonemap
             : MatchesName(g_state.cfg.finalTokens,   name) ? Kind::Final
             : Kind::None;
        g_state.kindCache.emplace(key, kind);
      }

      g_state.lastPs   = key;
      g_state.lastKind = kind;
      return kind;
    }

    // blessed: ps b2 of the tonemap draw, as a float pointer into the
    // mapped cbuffer, or nullptr (and a reason) when it can't be patched
    float* MapTonemapB2(const D3D11ContextState& state, const char** reason) {
      const auto& cbv = state.cbv[D3D11ShaderType::ePixel];

      if (cbv.maxCount <= 2u) {
        *reason = "b2 unbound";
        return nullptr;
      }

      D3D11Buffer* buffer = cbv.buffers[2].buffer.ptr();
      if (!buffer) {
        *reason = "b2 unbound";
        return nullptr;
      }

      auto* base = reinterpret_cast<uint8_t*>(buffer->GetMapPtr());
      if (!base) {
        *reason = "b2 not mapped";
        return nullptr;
      }

      // c0..c5: Flags, TimingData, Param, Cinematic, Tint, Fade
      UINT byteOffset = cbv.buffers[2].constantOffset * 16u;
      if (byteOffset + 6u * 16u > buffer->Desc()->ByteWidth) {
        *reason = "b2 too small";
        return nullptr;
      }

      return reinterpret_cast<float*>(base + byteOffset);
    }

    void DumpB2(const float* b2) {
      static const char* names[6] = { "Flags", "TimingData", "Param", "Cinematic", "Tint", "Fade" };

      std::string line = "{\"b2\":{";
      for (uint32_t i = 0; i < 6; i++) {
        const float* v = b2 + 4u * i;
        line += str::format(i ? "," : "", "\"", names[i], "\":[", v[0], ",", v[1], ",", v[2], ",", v[3], "]");
        Logger::info(str::format("BlessedLook: tonemap b2 c", i, " ", names[i], " = (",
          v[0], ", ", v[1], ", ", v[2], ", ", v[3], ")"));
      }
      line += "}}\n";

      if (g_state.file.is_open()) {
        g_state.file << line;
        g_state.file.flush();
      }
    }

    void WriteLogLine() {
      if (!g_state.file.is_open() || !g_state.frames)
        return;

      double t = std::chrono::duration<double>(
        dxvk::high_resolution_clock::now() - g_state.processStart).count();
      double frames = double(g_state.frames);
      float gpuMs = g_state.pass != nullptr ? g_state.pass->lastGpuMs() : -1.0f;

      std::string line = str::format("{\"t\":", t,
        ",\"frames\":", g_state.frames,
        ",\"tonemap_per_frame\":", double(g_state.tonemapMatches) / frames,
        ",\"final_per_frame\":", double(g_state.finalMatches) / frames,
        ",\"b2_patched_per_frame\":", double(g_state.b2Patched) / frames,
        ",\"look_per_frame\":", double(g_state.looksRun) / frames,
        ",\"gpu_ms\":", gpuMs,
        ",\"rt\":{\"fmt\":", g_state.rtFmt, ",\"w\":", g_state.rtW, ",\"h\":", g_state.rtH, "}");

      if (g_state.b2Fail)
        line += str::format(",\"b2_fail\":\"", g_state.b2Fail, "\"");

      line += "}\n";
      g_state.file << line;
      g_state.file.flush();

      g_state.frames         = 0;
      g_state.tonemapMatches = 0;
      g_state.finalMatches   = 0;
      g_state.b2Patched      = 0;
      g_state.looksRun       = 0;
      g_state.b2Fail         = nullptr;
    }

    bool ComputeLookEnabled() {
      return env::getEnvVar("BLESSED_LOOK") == "bless";
    }

  }


  namespace blessed_look_detail {
    extern const bool g_enabled = ComputeLookEnabled();
  }


  void BlessedLook::OnPreDraw(const D3D11ContextState& state) {
    EnsureInit();

    if (Classify(state) != Kind::Tonemap)
      return;

    const LookConfig& c = g_state.cfg;
    g_state.tonemapMatches++;
    g_state.tonemapTotal++;

    const char* reason = nullptr;
    float* b2 = MapTonemapB2(state, &reason);

    if (!b2) {
      g_state.b2Fail = reason;
      return;
    }

    // the untouched values, once, after the game has settled
    if (!g_state.b2Dumped && g_state.tonemapTotal >= 120u) {
      g_state.b2Dumped = true;
      DumpB2(b2);
    }

    float* param     = b2 + 8u;
    float* cinematic = b2 + 12u;
    float* tint      = b2 + 16u;
    float* fade      = b2 + 20u;

    if (!c.vanillaBloom)
      param[0] = 0.0f;

    if (!c.vanillaGrade) {
      cinematic[0] = 1.0f;  // saturation
      cinematic[2] = 1.0f;  // contrast
      cinematic[3] = 1.0f;  // brightness
      tint[3]      = 0.0f;  // tint weight
    }

    switch (c.debug) {
      case LookDebug::Tonemap:
        fade[0] = 0.0f; fade[1] = 1.0f; fade[2] = 0.0f; fade[3] = 1.0f;
        break;
      case LookDebug::VBloom:
        param[0] = 16.0f;
        break;
      case LookDebug::VGrade:
        cinematic[0] = 0.0f;
        break;
      default:
        break;
    }

    g_state.b2Patched++;
  }


  void BlessedLook::OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device) {
    EnsureInit();

    if (Classify(state) != Kind::Final)
      return;

    const LookConfig& c = g_state.cfg;
    g_state.finalMatches++;

    if (g_state.finalMatchesThisFrame++ != c.finalMatch)
      return;

    // tonemap-stage debug modes show vanilla's chain with one forced value
    if (int32_t(c.debug) >= 100 || g_state.passFailed)
      return;

    D3D11RenderTargetView* rtv0 = state.om.maxRtv > 0 ? state.om.rtvs[0].ptr() : nullptr;
    if (!rtv0)
      return;

    Rc<DxvkImageView> view = rtv0->GetImageView();
    Rc<DxvkImage> image = view->image();
    DxvkImageViewKey viewInfo = view->info();

    if (image->info().sampleCount != VK_SAMPLE_COUNT_1_BIT) {
      Logger::warn("BlessedLook: final copy target is multisampled; look disabled");
      g_state.passFailed = true;
      return;
    }

    if (g_state.pass == nullptr)
      g_state.pass = new BlessedLookPass(device);

    if (!g_state.pass->usable()) {
      g_state.passFailed = true;
      return;
    }

    VkExtent3D extent = image->mipLevelExtent(viewInfo.mipIndex);

    D3D11_RENDER_TARGET_VIEW_DESC rtvDesc = { };
    rtv0->GetDesc(&rtvDesc);
    g_state.rtFmt = uint32_t(rtvDesc.Format);
    g_state.rtW   = extent.width;
    g_state.rtH   = extent.height;

    BlessedLookArgs args = c.args;
    args.target = image;
    args.targetLayers = { VK_IMAGE_ASPECT_COLOR_BIT, viewInfo.mipIndex, viewInfo.layerIndex, 1u };
    args.width  = extent.width;
    args.height = extent.height;

    // 24 re-rolls a second (bless: floor(GameTime * 1200 * 24)), wrapped
    // to bless's own range so the hash input stays small
    double seconds = std::chrono::duration<double>(
      dxvk::high_resolution_clock::now() - g_state.processStart).count();
    args.composite.grainTime = float(std::fmod(std::floor(seconds * double(c.grainRate)), 28800.0));
    args.composite.splitX    = int32_t(float(extent.width) * std::clamp(c.split, 0.0f, 1.0f));

    ctx->EmitCs([cPass = g_state.pass, cArgs = std::move(args)] (DxvkContext* dxvkCtx) {
      cPass->run(dxvkCtx, cArgs);
    });

    // the pass bound its own compute shader and views; put the game's back
    ctx->RestoreCommandListState();

    g_state.looksRun++;
  }


  void BlessedLook::OnPresent() {
    EnsureInit();

    g_state.finalMatchesThisFrame = 0;
    g_state.frames++;

    if (g_state.frames >= 120u)
      WriteLogLine();
  }

}
