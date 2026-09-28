// blessed: half-rate far field -- app-thread half: far-draw skip, main-pass tracking, the live switch
#include "blessed_halfrate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "d3d11_buffer.h"
#include "d3d11_context_imm.h"
#include "d3d11_shader.h"
#include "d3d11_view_dsv.h"
#include "d3d11_view_rtv.h"

#include "../dxvk/blessed/blessed_halfrate.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // blessed: the shared switch the skse plugin publishes. Keep in step
    // with skse/skybench/src/halfrate_switch.cpp in the main repo.
    struct BlessedHalfRateShared {
      uint32_t      magic;
      uint32_t      version;
      volatile LONG enabled;
      volatile LONG toggles;
    };

    constexpr uint32_t    SharedMagic   = 0x46524842u; // "BHRF"
    constexpr uint32_t    SharedVersion = 1u;
    constexpr const wchar_t* SharedName = L"Local\\BlessedHalfRate";

    std::vector<std::string> SplitCsv(const std::string& csv) {
      std::vector<std::string> out;
      size_t pos = 0;

      while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        std::string tok = csv.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? csv.size() : comma + 1;

        if (!tok.empty())
          out.push_back(std::move(tok));
      }

      return out;
    }

    float EnvFloat(const char* name, float fallback) {
      std::string s = env::getEnvVar(name);
      return s.empty() ? fallback : std::strtof(s.c_str(), nullptr);
    }

    struct HalfRateConfig {
      bool                     farField     = true;
      bool                     defaultOn    = true;
      uint32_t                 period       = 2u;
      std::vector<std::string> vsTokens     = { "231ba7e4" };
      float                    minScale     = 3.5f;
      float                    farDist      = 0.0f;
      float                    captureDist  = 4000.0f;
      float                    maxMove      = 256.0f;
      float                    maxTurnCos   = 0.0f;
      bool                     timing       = false;
      bool                     reflectMoved = false;

      static const HalfRateConfig& Get() {
        static HalfRateConfig s_config = [] {
          HalfRateConfig c;

          std::string parts = env::getEnvVar("BLESSED_HALFRATE_PARTS");
          if (!parts.empty()) {
            c.farField = false;
            for (const auto& p : SplitCsv(parts)) {
              if (p == "far")     c.farField = true;
              // blessed: refl-harden -- the reflect part moved to
              // BLESSED_REFLECT_HALFRATE (blessed_reflect_halfrate.h),
              // which also picks this token up
              if (p == "reflect") c.reflectMoved = true;
            }
          }

          c.defaultOn = env::getEnvVar("BLESSED_HALFRATE_DEFAULT") != "0";

          std::string period = env::getEnvVar("BLESSED_HALFRATE_PERIOD");
          if (!period.empty())
            c.period = std::clamp<uint32_t>(uint32_t(std::strtoul(period.c_str(), nullptr, 10)), 2u, 4u);

          std::string vs = env::getEnvVar("BLESSED_HALFRATE_VS");
          if (!vs.empty())
            c.vsTokens = SplitCsv(vs == "none" ? std::string() : vs);

          c.minScale    = EnvFloat("BLESSED_HALFRATE_MINSCALE", c.minScale);
          c.farDist     = EnvFloat("BLESSED_HALFRATE_DIST", c.farDist);
          c.captureDist = EnvFloat("BLESSED_HALFRATE_CAPTURE_DIST", c.captureDist);
          c.maxMove     = EnvFloat("BLESSED_HALFRATE_MAX_MOVE", c.maxMove);

          float maxTurnDeg = EnvFloat("BLESSED_HALFRATE_MAX_TURN", 10.0f);
          c.maxTurnCos = std::cos(std::clamp(maxTurnDeg, 0.0f, 90.0f) * (3.14159265358979f / 180.0f));

          c.timing = env::getEnvVar("BLESSED_HALFRATE_TIMING") == "1";

          Logger::info(str::format("BlessedHalfRate: available (far=", c.farField ? 1 : 0,
            c.reflectMoved ? " reflect=moved to BLESSED_REFLECT_HALFRATE" : "", " period=", c.period,
            " minscale=", c.minScale, " capture_dist=", c.captureDist,
            " default=", c.defaultOn ? "on" : "off", ")"));
          return c;
        }();

        return s_config;
      }
    };

    bool ComputeAvailable() {
      return env::getEnvVar("BLESSED_HALFRATE") != "0";
    }

    struct Stats {
      uint64_t frames        = 0;
      uint64_t offFrames     = 0;
      uint64_t captures      = 0;
      uint64_t cameraRefused = 0;
      uint64_t farSkipped    = 0;
      uint64_t mainDraws     = 0;
      uint64_t liveFrames    = 0;
    };

    struct State {
      // the live switch
      HANDLE                       mapping  = nullptr;
      const BlessedHalfRateShared* shared   = nullptr;
      high_resolution_clock::time_point lastOpenTry;
      bool                         openTried = false;
      bool                         live      = true;
      bool                         liveKnown = false;

      // frame bookkeeping
      uint64_t frame              = 1;
      uint64_t lastCaptureFrame   = 0;     // 0: none
      bool     mainSeen           = false; // this frame's main pass has opened
      bool     mainOpen           = false;
      bool     frameOff           = false; // far draws are being skipped

      // the main lit pass of this frame
      uint64_t                 omGeneration = ~0ull;
      bool                     omIsMain     = false;
      D3D11RenderTargetView*   mainRtv[4]   = { };
      D3D11DepthStencilView*   mainDsv      = nullptr;
      BlessedHalfRateTargets   targets;
      float                    invViewProj[16] = { };
      float                    camPos[3]       = { };
      bool                     camOk           = false;

      // the layer's camera, app-thread copy (the pass keeps its own)
      float                    layerCamPos[3]  = { };
      float                    layerFwd[3]     = { };

      // memos, keyed by object pointer (same caveat as BlessedHook's)
      std::unordered_map<const void*, bool> vsMemo;

      Rc<BlessedHalfRatePass>  pass;
      bool                     passFailed = false;

      Stats                    stats;
      std::ofstream            logFile;
      bool                     logTried = false;
      high_resolution_clock::time_point processStart = high_resolution_clock::now();
    };

    State g;

    bool ReadCbFloats(
      const D3D11ContextState&  state,
            D3D11ShaderType     stage,
            uint32_t            slot,
            uint32_t            offset,
            float*              out,
            uint32_t            count) {
      const auto& cbvStage = state.cbv[stage];
      if (slot >= cbvStage.maxCount)
        return false;

      D3D11Buffer* buffer = cbvStage.buffers[slot].buffer.ptr();
      if (!buffer)
        return false;

      // replay-correct under the threaded front end: GetMapPtr is the slice
      // this draw saw (see D3D11Buffer::BlessedAppMapPtr)
      const void* mapPtr = buffer->GetMapPtr();
      if (!mapPtr)
        return false;

      UINT byteWidth  = buffer->Desc()->ByteWidth;
      UINT byteOffset = cbvStage.buffers[slot].constantOffset * 16u + offset;
      if (byteOffset + count * sizeof(float) > byteWidth)
        return false;

      std::memcpy(out, reinterpret_cast<const uint8_t*>(mapPtr) + byteOffset, count * sizeof(float));
      return true;
    }

    // the view direction of CameraViewProjInverse (row-major bytes, v * M)
    void ForwardOf(const float m[16], float out[3]) {
      const float v[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
      float h[4];

      for (uint32_t j = 0; j < 4; j++)
        h[j] = v[0] * m[4*j+0] + v[1] * m[4*j+1] + v[2] * m[4*j+2] + v[3] * m[4*j+3];

      float w = std::abs(h[3]) > 1.0e-20f ? h[3] : 1.0f;
      float p[3] = { h[0] / w, h[1] / w, h[2] / w };
      float len = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);

      for (uint32_t i = 0; i < 3; i++)
        out[i] = len > 1.0e-20f ? p[i] / len : 0.0f;
    }

    bool NameMatches(const std::string& name, const std::vector<std::string>& tokens) {
      // "vs.<hash>": prefix match on the hash, as BLESSED_HOOK_PS does
      if (name.size() < 4)
        return false;

      for (const auto& tok : tokens) {
        if (name.compare(3, tok.size(), tok) == 0)
          return true;
      }
      return false;
    }

    DXGI_FORMAT RtvFormat(D3D11RenderTargetView* rtv) {
      D3D11_RENDER_TARGET_VIEW_DESC desc = { };
      rtv->GetDesc(&desc);
      return desc.Format;
    }

    VkExtent3D RtvExtent(D3D11RenderTargetView* rtv) {
      Rc<DxvkImageView> view = rtv->GetImageView();
      return view != nullptr ? view->mipLevelExtent(0) : VkExtent3D { 0u, 0u, 0u };
    }

    // the main lit pass: exactly four targets (rgba16f, rg16f, rgba8, rg8)
    // of one size and a depth target. Cached per OM generation.
    bool IsMainPass(const D3D11ContextState& state) {
      if (state.om.blessedOmGeneration == g.omGeneration)
        return g.omIsMain;

      g.omGeneration = state.om.blessedOmGeneration;
      g.omIsMain = false;

      if (state.om.maxRtv != 4u || !state.om.dsv.ptr())
        return false;

      D3D11RenderTargetView* rtv[4];
      for (uint32_t i = 0; i < 4u; i++) {
        rtv[i] = state.om.rtvs[i].ptr();
        if (!rtv[i])
          return false;
      }

      if (RtvFormat(rtv[0]) != DXGI_FORMAT_R16G16B16A16_FLOAT
       || RtvFormat(rtv[1]) != DXGI_FORMAT_R16G16_FLOAT
       || RtvFormat(rtv[2]) != DXGI_FORMAT_R8G8B8A8_UNORM
       || RtvFormat(rtv[3]) != DXGI_FORMAT_R8G8_UNORM)
        return false;

      VkExtent3D e0 = RtvExtent(rtv[0]);
      for (uint32_t i = 1; i < 4u; i++) {
        VkExtent3D e = RtvExtent(rtv[i]);
        if (e.width != e0.width || e.height != e0.height)
          return false;
      }

      g.omIsMain = e0.width >= 64u && e0.height >= 64u;
      return g.omIsMain;
    }

    bool IsFarDraw(const D3D11ContextState& state) {
      const HalfRateConfig& c = HalfRateConfig::Get();

      auto vs = state.vs.ptr();
      if (!vs)
        return false;

      auto it = g.vsMemo.find(vs);
      bool tree;

      if (it != g.vsMemo.end()) {
        tree = it->second;
      } else {
        tree = NameMatches(vs->GetCommonShader()->GetName(), c.vsTokens);
        g.vsMemo.emplace(vs, tree);
      }

      if (tree)
        return true;

      // the lighting shader's World (vs b2, c0-c2, row-major float3x4)
      float w[12];
      if (!ReadCbFloats(state, D3D11ShaderType::eVertex, 2u, 0u, w, 12u))
        return false;

      float scale = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
      float dist  = std::sqrt(w[3]*w[3] + w[7]*w[7] + w[11]*w[11]);

      // a scaled-up mesh counts only beyond the capture distance, so a big
      // full-detail rock next to the player never becomes a hole (the lod
      // in the whiterun frame starts at 8,889 units)
      if (scale >= c.minScale && dist >= c.captureDist)
        return true;

      return c.farDist > 0.0f && dist >= c.farDist;
    }

    void UpdateWatch() {
      const HalfRateConfig& c = HalfRateConfig::Get();
      blessed_halfrate_detail::g_watch = g.live
        && c.farField && !g.passFailed && (!g.mainSeen || g.mainOpen);
    }

    BlessedHalfRateArgs MakeArgs() {
      const HalfRateConfig& c = HalfRateConfig::Get();
      BlessedHalfRateArgs args;
      args.targets     = g.targets;
      std::memcpy(args.invViewProj, g.invViewProj, sizeof(args.invViewProj));
      std::memcpy(args.camPos, g.camPos, sizeof(args.camPos));
      args.captureDist = c.captureDist;
      args.timing      = c.timing;
      return args;
    }

    void PollSwitch() {
      const HalfRateConfig& c = HalfRateConfig::Get();

      if (!g.shared) {
        auto now = high_resolution_clock::now();

        if (!g.openTried || now - g.lastOpenTry >= std::chrono::seconds(1)) {
          g.openTried   = true;
          g.lastOpenTry = now;

          HANDLE mapping = ::OpenFileMappingW(FILE_MAP_READ, FALSE, SharedName);

          if (mapping) {
            void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, sizeof(BlessedHalfRateShared));

            if (view) {
              g.mapping = mapping;
              g.shared  = reinterpret_cast<const BlessedHalfRateShared*>(view);
              Logger::info("BlessedHalfRate: found the plugin's switch (Local\\BlessedHalfRate)");
            } else {
              ::CloseHandle(mapping);
            }
          }
        }
      }

      bool live = c.defaultOn;

      if (g.shared && g.shared->magic == SharedMagic && g.shared->version == SharedVersion)
        live = g.shared->enabled != 0;

      if (!g.liveKnown || live != g.live) {
        Logger::info(str::format("BlessedHalfRate: ", live ? "on" : "off",
          g.shared ? " (plugin switch)" : " (default)"));
        g.liveKnown = true;
        g.live = live;
      }
    }

    void WriteLog() {
      Stats& s = g.stats;

      if (!g.logTried) {
        g.logTried = true;
        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

        if (!dir.empty()) {
          std::error_code ec;
          std::filesystem::create_directories(dir, ec);
          g.logFile.open(dir + env::PlatformDirSlash + "halfrate.jsonl", std::ios::out | std::ios::app);
        }
      }

      if (g.logFile.is_open()) {
        double t = std::chrono::duration<double>(high_resolution_clock::now() - g.processStart).count();
        double frames = double(std::max<uint64_t>(s.frames, 1u));
        double offs   = double(std::max<uint64_t>(s.offFrames, 1u));
        float capMs = g.pass != nullptr ? g.pass->lastCaptureMs() : -1.0f;
        float preMs = g.pass != nullptr ? g.pass->lastPrefillMs() : -1.0f;

        g.logFile << str::format("{\"t\":", t,
          ",\"frames\":", s.frames,
          ",\"live_frames\":", s.liveFrames,
          ",\"off_frames\":", s.offFrames,
          ",\"captures\":", s.captures,
          ",\"camera_refused\":", s.cameraRefused,
          ",\"far_skipped_per_off_frame\":", double(s.farSkipped) / offs,
          ",\"main_draws_per_frame\":", double(s.mainDraws) / frames,
          ",\"capture_gpu_ms\":", capMs,
          ",\"prefill_gpu_ms\":", preMs,
          ",\"switch\":", g.live ? 1 : 0,
          ",\"plugin\":", g.shared ? 1 : 0, "}\n");
        g.logFile.flush();
      }

      s = Stats();
    }

  }


  namespace blessed_halfrate_detail {
    extern const bool g_available = ComputeAvailable();
    bool g_watch = false;
  }


  void BlessedHalfRate::BuildStates(BlessedHalfRateArgs& args) {
    using Ctx = D3D11CommonContext<D3D11ImmediateContext>;
    args.rsState = Ctx::InitDefaultRasterizerState();
    args.rsState.setCullMode(VK_CULL_MODE_NONE);
    args.msState = Ctx::InitDefaultMultisampleState(D3D11_DEFAULT_SAMPLE_MASK);
    args.loState = Ctx::InitDefaultLogicOpState();
    args.cbState = Ctx::InitDefaultBlendState();
  }


  void BlessedHalfRate::OpenMainPass(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device) {
    const HalfRateConfig& c = HalfRateConfig::Get();

    g.mainSeen = true;
    g.mainOpen = true;
    g.frameOff = false;
    g.camOk    = false;

    for (uint32_t i = 0; i < 4u; i++) {
      g.mainRtv[i] = state.om.rtvs[i].ptr();
      g.targets.rtv[i] = g.mainRtv[i]->GetImageView();
    }

    g.mainDsv = state.om.dsv.ptr();
    g.targets.dsv = g.mainDsv->GetImageView();

    VkExtent3D extent = g.targets.rtv[0]->mipLevelExtent(0);
    g.targets.width  = extent.width;
    g.targets.height = extent.height;

    // the capture samples rt0, rt2, rt3 and copies the depth: check that
    // the images allow it before any draw is skipped
    for (uint32_t i : { 0u, 2u, 3u }) {
      if (!(g.targets.rtv[i]->image()->info().usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        Logger::warn(str::format("BlessedHalfRate: main pass target ", i, " is not sampleable; half-rate far field off"));
        g.passFailed = true;
        return;
      }
    }

    if (!(g.targets.dsv->image()->info().usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
      Logger::warn("BlessedHalfRate: main depth cannot be copied; half-rate far field off");
      g.passFailed = true;
      return;
    }

    // this frame's camera, from the pass's own per-frame cbuffer
    g.camOk = ReadCbFloats(state, D3D11ShaderType::ePixel, 12u, 512u, g.invViewProj, 16u)
           && ReadCbFloats(state, D3D11ShaderType::ePixel, 12u, 640u, g.camPos, 3u);

    if (!c.farField || !g.camOk)
      return;

    if (g.pass == nullptr)
      g.pass = new BlessedHalfRatePass(device);

    if (!g.pass->usable()) {
      Logger::warn("BlessedHalfRate: the far layer is unusable; half-rate far field off");
      g.passFailed = true;
      return;
    }

    uint64_t age = g.lastCaptureFrame ? g.frame - g.lastCaptureFrame : ~0ull;

    if (age < 1u || age >= uint64_t(c.period))
      return;

    // a cut or a fast move renders fresh: the layer would be too far off
    float d[3] = {
      g.camPos[0] - g.layerCamPos[0],
      g.camPos[1] - g.layerCamPos[1],
      g.camPos[2] - g.layerCamPos[2] };
    float fwd[3];
    ForwardOf(g.invViewProj, fwd);
    float move = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    float turn = fwd[0]*g.layerFwd[0] + fwd[1]*g.layerFwd[1] + fwd[2]*g.layerFwd[2];

    if (move > c.maxMove || turn < c.maxTurnCos) {
      g.stats.cameraRefused++;
      return;
    }

    g.frameOff = true;
    g.stats.offFrames++;

    BlessedHalfRateArgs args = MakeArgs();
    BuildStates(args);

    ctx->EmitCs([cPass = g.pass, cArgs = std::move(args)] (DxvkContext* dxvkCtx) {
      cPass->prefill(dxvkCtx, cArgs);
    });

    // the prefill bound its own targets, shaders and views; put the game's back
    ctx->RestoreCommandListState();
  }

  void BlessedHalfRate::CloseMainPass(D3D11ImmediateContext* ctx) {
    const HalfRateConfig& c = HalfRateConfig::Get();

    g.mainOpen = false;

    if (c.farField && !g.passFailed && !g.frameOff && g.camOk && g.pass != nullptr && g.pass->usable()) {
      BlessedHalfRateArgs args = MakeArgs();

      ctx->EmitCs([cPass = g.pass, cArgs = std::move(args)] (DxvkContext* dxvkCtx) {
        cPass->capture(dxvkCtx, cArgs);
      });

      ctx->RestoreCommandListState();

      g.lastCaptureFrame = g.frame;
      std::memcpy(g.layerCamPos, g.camPos, sizeof(g.layerCamPos));
      ForwardOf(g.invViewProj, g.layerFwd);
      g.stats.captures++;
    }

    g.frameOff = false;
    g.targets = BlessedHalfRateTargets();
    UpdateWatch();
  }


  bool BlessedHalfRate::OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device) {
    const HalfRateConfig& c = HalfRateConfig::Get();

    if (c.farField && !g.passFailed && IsMainPass(state)) {
      if (!g.mainSeen)
        OpenMainPass(ctx, state, device);

      if (!g.mainOpen)
        return false;

      g.stats.mainDraws++;

      if (g.frameOff && IsFarDraw(state)) {
        g.stats.farSkipped++;
        return true;
      }

      return false;
    }

    if (g.mainOpen)
      CloseMainPass(ctx);

    return false;
  }


  bool BlessedHalfRate::OnClearRtv(D3D11ImmediateContext* ctx, D3D11RenderTargetView* /* rtv */) {
    if (g.mainOpen)
      CloseMainPass(ctx);

    return false;
  }


  bool BlessedHalfRate::OnClearDsv(D3D11ImmediateContext* ctx, D3D11DepthStencilView* /* dsv */) {
    if (g.mainOpen)
      CloseMainPass(ctx);

    return false;
  }


  void BlessedHalfRate::OnDispatch(D3D11ImmediateContext* ctx) {
    if (g.mainOpen)
      CloseMainPass(ctx);
  }


  void BlessedHalfRate::OnPresent() {
    HalfRateConfig::Get(); // logs the configuration once

    PollSwitch();

    // a main pass still open at present never closed through a draw,
    // clear or dispatch: no capture this frame
    g.mainOpen = false;
    g.mainSeen = false;
    g.frameOff = false;
    g.targets  = BlessedHalfRateTargets();

    g.stats.frames++;
    if (g.live)
      g.stats.liveFrames++;

    g.frame++;

    if (!g.live) {
      // the layer is stale once the switch is back on
      g.lastCaptureFrame = 0;
    }

    // views die with their frames' owners; relearn now and then
    if ((g.frame % 600u) == 0u) {
      g.vsMemo.clear();
      g.omGeneration = ~0ull;
    }

    UpdateWatch();

    if (g.stats.frames >= 120u)
      WriteLog();
  }

}
