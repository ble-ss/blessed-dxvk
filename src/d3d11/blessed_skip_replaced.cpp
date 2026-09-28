// blessed: gpu track step 2 -- see blessed_skip_replaced.h
#include "blessed_skip_replaced.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d11_context_imm.h"
#include "d3d11_shader.h"
#include "d3d11_blend.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // blessed: shared with BlessedAo/BlessedHook's own copy of this same
    // idea -- a csv of bare 8+ char hex prefixes (no "fs."/"cs." needed,
    // this module's env vars only ever hold shader hashes).
    std::vector<std::string> SplitHexTokens(const std::string& csv, const char* fallback) {
      std::vector<std::string> out;
      const std::string& s = csv.empty() ? std::string(fallback) : csv;

      size_t pos = 0;
      while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? s.size() : comma + 1;

        if (tok.rfind("fs.", 0) == 0 || tok.rfind("cs.", 0) == 0)
          tok = tok.substr(3);
        if (tok.size() >= 8)
          out.push_back(std::move(tok));
      }

      return out;
    }

    bool NameMatchesTokens(const std::string& name, const std::vector<std::string>& tokens) {
      // name is "fs.<hash>..." or "cs.<hash>...": the hash starts at index 3
      if (name.size() < 11)
        return false;
      for (const auto& tok : tokens) {
        if (name.compare(3, tok.size(), tok) == 0)
          return true;
      }
      return false;
    }

    // blessed: true if the state's rtv slot 0 blend write mask is R only
    // (D3D11_COLOR_WRITE_ENABLE_RED, 0x1) -- the same test the whiterun
    // dump's b070feb5 draw shows (write_mask: 1, no other channel, no dsv).
    // A null blend state is D3D11's default (all channels enabled on every
    // target), so it correctly reads as "not R-only" here.
    bool Rtv0WriteMaskIsRedOnly(const D3D11ContextState& state) {
      D3D11BlendState* bs = state.om.cbState.ptr();
      if (!bs)
        return false;

      const D3D11_BLEND_DESC1& desc = bs->Desc();
      UINT8 mask = desc.RenderTarget[0].RenderTargetWriteMask;
      return mask == D3D11_COLOR_WRITE_ENABLE_RED;
    }

    // ---- per-feature periodic jsonl writer, one file each, 120-present window ----
    struct SkipLog {
      dxvk::high_resolution_clock::time_point processStart;
      bool                                    processStartSet = false;
      uint64_t                                presentsInWindow = 0;
      uint64_t                                skipped          = 0;
      uint64_t                                skipped2         = 0; // second counter, e.g. draw vs dispatch
      std::ofstream                           file;
      bool                                    fileTried        = false;
    };

    void TickLog(SkipLog& log, const char* fileName, const char* key1, const char* key2 = nullptr) {
      if (!log.processStartSet) {
        log.processStartSet = true;
        log.processStart = dxvk::high_resolution_clock::now();
      }

      if (++log.presentsInWindow < 120)
        return;

      if (!log.fileTried) {
        log.fileTried = true;
        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

        if (!dir.empty()) {
          std::error_code ec;
          std::filesystem::create_directories(dir, ec);
          log.file.open(dir + env::PlatformDirSlash + fileName, std::ios::out | std::ios::app);
        }
      }

      double t = std::chrono::duration<double>(
        dxvk::high_resolution_clock::now() - log.processStart).count();

      double perFrame  = double(log.skipped)  / double(log.presentsInWindow);
      double perFrame2 = double(log.skipped2) / double(log.presentsInWindow);

      if (log.file.is_open()) {
        log.file << str::format("{\"t\":", t,
          ",\"frames\":", log.presentsInWindow,
          ",\"", key1, "\":", perFrame);
        if (key2)
          log.file << str::format(",\"", key2, "\":", perFrame2);
        log.file << "}\n";
        log.file.flush();
      }

      log.presentsInWindow = 0;
      log.skipped  = 0;
      log.skipped2 = 0;
    }

    // ============================== ao ==============================

    struct AoSkipConfig {
      std::vector<std::string> psTokens;

      static const AoSkipConfig& Get() {
        static const AoSkipConfig s_config = [] {
          AoSkipConfig c;
          c.psTokens = SplitHexTokens(env::getEnvVar("BLESSED_AO_SKIP_PS"),
            "83560015,aebce6eb,e205894f");
          Logger::info(str::format("BlessedSkipAo: enabled, ",
            c.psTokens.size(), " ps hash(es)"));
          return c;
        }();
        return s_config;
      }
    };

    std::unordered_map<const void*, bool> g_aoMatchCache;
    SkipLog g_aoLog;

    // ============================ volumetrics ============================

    struct VolSkipConfig {
      std::vector<std::string> pass138PsTokens;

      static const VolSkipConfig& Get() {
        static const VolSkipConfig s_config = [] {
          VolSkipConfig c;
          c.pass138PsTokens = SplitHexTokens(env::getEnvVar("BLESSED_VOL_SKIP_PS138"), "c480e36e");
          Logger::info(str::format("BlessedSkipVolumetrics: draw skip enabled, ",
            c.pass138PsTokens.size(), " ps hash(es)"));
          return c;
        }();
        return s_config;
      }
    };

    struct VolRaymarchConfig {
      std::vector<std::string> csTokens;

      static const VolRaymarchConfig& Get() {
        static const VolRaymarchConfig s_config = [] {
          VolRaymarchConfig c;
          c.csTokens = SplitHexTokens(env::getEnvVar("BLESSED_VOL_SKIP_RAYMARCH_CS"), "ab674eb1,1c4ebb62");
          Logger::info(str::format("BlessedSkipVolumetrics: raymarch skip enabled (BLESSED_VOL_SKIP_RAYMARCH=1), ",
            c.csTokens.size(), " cs hash(es) -- verify the lens-flare consumer in game before trusting this"));
          return c;
        }();
        return s_config;
      }
    };

    std::unordered_map<const void*, bool> g_volDrawMatchCache;
    std::unordered_map<const void*, bool> g_volDispatchMatchCache;
    SkipLog g_volLog;

    // ============================== bloom ==============================

    struct BloomSkipConfig {
      std::vector<std::string> psTokens;

      static const BloomSkipConfig& Get() {
        static const BloomSkipConfig s_config = [] {
          BloomSkipConfig c;
          c.psTokens = SplitHexTokens(env::getEnvVar("BLESSED_LOOK_SKIP_BLOOM_PS"), "cda7dc03");
          Logger::info(str::format("BlessedSkipBloom: enabled, ",
            c.psTokens.size(), " ps hash(es)"));
          return c;
        }();
        return s_config;
      }
    };

    std::unordered_map<const void*, bool> g_bloomMatchCache;
    SkipLog g_bloomLog;

    // ============================ shadow mask ============================

    struct ShadowMaskSkipConfig {
      std::vector<std::string> psTokens;

      static const ShadowMaskSkipConfig& Get() {
        static const ShadowMaskSkipConfig s_config = [] {
          ShadowMaskSkipConfig c;
          c.psTokens = SplitHexTokens(env::getEnvVar("BLESSED_SHADOW_SKIP_MASK_PS"), "b070feb5");
          Logger::info(str::format("BlessedSkipShadowMask: enabled, ",
            c.psTokens.size(), " ps hash(es)"));
          return c;
        }();
        return s_config;
      }
    };

    std::unordered_map<const void*, bool> g_shadowMaskMatchCache;
    SkipLog g_shadowMaskLog;

    // ---- pure functions of env vars alone, safe at static-init time; see
    // ComputeCascadeEnabled's comment in blessed_cascades.cpp for why this
    // split matters (Logger::info in the *Config::Get() lazies above must
    // never run before Logger's own statics are ready) ----

    bool EnvOff(const char* name) {
      return env::getEnvVar(name) == "0";
    }

    bool EnvOn(const char* name) {
      return env::getEnvVar(name) == "1";
    }

    bool ComputeAoSkipEnabled() {
      return env::getEnvVar("BLESSED_AO") == "rt" && !EnvOff("BLESSED_AO_SKIP_VANILLA");
    }

    bool ComputeVolSkipEnabled() {
      // blessed: opt-in (lead, 2026-09-23): skipping pass 138's draw also skips
      // the post-draw hook our trace runs from (vol-rt-1: the sun haze vanished).
      return env::getEnvVar("BLESSED_VOLUMETRICS") == "rt" && EnvOn("BLESSED_VOL_SKIP_VANILLA");
    }

    bool ComputeVolRaymarchSkipEnabled() {
      // blessed: default OFF -- see blessed_skip_replaced.h's class comment
      // for the verified second consumer (a lens-flare draw) this would
      // starve. Requires the base switch above plus this explicit opt-in.
      return ComputeVolSkipEnabled() && EnvOn("BLESSED_VOL_SKIP_RAYMARCH");
    }

    bool ComputeBloomSkipEnabled() {
      // blessed: BLESSED_LOOK_VANILLA_BLOOM defaults to 0 (zeroed) in
      // BlessedLook itself -- see EnvBool's default in blessed_look.cpp.
      // Mirrored here rather than reached into BlessedLook's own config,
      // same as every other blessed_* file reading its own relevant env
      // vars independently.
      bool vanillaBloomZeroed = !EnvOn("BLESSED_LOOK_VANILLA_BLOOM");
      return env::getEnvVar("BLESSED_LOOK") == "bless"
          && vanillaBloomZeroed
          && !EnvOff("BLESSED_LOOK_SKIP_BLOOM");
    }

    bool ComputeShadowMaskSkipEnabled() {
      // blessed: skip-more -- was opt-in only (lead, 2026-09-23): skipping the
      // raster mask draw lost the traced shadows in game (shadowreg-fixed-1).
      // Root cause found: the compute dispatch's output barrier only named
      // COLOR_ATTACHMENT_OUTPUT as its source scope, so once the raster draw
      // stopped running the barrier no longer synchronized against our own
      // previous compute write -- a WAW race, not a target-selection bug (the
      // target itself was already read from m_state's bound rtv, same as
      // BlessedVolumetrics::OnSkippedDraw). Fixed in blessed_shadow.cpp's
      // dispatch() (widened to also name COMPUTE_SHADER_BIT, the same fix
      // vol-2 made for pass 138's output and point-lights already carries).
      // Default on now; worth ~0.2 ms. Opt out with BLESSED_SHADOW_SKIP_MASK_DRAW=0.
      return env::getEnvVar("BLESSED_HOOK_MODE") == "rtshadow" && !EnvOff("BLESSED_SHADOW_SKIP_MASK_DRAW");
    }

  }


  namespace blessed_skip_replaced_detail {
    extern const bool g_aoEnabled          = ComputeAoSkipEnabled();
    extern const bool g_volDrawEnabled     = ComputeVolSkipEnabled();
    extern const bool g_volDispatchEnabled = ComputeVolRaymarchSkipEnabled();
    extern const bool g_bloomEnabled       = ComputeBloomSkipEnabled();
    extern const bool g_shadowMaskEnabled  = ComputeShadowMaskSkipEnabled();
  }


  // ============================== ao ==============================

  bool BlessedSkipAo::ShouldSkipDrawSlow(const D3D11ContextState& state) {
    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps)
      return false;

    auto it = g_aoMatchCache.find(ps);
    if (it != g_aoMatchCache.end())
      return it->second;

    bool match = NameMatchesTokens(ps->GetCommonShader()->GetName(), AoSkipConfig::Get().psTokens);
    g_aoMatchCache.emplace(ps, match);
    return match;
  }

  void BlessedSkipAo::RecordSkipped() {
    g_aoLog.skipped++;
  }

  void BlessedSkipAo::OnPresent() {
    if (!IsEnabled())
      return;
    TickLog(g_aoLog, "ao_skip.jsonl", "skipped_draws");
  }


  // ============================ volumetrics ============================

  bool BlessedSkipVolumetrics::ShouldSkipDrawSlow(const D3D11ContextState& state) {
    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps)
      return false;

    auto it = g_volDrawMatchCache.find(ps);
    if (it != g_volDrawMatchCache.end())
      return it->second;

    bool match = NameMatchesTokens(ps->GetCommonShader()->GetName(), VolSkipConfig::Get().pass138PsTokens);
    g_volDrawMatchCache.emplace(ps, match);
    return match;
  }

  bool BlessedSkipVolumetrics::ShouldSkipDispatchSlow(const D3D11ContextState& state) {
    D3D11ComputeShader* cs = state.cs.ptr();
    if (!cs)
      return false;

    auto it = g_volDispatchMatchCache.find(cs);
    if (it != g_volDispatchMatchCache.end())
      return it->second;

    bool match = NameMatchesTokens(cs->GetCommonShader()->GetName(), VolRaymarchConfig::Get().csTokens);
    g_volDispatchMatchCache.emplace(cs, match);
    return match;
  }

  void BlessedSkipVolumetrics::RecordSkippedDraw() {
    g_volLog.skipped++;
  }

  void BlessedSkipVolumetrics::RecordSkippedDispatch() {
    g_volLog.skipped2++;
  }

  void BlessedSkipVolumetrics::OnPresent() {
    if (!IsDrawSkipEnabled() && !IsDispatchSkipEnabled())
      return;
    TickLog(g_volLog, "volumetrics_skip.jsonl", "skipped_draws", "skipped_dispatches");
  }


  // ============================== bloom ==============================

  bool BlessedSkipBloom::ShouldSkipDrawSlow(const D3D11ContextState& state) {
    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps)
      return false;

    auto it = g_bloomMatchCache.find(ps);
    if (it != g_bloomMatchCache.end())
      return it->second;

    bool match = NameMatchesTokens(ps->GetCommonShader()->GetName(), BloomSkipConfig::Get().psTokens);
    g_bloomMatchCache.emplace(ps, match);
    return match;
  }

  void BlessedSkipBloom::RecordSkipped() {
    g_bloomLog.skipped++;
  }

  void BlessedSkipBloom::OnPresent() {
    if (!IsEnabled())
      return;
    TickLog(g_bloomLog, "bloom_skip.jsonl", "skipped_draws");
  }


  // ============================ shadow mask ============================

  bool BlessedSkipShadowMask::ShouldSkipDrawSlow(const D3D11ContextState& state) {
    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps)
      return false;

    auto it = g_shadowMaskMatchCache.find(ps);
    bool nameMatch;
    if (it != g_shadowMaskMatchCache.end()) {
      nameMatch = it->second;
    } else {
      nameMatch = NameMatchesTokens(ps->GetCommonShader()->GetName(), ShadowMaskSkipConfig::Get().psTokens);
      g_shadowMaskMatchCache.emplace(ps, nameMatch);
    }

    if (!nameMatch)
      return false;

    // blessed: only when the draw writes r alone -- the write-mask check is
    // per-draw, not cached, since blend state can legitimately change
    // between calls to the same shader.
    return Rtv0WriteMaskIsRedOnly(state);
  }

  void BlessedSkipShadowMask::RecordSkipped() {
    g_shadowMaskLog.skipped++;
  }

  void BlessedSkipShadowMask::OnPresent() {
    if (!IsEnabled())
      return;
    TickLog(g_shadowMaskLog, "shadow_mask_skip.jsonl", "skipped_draws");
  }

}
