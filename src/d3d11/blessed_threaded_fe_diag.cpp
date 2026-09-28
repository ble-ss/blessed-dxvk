// blessed: fe-getters -- see blessed_threaded_fe_diag.h
#include "blessed_threaded_fe_diag.h"

#include <array>
#include <chrono>

#include "../util/log/log.h"
#include "../util/util_env.h"
#include "../util/util_string.h"

// Own name tables, independent of util_blessed_probe.cpp's (which do not
// exist in a noprobe build): both are built from the same X-macro lists in
// util_blessed_fe_census.h, so they never drift from FeCall/FeDrain.
namespace {

  const char* const g_callNames[uint32_t(dxvk::blessed::FeCall::Count)] = {
#define BLESSED_FE_DIAG_NAME(name) #name,
    BLESSED_FE_CALLS(BLESSED_FE_DIAG_NAME)
  };

  const char* const g_drainNames[uint32_t(dxvk::blessed::FeDrain::Count)] = {
    BLESSED_FE_DRAINS(BLESSED_FE_DIAG_NAME)
#undef BLESSED_FE_DIAG_NAME
  };

  constexpr auto kFlushEvery = std::chrono::seconds(5);

}

namespace dxvk::blessed {

  // --- drain stats ---

  bool FeDrainStatsEnabled() {
    static const bool s_enabled = [] {
      std::string v = env::getEnvVar("BLESSED_FE_DRAIN_STATS");
      bool on = !v.empty() && v != "0";

      if (on)
        Logger::info("blessed_fe_drain_stats: on, a line every 5s");

      return on;
    }();
    return s_enabled;
  }


  namespace {
    // Game thread only, same as every other front-end counter here: the
    // producer is always one thread at a time (see D3D11ThreadedContext's
    // own header comment on that rule), so plain increments are enough.
    struct DrainStatsState {
      std::array<uint64_t, uint32_t(FeDrain::Count)> byReason = { };
      std::array<uint64_t, uint32_t(FeCall::Count)>  byCall   = { };
      std::chrono::steady_clock::time_point           windowStart;
      bool                                             haveStart = false;
    };

    DrainStatsState g_drainStats;

    void FlushDrainStatsIfDue() {
      auto now = std::chrono::steady_clock::now();

      if (!g_drainStats.haveStart) {
        g_drainStats.windowStart = now;
        g_drainStats.haveStart   = true;
        return;
      }

      if (now - g_drainStats.windowStart < kFlushEvery)
        return;

      std::string line = "blessed_fe_drain_stats: reasons={";
      bool first = true;

      for (uint32_t i = 0; i < uint32_t(FeDrain::Count); i++) {
        if (!g_drainStats.byReason[i])
          continue;

        line += str::format(first ? "" : ",", g_drainNames[i], ":", g_drainStats.byReason[i]);
        first = false;
      }

      line += "} calls={";
      first = true;

      for (uint32_t i = 0; i < uint32_t(FeCall::Count); i++) {
        if (!g_drainStats.byCall[i])
          continue;

        line += str::format(first ? "" : ",", g_callNames[i], ":", g_drainStats.byCall[i]);
        first = false;
      }

      line += "}";

      Logger::info(line);

      g_drainStats.byReason.fill(0u);
      g_drainStats.byCall.fill(0u);
      g_drainStats.windowStart = now;
    }
  }


  void FeDrainStatsTick(FeDrain Reason, FeCall Call) {
    if (likely(!FeDrainStatsEnabled()))
      return;

    g_drainStats.byReason[uint32_t(Reason)] += 1u;

    if (Call != FeCall::Count)
      g_drainStats.byCall[uint32_t(Call)] += 1u;

    FlushDrainStatsIfDue();
  }


  // --- shadow verify ---

  bool FeShadowVerifyEnabled() {
    static const bool s_enabled = [] {
      std::string v = env::getEnvVar("BLESSED_FE_SHADOW_VERIFY");
      bool on = !v.empty() && v != "0";

      if (on)
        Logger::info("blessed_fe_shadow_verify: on, every shadowed getter also drains and compares");

      return on;
    }();
    return s_enabled;
  }


  namespace {
    // Caps how much we ever log: this mode is meant to run briefly, watched,
    // not to fill a log file if something is actually wrong.
    constexpr uint32_t kMaxLoggedMismatches = 64u;

    struct ShadowVerifyState {
      uint64_t calls      = 0;
      uint64_t compared   = 0;
      uint64_t mismatches = 0;
      bool     capHit      = false;
      std::chrono::steady_clock::time_point windowStart;
      bool     haveStart   = false;
    };

    ShadowVerifyState g_verify;

    void NoteMismatch(FeCall Call, const char* What, uint32_t Slot, const void* Shadow, const void* Real) {
      g_verify.mismatches += 1u;

      if (g_verify.mismatches > kMaxLoggedMismatches) {
        if (!g_verify.capHit) {
          Logger::warn("blessed_fe_shadow_verify: further mismatches suppressed (cap reached)");
          g_verify.capHit = true;
        }
        return;
      }

      Logger::warn(str::format("blessed_fe_shadow_verify: mismatch call=", g_callNames[uint32_t(Call)],
        " ", What, " slot=", Slot, " shadow=", Shadow, " real=", Real));
    }

    void FlushShadowVerifyIfDue() {
      auto now = std::chrono::steady_clock::now();

      if (!g_verify.haveStart) {
        g_verify.windowStart = now;
        g_verify.haveStart    = true;
        return;
      }

      if (now - g_verify.windowStart < kFlushEvery)
        return;

      Logger::info(str::format("blessed_fe_shadow_verify: calls=", g_verify.calls,
        " compared=", g_verify.compared, " mismatches=", g_verify.mismatches));

      g_verify.windowStart = now;
    }
  }


  void FeShadowVerifyRtv(FeCall Call, uint32_t Slot, ID3D11RenderTargetView* pShadow, ID3D11RenderTargetView* pReal) {
    if (likely(!FeShadowVerifyEnabled()))
      return;

    g_verify.compared += 1u;

    if (pShadow != pReal)
      NoteMismatch(Call, "rtv", Slot, pShadow, pReal);
  }


  void FeShadowVerifyDsv(FeCall Call, ID3D11DepthStencilView* pShadow, ID3D11DepthStencilView* pReal) {
    if (likely(!FeShadowVerifyEnabled()))
      return;

    g_verify.compared += 1u;

    if (pShadow != pReal)
      NoteMismatch(Call, "dsv", 0u, pShadow, pReal);
  }


  void FeShadowVerifyUav(FeCall Call, uint32_t Slot, ID3D11UnorderedAccessView* pShadow, ID3D11UnorderedAccessView* pReal) {
    if (likely(!FeShadowVerifyEnabled()))
      return;

    g_verify.compared += 1u;

    if (pShadow != pReal)
      NoteMismatch(Call, "uav", Slot, pShadow, pReal);
  }


  void FeShadowVerifyCallDone() {
    if (likely(!FeShadowVerifyEnabled()))
      return;

    g_verify.calls += 1u;
    FlushShadowVerifyIfDue();
  }

}
