// blessed: fe-getters -- two small diagnostics for the threaded front end's
// getters, both independent of the render-thread probe (BLESSED_PROBE_DIR)
// so they work in a noprobe build too:
//
//  - BLESSED_FE_DRAIN_STATS=1: a per-reason and per-call drain counter,
//    logged every ~5s. Answers "what still drains, and how often" once the
//    OM shadow (or a later one) removes the obvious cases.
//  - BLESSED_FE_SHADOW_VERIFY=1: on every getter answered from a shadow
//    instead of a drain, also drains and compares against the real
//    context's answer, logging the first mismatches. Meant to run once in
//    game before trusting a shadow for timing; it drains every shadowed
//    getter on purpose, so it costs real frame time while it is on.
//
// Both read their env var once, lazily, on first use; both are one relaxed
// bool check when their var is unset, no globals touched otherwise.
#pragma once

#include <cstdint>

#include "../util/util_blessed_fe_census.h" // FeCall, FeDrain

struct ID3D11RenderTargetView;
struct ID3D11DepthStencilView;
struct ID3D11UnorderedAccessView;

namespace dxvk::blessed {

  // --- drain stats ---

  bool FeDrainStatsEnabled();

  // Call once per Drain(), from the game thread only (same rule as every
  // other front-end counter: no synchronization needed). Call may be
  // FeCall::Count when no single entry point owns the drain (shutdown).
  void FeDrainStatsTick(FeDrain Reason, FeCall Call);


  // --- shadow verify ---

  bool FeShadowVerifyEnabled();

  // pShadow/pReal may be null (an unrequested output); Slot is the RTV or
  // UAV array index for logging, ignored for the DSV call.
  void FeShadowVerifyRtv(FeCall Call, uint32_t Slot,
    ID3D11RenderTargetView* pShadow, ID3D11RenderTargetView* pReal);
  void FeShadowVerifyDsv(FeCall Call,
    ID3D11DepthStencilView* pShadow, ID3D11DepthStencilView* pReal);
  void FeShadowVerifyUav(FeCall Call, uint32_t Slot,
    ID3D11UnorderedAccessView* pShadow, ID3D11UnorderedAccessView* pReal);

  // Call once per verified getter call (after the Rtv/Dsv/Uav checks above),
  // so the periodic summary can report calls compared, not just mismatches.
  void FeShadowVerifyCallDone();

}
