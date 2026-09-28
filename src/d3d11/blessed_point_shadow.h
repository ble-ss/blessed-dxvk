// blessed: point-light shadow mask draws -- recognise them, trace them (tier a), gate the sun hook, skip their raster
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;
  struct BlessedPointShadowDispatchArgs;

  // blessed: backing flags for the inline gates below, set once at static
  // init from env vars alone (same pattern as blessed_hook_detail::g_enabled).
  //  - g_enabled: something here needs to see every draw -- the point trace
  //    (BLESSED_POINT_SHADOWS=1), or the sun hook's interior gate, which
  //    watches the point mask draws for a claim on channel r (on whenever
  //    BLESSED_HOOK_MODE=rtshadow, unless BLESSED_SHADOW_SUN_GATE=0).
  //  - g_skip: BLESSED_POINT_SKIP_RASTER=1 (requires BLESSED_POINT_SHADOWS=1)
  namespace blessed_point_detail {
    extern const bool g_enabled;
    extern const bool g_skip;
  }

  /**
   * \brief Skyrim's point-light shadow mask draws, and what we do at them
   *
   * The engine shadows at most 4 point lights per frame, one per channel of
   * the same rgba8 shadow mask the sun writes. Each one is a full-screen
   * draw with one of the vanilla utility pixel shaders RENDER_SHADOWMASKSPOT
   * / PB / DPB, the channel picked by the blend state's write mask. See
   * docs/research/point-lights.md.
   *
   * Tier a: at each such draw, read the light's camera-relative origin out
   * of ShadowMapProj[0] (ps b2 c16-c19) and its radius out of
   * ShadowLightParam.x (ps b2 c3), and trace one ray per pixel toward the
   * light, writing the draw's own channel(s).
   *
   * Every entry point is a no-op behind one cached bool unless enabled.
   */
  class BlessedPointShadow {
  public:

    static bool IsEnabled() {
      return blessed_point_detail::g_enabled;
    }

    // Pre-draw, immediate context only. True: drop this draw. Drops the
    // point-light shadow-map depth draws learned from the mask draws' srvs
    // (per image and slice), and the point mask draws themselves -- in which
    // case the trace is dispatched right here, in the draw's place.
    static bool ShouldSkipDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
      return blessed_point_detail::g_skip && ShouldSkipDrawSlow(ctx, state);
    }

    // Post-draw, immediate context only: classifies the draw and, for a
    // point mask draw, records its channel claim and dispatches the trace.
    static void OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

    // Frame boundary: rolls the channel claims and the pointshadow.jsonl window.
    static void OnPresent();

    // The sun hook's interior gate (BlessedShadow::OnDraw). True: this draw
    // really is the sun's mask draw and channel r is the sun's to write --
    // ps b2 is 400 bytes (a point variant's is 448), the write mask
    // includes r, and no point mask draw claimed r last frame or earlier
    // this frame. Always true with BLESSED_SHADOW_SUN_GATE=0.
    static bool SunMayWriteR(const D3D11ContextState& state);

    // Hands one light's dispatch to the cs thread (needs the context's
    // EmitCs, hence a member: D3D11CommonContext befriends this class).
    static void EmitDispatch(D3D11ImmediateContext* ctx, BlessedPointShadowDispatchArgs&& args);

  private:

    static bool ShouldSkipDrawSlow(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

  };

}
