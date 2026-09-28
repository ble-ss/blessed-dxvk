// blessed: BLESSED_VOLUMETRICS -- app-thread hooks on skyrim's volumetric-light pass (138) and the sun mask draw
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;
  struct BlessedVolDispatchArgs;

  // blessed: backing flag for BlessedVolumetrics::IsEnabled(), set once at
  // static init from BLESSED_VOLUMETRICS alone (same pattern as
  // blessed_hook_detail::g_enabled)
  namespace blessed_vol_detail {
    extern const bool g_enabled;
    // blessed: async-compute -- BLESSED_VOLUMETRICS=rt with BLESSED_ASYNC=1
    // and pass "vol": the pre-draw kick hook is live
    extern const bool g_asyncKick;
  }

  /**
   * \brief Ray-traced volumetric sun light, and the dumps that prove its hooks
   *
   * BLESSED_VOLUMETRICS selects the mode (unset or "off": disabled, one
   * inline bool per draw):
   *  - "vanilla": hooks active, pass 138 untouched; only BLESSED_VOL_DUMP.
   *  - "fill":    clear pass 138's target to BLESSED_VOL_FILL after it draws
   *               (forced output: proves what pass 167 composites).
   *  - "rt":      replace pass 138's output with our traced in-scatter.
   *
   * Two draws are watched, by pixel shader hash: pass 138
   * (ISApplyVolumetricLighting, BLESSED_VOL_PS, default c480e36e) and the
   * sun shadow-mask draw (BLESSED_VOL_SUN_PS, default b070feb5), where the
   * sun direction (and by default the camera) is read, as the shadow pass
   * reads them. Independent of BlessedHook, so it runs next to rtshadow.
   */
  class BlessedVolumetrics {
  public:

    static bool IsEnabled() {
      return blessed_vol_detail::g_enabled;
    }

    // Post-draw, immediate context only.
    static void OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

    // blessed: vol-2 -- called instead of the draw when BlessedSkipVolumetrics
    // drops pass 138 (BLESSED_VOL_SKIP_VANILLA=1): runs the same hook, and
    // clears the target whenever the trace cannot run, since nothing else
    // wrote it this frame.
    static void OnSkippedDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

    // blessed: async-compute -- cached bool, see g_asyncKick
    static bool IsAsyncKickEnabled() {
      return blessed_vol_detail::g_asyncKick;
    }

    // blessed: async-compute -- pre-draw, immediate context only. At the
    // kick draw (BLESSED_ASYNC_VOL_KICK_PS, default ddcce9bc: the sao
    // composite, pass 127, where depth is final) emits the trace half for
    // the async queue; pass 138 then only upsamples.
    static void OnDrawPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

    // blessed: async-compute -- frame boundary, app thread
    static void OnPresent();

    // Hands one dispatch to the cs thread (a friend of the context, for EmitCs).
    static void Emit(D3D11ImmediateContext* ctx, BlessedVolDispatchArgs&& args);

  };

}
