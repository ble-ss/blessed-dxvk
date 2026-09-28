// blessed: app-thread half of BLESSED_AO=rt -- finds the sao composite draw and hands the ao dispatch to the cs thread
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;

  // blessed: backing flag for BlessedAo::IsEnabled(), set once at static
  // init from BLESSED_AO alone (same pattern as blessed_hook_detail)
  namespace blessed_ao_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief Ray-traced ao written into the texture skyrim's sao composite reads
   *
   * BLESSED_AO=rt turns it on (unset: one inline bool read per draw).
   * Pre-draw hook: right before the draw whose pixel shader matches
   * BLESSED_AO_PS (default `ddcce9bc`, ISSAOComposite, pass 127 of the
   * whiterun dump), a compute pass overwrites the x channel of the texture
   * bound at that draw's ps srv BLESSED_AO_SRV (default 1, vanilla's
   * blurred sao, rgba8) with traced ao. Depth comes from the same draw's
   * ps srv BLESSED_AO_DEPTH_SRV (default 2). No shader edits; vanilla's sao
   * still runs and is overwritten, never stacked.
   *
   * Knobs: BLESSED_AO_RAYS (2), BLESSED_AO_RADIUS (64 units),
   * BLESSED_AO_STRENGTH (1.0), BLESSED_AO_HALF (1 = half-res trace),
   * BLESSED_AO_DEBUG (black, white, raw, noblur, hist, normal, reproj),
   * BLESSED_AO_INVVP / BLESSED_AO_CAMPOS ("ps:12:512" / "ps:12:640"),
   * BLESSED_AO_FAR (1; 0 for reversed z).
   */
  class BlessedAo {
  public:

    static bool IsEnabled() {
      return blessed_ao_detail::g_enabled;
    }

    // Called from the immediate context right before a draw is recorded.
    static void OnDrawPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

  };

}
