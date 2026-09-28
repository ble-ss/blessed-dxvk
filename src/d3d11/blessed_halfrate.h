// blessed: half-rate far field -- app-thread half: far-draw skip, main-pass tracking, the live switch
#pragma once

#include <cstdint>

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;
  class D3D11RenderTargetView;
  class D3D11DepthStencilView;
  class DxvkDevice;
  struct BlessedHalfRateArgs;

  // blessed: backing flags for the inline gates below (same pattern as
  // blessed_cascades_detail::g_enabled). g_available is fixed at static
  // init from BLESSED_HALFRATE; g_watch is live: true while this frame
  // still has something to do (the main pass not yet seen or open).
  // Written only at present and at
  // pass boundaries, on the thread that also reads it.
  namespace blessed_halfrate_detail {
    extern const bool g_available;
    extern bool       g_watch;
  }

  /**
   * \brief Half-rate far field (BLESSED_HALFRATE)
   *
   * Design: docs/research/halfrate-far.md in the main repo. On "on"
   * frames everything renders and the finished main lit pass is copied
   * into the far layer. On "off" frames the main lit pass's far draws
   * (distant trees, lod) are skipped and their pixels are prefilled from
   * the layer, reprojected with our own previous camera.
   *
   * The water reflection part moved to BLESSED_REFLECT_HALFRATE
   * (blessed_reflect_halfrate.h); BLESSED_HALFRATE_PARTS=reflect now
   * turns that one on.
   *
   * The switch is live: the skse plugin (skse/skybench) publishes
   * `Local\BlessedHalfRate`, the fork reads it at every present.
   * BLESSED_HALFRATE=0 turns the feature off for the whole process.
   *
   * All entry points: immediate context only.
   */
  class BlessedHalfRate {
  public:

    static bool IsAvailable() {
      return blessed_halfrate_detail::g_available;
    }

    /// One predictable branch per draw: false whenever there is nothing to do
    static bool IsWatching() {
      return blessed_halfrate_detail::g_watch;
    }

    /// Before a draw. True: drop this draw.
    static bool OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device);

    /// Before a clear: closes the main pass if it is open. Never drops the clear.
    static bool OnClearRtv(D3D11ImmediateContext* ctx, D3D11RenderTargetView* rtv);
    static bool OnClearDsv(D3D11ImmediateContext* ctx, D3D11DepthStencilView* dsv);

    /// Before a dispatch: closes the main pass if it is open
    static void OnDispatch(D3D11ImmediateContext* ctx);

    /// Frame boundary: reads the live switch, decides the next frame,
    /// writes <BLESSED_PROBE_DIR>/halfrate.jsonl every 120 presents
    static void OnPresent();

  private:

    // members, not free functions: they need the context's EmitCs,
    // RestoreCommandListState and default states (friend access)
    static void OpenMainPass(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device);
    static void CloseMainPass(D3D11ImmediateContext* ctx);
    static void BuildStates(BlessedHalfRateArgs& args);

  };

}
