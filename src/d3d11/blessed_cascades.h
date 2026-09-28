// blessed: BLESSED_SKIP_CASCADES -- skip skyrim's raster sun shadow cascades once the ray-traced mask replaces them
#pragma once

#include <cstdint>

#include "d3d11_context_state.h"

namespace dxvk {

  class DxvkImage;

  // blessed: backing flag for BlessedCascadeSkip::IsEnabled(), split out so
  // the hot-path gate is a trivial inline read -- same pattern as
  // blessed_dump_detail::g_enabled in blessed_dump.h. Set once at
  // static-init time (dll load) from BLESSED_SKIP_CASCADES/BLESSED_HOOK_MODE;
  // never changes after.
  namespace blessed_cascades_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief Learns the sun shadow cascade depth targets and skips draws/clears into them
   *
   * Once the ray-traced shadow mask (BLESSED_HOOK_MODE=rtshadow) has
   * overwritten skyrim's sun shadow mask, the raster cascades that used to
   * feed it are wasted work: about half this frame's draws
   * (docs/research/whiterun-frame.md, passes 7-15, 1,978 of 3,841 draws).
   * BLESSED_SKIP_CASCADES=1 learns which depth images those cascades
   * render into -- by watching the mask draw's own bound ps
   * shader-resource views, exactly, not by size or format -- and from the
   * next frame on, skips any draw or ClearDepthStencilView that targets
   * one of them with no render target bound.
   *
   * Single-threaded by construction: every entry point here is only ever
   * called from the d3d11 immediate context's app thread (never a deferred
   * context, never the cs thread), always under the D3D10DeviceLock its
   * caller already holds -- the same convention BlessedSceneCapture's
   * swapchain-extent cache documents. No locking of its own.
   *
   * Everything here is cheap when disabled: IsEnabled() is a cached bool,
   * and the per-draw checks short-circuit immediately when the learned set
   * is empty (e.g. before the mask has ever matched).
   */
  class BlessedCascadeSkip {
  public:

    // True once BLESSED_SKIP_CASCADES=1 and BLESSED_HOOK_MODE=rtshadow.
    // Inline read of a plain global set once at load -- no out-of-line call,
    // no magic-static guard (see blessed_cascades_detail::g_enabled above).
    static bool IsEnabled() {
      return blessed_cascades_detail::g_enabled;
    }

    // Called from BlessedShadow::OnDraw, once per matched mask draw:
    // (re)records the images bound at the ps srv slots named by
    // BLESSED_SKIP_CASCADES_SRV (default "4,6") as this frame's cascade
    // shadow maps, replacing whatever was learned last frame.
    static void LearnFromMaskDraw(const D3D11ContextState& state);

    // True if this draw has no rtv bound, has a dsv bound, and that dsv's
    // image is a currently-learned cascade shadow map. Called from every
    // Draw/DrawIndexed/DrawInstanced on the immediate context regardless of
    // whether the feature is enabled, so the disabled case must never reach
    // an out-of-line call: inlined here, same global as IsEnabled().
    static bool ShouldSkipDraw(const D3D11ContextState& state) {
      return blessed_cascades_detail::g_enabled && ShouldSkipDrawSlow(state);
    }

    // True if `dsv`'s image is a currently-learned cascade shadow map. Same
    // hot-path reasoning as ShouldSkipDraw above (called from every
    // ClearDepthStencilView).
    static bool ShouldSkipClear(D3D11DepthStencilView* dsv) {
      return blessed_cascades_detail::g_enabled && ShouldSkipClearSlow(dsv);
    }

    static void RecordSkippedDraw();
    static void RecordSkippedClear();

    // D3D11SwapChain::Present calls this once per present, before any draw
    // of the next frame can run -- a changed width/height clears the
    // learned set (whatever the old images were, they're gone; the mask
    // draw re-learns fresh ones on its next match).
    static void NotifySwapchainExtent(uint32_t width, uint32_t height);

    // Writes <BLESSED_PROBE_DIR>/cascades.jsonl every 120 presents.
    static void OnPresent();

  private:

    // out-of-line: the actual learned-set lookup, only ever reached once
    // g_enabled is already known true.
    static bool ShouldSkipDrawSlow(const D3D11ContextState& state);
    static bool ShouldSkipClearSlow(D3D11DepthStencilView* dsv);

  };

}
