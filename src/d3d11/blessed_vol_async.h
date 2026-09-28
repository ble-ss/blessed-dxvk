// blessed: BLESSED_VOL_ASYNC -- app-thread hooks moving vanilla's own volumetric compute onto the async queue
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;

  // blessed: backing flag for BlessedVolAsync::IsEnabled(), set once at
  // static init from BLESSED_VOL_ASYNC alone (same pattern as
  // blessed_hook_detail::g_enabled)
  namespace blessed_vol_async_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief Vanilla's own volumetric compute chain on the async queue (BLESSED_VOL_ASYNC)
   *
   * The whiterun dump (bench/runs/whiterun-dump-1, frame 1920) shows the
   * froxel generate (cs `ab674eb1`) and its z-integration chain (cs
   * `1c4ebb62` x90, or BLESSED_VOL_COLLAPSE's one replacement dispatch)
   * running back to back with nothing else in between: passes 17-107 are
   * dispatch-only, no draws. Pass 138's draw (ps `c480e36e`,
   * BLESSED_VOL_ASYNC_WAIT_PS) is the first later reader of the froxel
   * volume (ps srv slot 2), ~1 ms of raster later (the depth prepass, the
   * sun and point shadow masks, the lit pass, sao). BLESSED_VOL_ASYNC=1
   * records generate-through-chain-end on the async queue instead, and
   * pass 138 waits before it draws.
   *
   * (The two blur dispatches, `cf1e1a21`/`dbb99d2e`, are NOT moved here:
   * the same dump shows they run right *after* pass 138's draw this frame,
   * writing the history buffer pass 138 reads back only next frame -- a
   * window of nearly a whole frame already, not the tight one this seat
   * was asked to close. Left on graphics; a separate, lower-risk follow-up.)
   *
   * The window opens at the generate dispatch (BLESSED_VOL_ASYNC_GEN_CS)
   * and closes at the very next draw call, whatever it is, on every draw
   * entry point: only compute-legal work may be recorded while the async
   * queue's command buffer is active, so a draw must never be allowed to
   * slip in. This does not need to know whether BLESSED_VOL_COLLAPSE is
   * on: 1 dispatch or 90, all of them are equally safe to move, since
   * nothing but dispatches happen in that stretch.
   *
   * Same-family queue only (BLESSED_ASYNC_QUEUE=graphics, forced whenever
   * BLESSED_VOL_ASYNC=1: see BlessedAsync::WantsGraphicsFamily). The
   * images this chain touches -- the two froxel volumes, the sun cascade
   * atlas the generate dispatch reads -- are the game's own, persistent,
   * and read by other draws on other frames (the lens-flare draw among
   * them); they were never tagged concurrent across queue families
   * (DxvkImageCreateInfo::blessedConcurrent is opt-in, set only on our own
   * pass-owned scratch images so far), so a genuinely separate compute
   * family would need real queue-ownership transfer barriers this does
   * not perform. A same-family queue needs none (Vulkan: an
   * exclusive-sharing-mode resource may be used from any queue in the
   * same family without a transfer), so vanilla's dispatches run
   * completely unmodified, same data, same bits.
   * DxvkContext::blessedVolAsyncBegin() re-checks this at runtime and
   * stays off if the picked queue is a different family after all.
   */
  class BlessedVolAsync {
  public:

    static bool IsEnabled() {
      return blessed_vol_async_detail::g_enabled;
    }

    // Pre-dispatch, immediate context only: opens the window at the
    // generate dispatch (a cached-bool-guarded name compare, memoized per
    // shader pointer would be overkill here -- one dispatch a frame).
    static void OnDispatchPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

    // Pre-draw, immediate context only, every draw entry point (Draw,
    // DrawIndexed, DrawInstanced, DrawIndexedInstanced, DrawAuto,
    // DrawIndexedInstancedIndirect, DrawInstancedIndirect): closes the
    // window first if still open (a single bool test when it is not; it
    // always is, exactly once, at whatever draw follows the generate
    // dispatch -- normally the depth prepass, long before pass 138), then
    // -- only if this draw's pixel shader matches BLESSED_VOL_ASYNC_WAIT_PS
    // (default c480e36e, pass 138) -- waits for the kick. Self-healing
    // across a frame boundary: if no draw ever followed (should not
    // happen), the bool stays set and the very next frame's first draw
    // closes it there instead.
    static void OnDrawPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

  };

}
