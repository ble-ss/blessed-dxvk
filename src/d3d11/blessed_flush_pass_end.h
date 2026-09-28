// blessed: defer non-synchronisation implicit flushes to the render-pass
// end instead of splitting the pass mid-flight (BLESSED_FLUSH_AT_PASS_END)
#pragma once

#include <cstdint>

namespace dxvk {

  /**
   * \brief Flush-at-pass-end policy
   *
   * Upstream's weak implicit flush hint fires every time a cs chunk fills
   * up, with no idea whether a render pass is currently open. When one
   * lands mid-pass, flushCommandList ends the render pass early (store
   * ops, a barrier set) and the next draw with the same targets reopens
   * it (load ops, another barrier set) for no reason but submission size.
   *
   * BLESSED_FLUSH_AT_PASS_END=1: D3D11ImmediateContext::ConsiderFlush
   * defers a weak hint the flush tracker approved while render targets
   * are bound, and the upstream ConsiderFlush at the next render-target
   * change (NotifyRenderPassBoundary) takes it: the tracker still holds
   * it as its missed type and its heuristic only gets more willing as
   * chunks pile up. Never deferred: explicit flushes, synchronisation
   * flushes (a waiting Map, a query read-back, fence Signal, Present),
   * strong hints (read-back latency; they follow copies, which end a
   * dxvk pass anyway), the cost cap, and any flush once the pending
   * command list reaches MaxChunks() chunks
   * (BLESSED_FLUSH_AT_PASS_END_MAX_CHUNKS, default 32).
   *
   * Counters are touched only by the immediate context's owner thread
   * (the game thread, or the threaded front end's worker, which takes
   * the context over through its drain handshake).
   */
  class BlessedFlushPassEnd {
  public:

    /// Whether the policy is on. Read once, cached by the context.
    static bool Enabled();

    /// Pending chunk count at which a deferral is refused.
    static uint32_t MaxChunks();

    /// A render-target change (D3D11 level pass boundary).
    static void OnBoundary();

    /// A flush executed while targets were bound, i.e. a pass split.
    static void OnSplit();

    /// A weak hint deferred; \p GpuIdle when nothing was in flight.
    static void OnDeferred(bool GpuIdle);

    /// A deferral refused by the chunk or cost cap.
    static void OnCapped();

    /// Frame end: logs per-frame averages roughly every 5 seconds.
    static void OnFrame();

  };

}
