// blessed: auto-instancing stage 0 -- a live census of draws that differ only in their constant buffers
#pragma once

#include <cstdint>

#include "d3d11_context_state.h"

#include "../util/util_likely.h"

namespace dxvk {

  // blessed: backing flag for BlessedAutoInstance::IsCensusEnabled(), split
  // out so the hot-path gate is a trivial inline read -- same pattern as
  // blessed_cascades_detail::g_enabled. Set once at static-init time (dll
  // load) from BLESSED_AUTOINSTANCE_CENSUS; never changes after.
  namespace blessed_autoinstance_detail {
    extern const bool g_census;
  }

  /**
   * \brief Draw parameters the census keys on
   *
   * Non-indexed draws leave \c indexed false and put their vertex count
   * and first vertex in \c count and \c first.
   */
  struct BlessedAutoInstanceDraw {
    bool      indexed       = false;
    uint32_t  count         = 0u;
    uint32_t  first         = 0u;
    int32_t   baseVertex    = 0;
    uint32_t  instanceCount = 1u;
    uint32_t  startInstance = 0u;
  };

  /**
   * \brief Auto-instancing census (BLESSED_AUTOINSTANCE_CENSUS=1)
   *
   * Counts, live and per frame, the draws an auto-instancing pass could
   * merge: draws whose whole bound state and geometry are equal and whose
   * only difference is the contents of their constant buffers. Design
   * and the one-frame numbers from the whiterun dump:
   * docs/research/auto-instancing.md (main repo).
   *
   * Two merge models are counted:
   * - consecutive: a draw joins the run of the draw right before it
   *   (nothing but cbuffer maps in between). Order is kept, so this is
   *   always legal.
   * - bucketed: inside one segment (same render targets, no clear,
   *   copy, dispatch, resolve or query in between) every candidate draw
   *   with an equal key could join one instanced draw. That reorders
   *   draws, which is legal only when every draw in the segment is
   *   depth-only and commutative (no rtv, no uav, no stream-out, stencil
   *   off, depth write on, LESS or LESS_EQUAL). Legal segments are
   *   "sortable"; the rest are counted apart as what reordering would
   *   find if it were legal.
   *
   * For both, which cbuffer slots actually change contents between the
   * merged draws (vs b0-b13, ps b0-b13) is counted, with the per-instance
   * byte cost that follows.
   *
   * Writes <BLESSED_PROBE_DIR>/autoinstance.jsonl, one line per 120
   * presents, per-frame averages. Immediate context only; every entry
   * point runs on the thread that replays d3d11 calls (the app thread,
   * or the threaded front end, which is drained at Present), under the
   * device lock the caller already holds. No locking of its own.
   *
   * Cheap when disabled: every hook is one inline read of a const bool.
   * When enabled it hashes the bound state and copies the bound
   * cbuffers on every draw: a probe, not for timing runs.
   */
  class BlessedAutoInstance {
  public:

    static bool IsCensusEnabled() {
      return blessed_autoinstance_detail::g_census;
    }

    /// Every draw that reaches dxvk on the immediate context, after the
    /// blessed skip checks.
    static void OnDraw(const D3D11ContextState& state, const BlessedAutoInstanceDraw& draw) {
      if (unlikely(IsCensusEnabled()))
        OnDrawSlow(state, draw);
    }

    /// Clears, copies, resolves, dispatches, query begin/end: ends the
    /// current segment and any consecutive run.
    static void OnBarrier() {
      if (unlikely(IsCensusEnabled()))
        OnBarrierSlow();
    }

    /// Frame boundary, from D3D11SwapChain::Present.
    static void OnPresent();

  private:

    static void OnDrawSlow(const D3D11ContextState& state, const BlessedAutoInstanceDraw& draw);
    static void OnBarrierSlow();

  };

}
