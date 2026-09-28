// blessed: BLESSED_VOL_ASYNC_VERIFY -- shadow-replays the async window onto scratch, bitwise-checks it against the real (possibly async) result
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;

  namespace blessed_vol_async_verify_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief BLESSED_VOL_ASYNC_VERIFY=1: an independent synchronous reference
   *        for BLESSED_VOL_ASYNC's window, checked every n-th frame
   *
   * BLESSED_VOL_COLLAPSE_VERIFY covers the chain's own correctness given
   * whatever generate already left in the two froxel volumes -- if a race
   * corrupts generate's *own* output (round two audit finding 5: a mirrored
   * constant the async dispatches read, copied too late), both the
   * collapsed and the vanilla side of that verify start from the same
   * already-wrong state and agree with each other. This is a second,
   * independent check: on a tracked frame, every dispatch the window
   * records (generate, then whatever chain shape is running -- collapsed
   * or raw, this file does not need to know which) is recorded as it runs
   * (shader, views, samplers, and the host bytes of each dynamic cbuffer,
   * read from the map pointer rather than the cb ring's vram mirror), then
   * replayed once the window closes, entirely through ordinary dispatches
   * on the graphics queue, onto a scratch pair: every srv or uav that
   * referenced one of the two froxel volumes is pointed at its scratch
   * twin, every constant comes from a default-usage copy of the recorded
   * bytes. Nothing of the replay touches the async queue, the mirror, or
   * the real volumes. One bitwise check, "vol-async-final", lands in
   * "<BLESSED_PROBE_DIR>/shader-verify.jsonl" (and the dxvk log) once the
   * frame's kick has actually been waited for (BlessedVolAsync's own
   * OnSyncPoint call, right after blessedAsyncSync()): both real froxel
   * volumes at the wait point against the replay's. Since the replay's
   * generate reads its own copy of every input, a wrong generate input on
   * the real side (a stale mirror) shows here too; there is no separate
   * generate check. (A snapshot copy taken inside the kick right after
   * generate was tried and dropped: in the synthetic it disagreed with
   * both the replay and a graphics-only run in 13-18 of 43 windows while
   * the real final volumes agreed in all 43 -- see NOTES-volasync.md.)
   *
   * BLESSED_VOL_ASYNC_VERIFY_PERIOD=<n> (default 60): check every n-th
   * window (counted by generate dispatches seen, so a half-rate/skipped
   * frame just delays the next check rather than miscounting).
   * BLESSED_VOL_ASYNC_VERIFY_MAX=<n> (default 200): a lifetime cap on
   * logged checks, so an always-on verify run doesn't grow the jsonl
   * without bound; 0 = unlimited.
   *
   * If a window's dispatches don't have the expected shape (an unrecognized
   * uav pair -- should not happen per the dump, "nothing but dispatches
   * happen in that stretch", but this is a debug tool and must not
   * silently compare mismatched work), that window's checks are skipped
   * with one warning, not counted as a check, and coverage stays honest.
   *
   * Requires BLESSED_VOL_ASYNC=1 or 2. Disabled: one cached-bool check per
   * dispatch/draw, no state.
   */
  class BlessedVolAsyncVerify {
  public:

    static bool IsEnabled() {
      return blessed_vol_async_verify_detail::g_enabled;
    }

    // Post-dispatch, immediate context only, every dispatch (mirrors
    // BlessedVolCollapse::OnDispatchDone's call site): starts tracking at
    // the generate dispatch of a checked window, shadow-replays every
    // later dispatch while BlessedVolAsync's window is open.
    static void OnDispatchDone(
            D3D11ImmediateContext*  ctx,
      const D3D11ContextState&      state,
            UINT                    groupsX,
            UINT                    groupsY,
            UINT                    groupsZ);

    // Right after blessedAsyncSync()'s wait is issued at the wait draw
    // (BlessedVolAsync::OnDrawPre): safe point to read back and compare.
    static void OnSyncPoint(D3D11ImmediateContext* ctx);

    // The window closed (BlessedVolAsync::OnDrawPre, at whatever draw
    // came first -- in game the depth prepass, long before the wait
    // draw): replays the recorded dispatches on graphics, onto scratch.
    static void OnWindowClose(D3D11ImmediateContext* ctx);

    // True while this file's own replay or copies are being recorded, so
    // BlessedVolAsync does not open a window for the replayed generate.
    static bool IsReplaying();

  };

}
