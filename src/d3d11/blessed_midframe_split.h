// blessed: forces one extra dxvk submission boundary at a fixed pass index each
// frame, off by default (BLESSED_MIDFRAME_SPLIT). evidence: the present-idle
// synthetic app shows nvidia's present-method bit 8 does not add gpu work
// evenly across the frame -- it makes one specific pass (not the first, not
// the last) run far slower while dxvk's own submit/barrier counts stay
// identical to the flag off. this switch tests whether inserting a submission
// boundary right before that pass gives the driver's post-flip work a gap to
// land in instead of stalling the pass itself. see docs/tuning-report.md,
// "nvidia's hidden present-method flag, bit by bit".
#pragma once

#include "blessed_gpu_passes.h"

namespace dxvk {

  class D3D11ImmediateContext;

  class BlessedMidframeSplit {

  public:

    /**
     * \brief Whether BLESSED_MIDFRAME_SPLIT names a pass index
     *
     * Cached on first call. Cost when unset: one cached bool read.
     */
    static bool IsEnabled();

    /**
     * \brief Called from BlessedGpuPassEvent for every pass boundary
     *
     * Counts pass-starting events on the immediate context and, once per
     * frame, forces a non-blocking flush right before the configured pass
     * index. Resets on the next OnPresent().
     *
     * \param [in] ctx Immediate context (never a deferred context)
     * \param [in] kind Kind of pass this call is about to start
     */
    static void OnPassEvent(D3D11ImmediateContext* ctx, BlessedGpuPassKind kind);

    /**
     * \brief Present boundary: resets the per-frame pass counter
     */
    static void OnPresent();

  };

}
