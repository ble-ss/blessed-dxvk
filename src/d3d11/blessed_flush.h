// blessed: submission policy for the immediate context, fewer mid-pass submissions (BLESSED_FLUSH)
#pragma once

#include <cstdint>

#include "../util/util_flush.h"

namespace dxvk {

  namespace blessed_flush_detail {
    // 0: upstream, 1: "pass", 2: "strong". Set once at dll load.
    extern const uint32_t g_policy;
  }

  /**
   * \brief Immediate context flush policy
   *
   * Upstream considers a flush (a weak hint) every time a cs chunk fills
   * up, which can land in the middle of a render pass: the pass is ended,
   * its attachments stored, and reloaded in the next command list.
   *
   * BLESSED_FLUSH=pass: a full cs chunk no longer hints. Weak hints stay
   * at render-target changes and ExecuteCommandList, so submissions line
   * up with pass boundaries. BLESSED_FLUSH=strong: weak hints are ignored
   * everywhere (the flush tracker's cap, as on tiling gpus). Both keep the
   * cost cap (GpuCostEstimate::MaxCostPerSubmission) and every explicit,
   * synchronization and strong-hint flush.
   */
  class BlessedFlush {
  public:

    /// Hint for a cs chunk that filled up. None still flushes on the cost cap.
    static GpuFlushType ChunkHint() {
      return blessed_flush_detail::g_policy
        ? GpuFlushType::None
        : GpuFlushType::ImplicitWeakHint;
    }

    /// Policy name for the log, null when upstream.
    static const char* Name() {
      static const char* names[] = { nullptr, "pass", "strong" };
      return names[blessed_flush_detail::g_policy];
    }

    /// Caps the flush tracker at strong hints.
    static bool IgnoreWeakHints() {
      return blessed_flush_detail::g_policy == 2u;
    }

  };

}
