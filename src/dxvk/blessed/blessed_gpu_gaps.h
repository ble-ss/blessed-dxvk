// blessed: gpu timestamps at every command buffer's begin and end, so a frame splits into app passes, dxvk's own work and idle (BLESSED_GPU_PASSES)
#pragma once

#include <cstdint>
#include <vector>

#include "../../vulkan/vulkan_loader.h"

namespace dxvk {

  class DxvkDevice;

  namespace blessed_gpu_gaps_detail {
    // set once at dll load: on with BLESSED_GPU_PASSES=1 unless
    // BLESSED_GPU_GAPS=0; with BLESSED_GPU_PASSES=frame only if =1
    extern const bool g_enabled;
  }

  /**
   * \brief What a graphics submission waited on before its first buffer ran
   *
   * Only the first graphics command buffer of a chunk carries a wait. The
   * idle time right before it is filed under that wait.
   */
  enum class BlessedGapWait : uint8_t {
    None    = 0,  ///< no semaphore: the queue ran dry (late submit, per-submit overhead)
    Acquire = 1,  ///< swapchain acquire semaphore (the present blit's command list)
    Async   = 2,  ///< blessed async-compute kick
    Sdma    = 3,  ///< dxvk's dedicated transfer queue (uploads in this chunk)
    Fence   = 4,  ///< per command list waits (tracking fence for sdma reuse, app fences)
    Sparse  = 5,  ///< sparse binding
    Count
  };

  /**
   * \brief One recorded timestamp, owned by the command list until submit
   */
  struct BlessedGapMark {
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    uint32_t        query     = 0u;
    uint8_t         type      = 0u;   // DxvkCmdBuffer
    bool            end       = false;
  };

  /**
   * \brief One executed command buffer, as handed over at submit
   */
  struct BlessedGapSubmitEntry {
    uint32_t        beginQuery = 0u;
    uint32_t        endQuery   = 0u;
    uint8_t         type       = 0u;   // DxvkCmdBuffer
    bool            graphics   = true; // false: dedicated transfer queue
    BlessedGapWait  wait       = BlessedGapWait::None;
  };

  /**
   * \brief Per command list data handed over at submit
   */
  struct BlessedGapSubmitInfo {
    std::vector<BlessedGapSubmitEntry> entries;
    uint32_t  graphicsSubmits = 0u;  // vkQueueSubmit calls on the graphics queue
    uint32_t  sdmaChunks      = 0u;  // chunks with work on the dedicated transfer queue
    uint32_t  barriers        = 0u;
    uint32_t  renderPasses    = 0u;
    bool      present         = false;
  };

  /**
   * \brief One graphics command buffer on the gpu timeline, raw ticks
   */
  struct BlessedGapSegment {
    uint64_t        begin = 0u;
    uint64_t        end   = 0u;
    uint8_t         type  = 0u;   // DxvkCmdBuffer
    BlessedGapWait  wait  = BlessedGapWait::None;
  };

  /**
   * \brief One resolved frame: every command list from the one after the
   *   previous present's up to and including this present's
   */
  struct BlessedGapFrame {
    std::vector<BlessedGapSegment> gfx;       // in graphics queue order
    uint64_t  prevEnd         = 0u;  // previous frame's last graphics end, 0 if unknown
    uint64_t  sdmaTicks       = 0u;  // busy on the dedicated transfer queue
    uint32_t  cmdLists        = 0u;
    uint32_t  graphicsSubmits = 0u;
    uint32_t  sdmaChunks      = 0u;
    uint32_t  barriers        = 0u;
    uint32_t  renderPasses    = 0u;
    /// At the submit of this frame's first command list, had the previous
    /// frame's last graphics buffer already finished on the gpu?
    /// 0: yes (the queue ran dry, the submit came late), 1: no (queued in
    /// time), 2: unknown (no previous frame).
    uint32_t  startState      = 2u;
    /// CPU time the submission thread spent in the present call right
    /// before this frame (vkQueuePresentKHR), and from its return to this
    /// frame's first submit. 0 when unknown.
    uint64_t  presentCallNs     = 0u;
    uint64_t  presentToSubmitNs = 0u;
  };

  /**
   * \brief GPU gap timer, the dxvk half of BLESSED_GPU_PASSES
   *
   * Every graphics and transfer command buffer gets a timestamp
   * (ALL_COMMANDS) right after it begins and right before it ends, from a
   * query pool of our own with host resets. At submit, the command list
   * hands over which buffers actually ran and what each chunk waited on.
   * A present command list closes a frame. Frames resolve without a gpu
   * sync on the submission thread; d3d11's BlessedGpuPasses takes them,
   * lines them up with its own pass marks (same queue, same clock) and
   * writes the split into its jsonl.
   *
   * When off, every call site pays one cached-bool branch.
   */
  class BlessedGpuGaps {
  public:

    static bool IsEnabled() {
      return blessed_gpu_gaps_detail::g_enabled;
    }

    /// Any thread that records. Returns a reset query index, ~0u if none
    /// (pool failed, another device, or no timestamps on the transfer family).
    static uint32_t AllocQuery(DxvkDevice* device, bool transferQueue);

    /// Query pool for AllocQuery's indices.
    static VkQueryPool QueryPool();

    /// Submission thread, in submission order.
    static void OnSubmit(DxvkDevice* device, const BlessedGapSubmitInfo& info);

    /// Submission thread, around DxvkPresenter::presentImage.
    static void OnPresentCall(uint64_t beginNs, uint64_t endNs);

    /// Monotonic nanoseconds for OnPresentCall.
    static uint64_t NowNs();

    /// App thread: takes the oldest resolved frame. False when none.
    static bool TakeFrame(BlessedGapFrame& frame);

    /// BLESSED_NO_TRANSFER_QUEUE=1: device creation maps dxvk's transfer
    /// queue onto the graphics queue (sdma buffers run there, in order,
    /// without semaphores). Independent of the timer.
    static bool NoTransferQueue();

  };

}
