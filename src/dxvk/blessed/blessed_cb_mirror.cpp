// blessed: cb-mirror -- copies each ring block's byte range written since
// the last submission to its device-local mirror, on the transfer queue,
// before this submission's graphics work can read it (d3d11.blessedCbMirror)
#include "blessed_cb_mirror.h"

#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../../util/util_env.h"

namespace dxvk {

  namespace blessed {

    CbMirrorStats g_cbMirrorStats;

    void BlessedCbMirrorMaybeLog(uint64_t frame, uint32_t periodFrames) {
      if (frame == 0u || (frame % periodFrames) != 0u)
        return;

      uint64_t mirrored    = g_cbMirrorStats.mirrored.exchange(0u, std::memory_order_relaxed);
      uint64_t cpuRead     = g_cbMirrorStats.cpuRead.exchange(0u, std::memory_order_relaxed);
      uint64_t compact     = g_cbMirrorStats.compact.exchange(0u, std::memory_order_relaxed);
      uint64_t compactMiss = g_cbMirrorStats.compactMiss.exchange(0u, std::memory_order_relaxed);
      uint64_t bytes       = g_cbMirrorStats.bytesCopied.exchange(0u, std::memory_order_relaxed);
      uint64_t copies      = g_cbMirrorStats.copies.exchange(0u, std::memory_order_relaxed);
      uint64_t submits     = g_cbMirrorStats.submissions.exchange(0u, std::memory_order_relaxed);
      uint64_t early       = g_cbMirrorStats.earlySplits.exchange(0u, std::memory_order_relaxed);

      // ring is on but this window never mirrored a cbuffer (loading
      // screen, menu, or the feature is off): say nothing rather than log
      // a line of zeroes every window
      if (!mirrored && !cpuRead && !compact)
        return;

      Logger::info(str::format("d3d11.blessedCbMirror: last ", periodFrames, " frames -- renames ",
        mirrored, " mirrored, ", cpuRead, " cpu-read",
        " (compact path ", compact, " replayed, ", compactMiss, " lookup misses); ",
        copies, " copies (", copies / periodFrames, "/frame, ",
        copies ? bytes / copies : 0u, " bytes/copy); ",
        submits, " transfer submissions (", double(submits) / double(periodFrames), "/frame, ",
        early, " early -- ", double(early) / double(periodFrames), "/frame)"));
    }

  }

  void DxvkContext::blessedTrackCbMirrorDirty(
    const Rc<DxvkResourceAllocation>& block,
    const Rc<DxvkResourceAllocation>& mirror,
          VkDeviceSize                lo,
          VkDeviceSize                hi) {
    for (auto& e : m_blessedCbMirrorPending) {
      if (e.block == block) {
        e.lo = std::min(e.lo, lo);
        e.hi = std::max(e.hi, hi);
        return;
      }
    }

    m_blessedCbMirrorPending.push_back({ block, mirror, lo, hi });
  }


  void DxvkContext::blessedFlushCbMirror() {
    if (m_blessedCbMirrorPending.empty())
      return;

    // Constant buffers can be read from any shader stage; one plain memory
    // barrier per range (not a per-buffer one) covers the whole batch.
    constexpr VkPipelineStageFlags2 CbReadStages =
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
      | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT
      | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT
      | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT
      | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    constexpr VkAccessFlags2 CbReadAccess = VK_ACCESS_2_UNIFORM_READ_BIT;

    auto& transferBatch = getBarrierBatch(DxvkCmdBuffer::SdmaBuffer);
    auto& initBatch     = getBarrierBatch(DxvkCmdBuffer::InitBarriers);

    blessed::g_cbMirrorStats.submissions.fetch_add(1u, std::memory_order_relaxed);

    for (const auto& e : m_blessedCbMirrorPending) {
      auto srcSlice = e.block->getBufferInfo();
      auto dstSlice = e.mirror->getBufferInfo();

      VkBufferCopy2 region = { VK_STRUCTURE_TYPE_BUFFER_COPY_2 };
      region.srcOffset = srcSlice.offset + e.lo;
      region.dstOffset = dstSlice.offset + e.lo;
      region.size      = e.hi - e.lo;

      VkCopyBufferInfo2 copyInfo = { VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2 };
      copyInfo.srcBuffer   = srcSlice.buffer;
      copyInfo.dstBuffer   = dstSlice.buffer;
      copyInfo.regionCount = 1;
      copyInfo.pRegions    = &region;

      m_cmd->cmdCopyBuffer(DxvkCmdBuffer::SdmaBuffer, &copyInfo);

      // Same split-barrier shape as DxvkContext::accessBufferTransfer: a
      // release on the transfer queue's batch, an acquire on InitBarriers,
      // which runs on the graphics queue right before ExecBuffer in this
      // same submission -- after the graphics queue has already waited on
      // the transfer queue's semaphore (DxvkCommandList::submit).
      VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };

      if (m_device->hasDedicatedTransferQueue()) {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_NONE;
        transferBatch.addMemoryBarrier(barrier);

        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask  = CbReadStages;
        barrier.dstAccessMask = CbReadAccess;
        initBatch.addMemoryBarrier(barrier);
      } else {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask  = CbReadStages;
        barrier.dstAccessMask = CbReadAccess;
        transferBatch.addMemoryBarrier(barrier);
      }

      // Keep both allocations alive until this command list retires. By
      // then the graphics queue has waited on the transfer semaphore this
      // copy signals, so the transfer-queue use is already done too.
      m_cmd->track(e.block);
      m_cmd->track(e.mirror);

      blessed::g_cbMirrorStats.bytesCopied.fetch_add(region.size, std::memory_order_relaxed);
      blessed::g_cbMirrorStats.copies.fetch_add(1u, std::memory_order_relaxed);
    }

    m_blessedCbMirrorPending.clear();
  }


  void DxvkContext::blessedFlushCbMirrorEarly() {
    // blessed: cb-mirror -- called when a ring block retires (its dirty
    // range is now final): flush its copy now instead of waiting for this
    // chunk's own end, and split the command list right after, so this
    // copy's transfer-queue submission lands strictly before whatever
    // graphics work the rest of the chunk goes on to record -- letting it
    // run on the transfer queue while that graphics work (which mostly
    // doesn't touch this range) keeps the graphics queue busy, instead of
    // both landing in the same submission with the draw waiting on the
    // copy it was submitted alongside (the audit's finding 5, idle_by.sdma).
    //
    // Correctness is unchanged either way: any draw needing this range
    // still waits on the same semaphore blessedFlushCbMirror always set
    // up; this only moves when the copy (and thus the semaphore signal)
    // is submitted, never what it waits on or what waits on it.
    // blessed: gated for an a/b (lead): BLESSED_CB_MIRROR_EARLY=1; unset
    // keeps the measured c52/c55 behaviour (copy at the chunk's end only)
    static const bool s_early = env::getEnvVar("BLESSED_CB_MIRROR_EARLY") == "1";
    if (!s_early || m_blessedCbMirrorPending.empty())
      return;

    blessedFlushCbMirror();
    blessed::g_cbMirrorStats.earlySplits.fetch_add(1u, std::memory_order_relaxed);
    splitCommands();
  }

}
