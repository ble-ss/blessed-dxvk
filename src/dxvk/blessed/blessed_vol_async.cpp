// blessed: vanilla-vol-async -- DxvkContext side of BLESSED_VOL_ASYNC: =1 a same-family wrapper over blessedAsyncBegin/End, =2 the chain on the compute family with ownership transfers
#include <array>
#include <vector>

#include "blessed_async.h"

#include "../dxvk_cmdlist.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../dxvk_image.h"

#include "../../util/log/log.h"
#include "../../util/util_env.h"
#include "../../util/util_string.h"

namespace dxvk {

  bool DxvkContext::blessedVolAsyncBegin() {
    if (unlikely(m_blessedVolAsyncActive))
      return true; // already open (should not happen; the app-thread side never re-opens)

    if (!blessedAsyncAvailable())
      return false;

    // blessed: vol-async-3 -- BLESSED_VOL_ASYNC_DRYRUN=1: the window, the
    // verify mode and every hook run, but nothing leaves the graphics
    // queue; a verify mismatch here is the verify mode's own, not the
    // async path's (read once, cached)
    static const bool s_dryRun = env::getEnvVar("BLESSED_VOL_ASYNC_DRYRUN") == "1";
    if (unlikely(s_dryRun))
      return false;

    // blessed: vanilla's own textures (the froxel volumes, the sun cascade
    // atlas the generate dispatch reads) are never made concurrent across
    // queue families -- see src/d3d11/blessed_vol_async.h. A different-
    // family async queue would need real queue-ownership transfer barriers
    // this does not perform, so this stays off there; only a second
    // graphics-family queue (BLESSED_ASYNC_QUEUE=graphics, which
    // BlessedAsync::WantsGraphicsFamily() already forces whenever
    // BLESSED_VOL_ASYNC=1) is safe. This is the runtime double-check for
    // that invariant.
    if (m_device->blessedAsyncNeedsConcurrent()) {
      static bool s_warned = false;
      if (!s_warned) {
        s_warned = true;
        Logger::warn("blessed: vol-async: the async queue is a different family; "
          "BLESSED_VOL_ASYNC needs BLESSED_ASYNC_QUEUE=graphics (vanilla's own "
          "textures are not tagged concurrent there), staying off");
      }
      return false;
    }

    // blessed: readiness fix (round two audit finding 5) -- the cb vram
    // mirror only records its copies here, in flushCommandList, once per
    // whole command list; splitting recording for the async window does
    // not itself flush them. Without this, every rename since the last
    // real flush (the whole producer chunk's worth) stays pending until
    // the frame's very last chunk, submitted on the transfer queue well
    // after the kick already went out (DxvkCommandList::submit submits
    // kicks strictly before the chunk that would carry a flush this late)
    // -- the async dispatches could then read a stale mirror. Flushing
    // here, before splitCommands() below, attaches those copies to the
    // still-current producer chunk instead: its own per-chunk transfer-
    // to-graphics wait (blessedFlushCbMirror's barrier pair, applied by
    // DxvkCommandList::submit before that chunk's own graphics submit)
    // then guarantees the copy is done before the kick, which waits on
    // this same chunk's graphics semaphore. Renames that happen *during*
    // the window itself (the generate/chain dispatches' own constants)
    // are handled the other way, in blessedRenameBuffers: kept off the
    // mirror entirely while m_blessedVolAsyncActive is set (see
    // blessed_cb_ring.cpp), since by definition there is no "before" left
    // to flush them into. A no-op (empty-vector check) most frames.
    //
    // blessed: vol-async-3 -- and count how often each half is live, so a
    // log line says whether a run exercised the fix at all
    m_blessedVolStats.windows += 1u;
    m_blessedVolStats.pending += m_blessedCbMirrorPending.empty() ? 0u : 1u;

    uint32_t n = m_blessedVolStats.windows;
    if (n == 30u || n == 300u || !(n % 3000u)) {
      Logger::info(str::format("blessed: vol-async: ", n, " windows, ",
        m_blessedVolStats.pending, " opened with cb mirror copies pending (flushed before the kick), ",
        m_blessedVolStats.hostRenames, " cb renames inside a window kept on the host block, ",
        m_blessedVolStats.chainKicks, " chains on the compute family, ",
        m_blessedVolStats.cutByFlush, " windows ended early by a flush"));
    }

    blessedFlushCbMirror();

    // blessed: vol-async-3 -- =2 left the froxel volumes owned by the
    // compute family at the end of the last window; if that window's wait
    // never came (no wait draw that frame), wait and take them back now,
    // before this window's generate (graphics family) writes them
    if (unlikely(m_blessedVolNeedsAcquire))
      blessedAsyncSync();

    // blessed: vol-async-3 -- clears deferred before the window must run
    // on graphics, here: a dispatch in the window would otherwise flush
    // them (endRenderPass -> prepareShaderReadableImages) into the kick,
    // and a compute-family kick cannot record a clear at all
    this->endCurrentPass(false);

    blessedAsyncBegin();
    m_blessedVolAsyncActive = true;
    return true;
  }


  void DxvkContext::blessedVolAsyncEnd() {
    if (!m_blessedVolAsyncActive)
      return;

    // blessed: vol-async-3 -- the chain ran on the compute family: its own
    // pending barriers go into its own kick, then the release to graphics
    if (m_blessedVolOnCompute) {
      flushBarriers();
      blessedVolOwnership(true, false);
      m_blessedVolOnCompute = false;
      m_blessedVolNeedsAcquire = true;
    }

    blessedAsyncEnd();
    m_blessedVolAsyncActive = false;
  }


  bool DxvkContext::blessedVolAsyncSwitch(
    const Rc<DxvkImage>&            vol0,
    const Rc<DxvkImage>&            vol1) {
    // blessed: vol-async-3 -- BLESSED_VOL_ASYNC=2: after the generate
    // dispatch (graphics family kick), the chain goes to a kick of its own
    // on the compute-only family. Only the two froxel volumes cross: the
    // chain reads and writes nothing else but buffers (cbuffers, the
    // descriptor heaps), which are all created concurrent across every
    // family the device has (DxvkDevice::getSharingMode), so they need no
    // transfer. Ownership: release here (graphics -> compute, in the
    // generate kick), acquire at the top of the chain kick, release at its
    // end (blessedVolAsyncEnd), acquire on graphics after the wait
    // (blessedAsyncSync). The generate kick signals the async timeline,
    // the chain kick waits for it (DxvkCommandList::blessedSubmitKick).
    if (!m_blessedVolAsyncActive || m_blessedVolOnCompute
     || !m_device->blessedHasVolComputeQueue() || vol0 == nullptr || vol1 == nullptr)
      return false;

    m_blessedVolImages[0] = vol0;
    m_blessedVolImages[1] = vol1;

    // everything the generate kick still owes (its post-dispatch barriers)
    // goes into it, then the release
    flushBarriers();
    blessedVolOwnership(true, true);

    m_cmd->blessedAsyncEnd();
    m_cmd->blessedAsyncBegin(true);

    blessedVolOwnership(false, true);
    m_blessedVolOnCompute = true;
    m_blessedVolStats.chainKicks += 1u;
    return true;
  }


  void DxvkContext::blessedVolOwnership(bool release, bool toCompute) {
    // one ownership transfer half on both froxel volumes, whole image, in
    // the current exec buffer (the kick being recorded, or graphics)
    uint32_t gfx = m_device->queues().graphics.queueFamily;
    uint32_t cmp = m_device->queues().blessedVolCompute.queueFamily;

    constexpr VkPipelineStageFlags2 CsCopy = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                           | VK_PIPELINE_STAGE_2_COPY_BIT;
    constexpr VkAccessFlags2 AnyAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                       | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                       | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                                       | VK_ACCESS_2_TRANSFER_READ_BIT
                                       | VK_ACCESS_2_TRANSFER_WRITE_BIT;

    std::array<VkImageMemoryBarrier2, 2u> barriers = { };

    for (uint32_t i = 0u; i < 2u; i++) {
      auto& b = barriers[i];
      b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;

      if (release) {
        b.srcStageMask  = CsCopy;
        b.srcAccessMask = AnyAccess;
      } else {
        // graphics takes them back for whatever reads them next (pass
        // 138's fragment shader, the lens flare): all commands
        b.dstStageMask  = toCompute ? CsCopy : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.dstAccessMask = toCompute ? AnyAccess
          : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
      }

      b.srcQueueFamilyIndex = toCompute ? gfx : cmp;
      b.dstQueueFamilyIndex = toCompute ? cmp : gfx;
      b.oldLayout = m_blessedVolImages[i]->info().layout;
      b.newLayout = m_blessedVolImages[i]->info().layout;
      b.image = m_blessedVolImages[i]->handle();
      b.subresourceRange = { m_blessedVolImages[i]->formatInfo()->aspectMask,
        0u, VK_REMAINING_MIP_LEVELS, 0u, VK_REMAINING_ARRAY_LAYERS };

      m_cmd->track(m_blessedVolImages[i], DxvkAccess::Write);
    }

    VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = uint32_t(barriers.size());
    dep.pImageMemoryBarriers = barriers.data();

    // the graphics acquire goes first in its submission (InitBarriers runs
    // after the semaphore wait, before the init and exec buffers): an out-
    // of-order transfer placed in that chunk's init buffer may touch the
    // volumes too, and must not run before they are owned again
    bool graphicsAcquire = !release && !toCompute;

    m_cmd->cmdPipelineBarrier(graphicsAcquire
      ? DxvkCmdBuffer::InitBarriers
      : DxvkCmdBuffer::ExecBuffer, &dep);
  }


  void DxvkContext::blessedVolAcquireIfPending() {
    // blessed: vol-async-3 -- blessedAsyncSync's tail, right after the wait:
    // the chunk that waited for the kicks takes the volumes back first
    if (likely(!m_blessedVolNeedsAcquire))
      return;

    blessedVolOwnership(false, false);
    m_blessedVolNeedsAcquire = false;
  }


  void DxvkCommandList::blessedComputeBarrier(const VkDependencyInfo* dependencyInfo) {
    // blessed: vol-async-3 -- dxvk's barriers name every stage an image or
    // buffer may ever be used in (DxvkImageCreateInfo::stages), graphics
    // stages included; a compute-only queue rejects those. The graphics
    // work they refer to never runs on this queue (the semaphores order it
    // against this kick), so: graphics shader stages become the compute
    // shader stage, fixed-function graphics stages and the accesses only
    // they perform are dropped. Everything else passes through unchanged.
    constexpr VkPipelineStageFlags2 GfxShader =
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
      | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT
      | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT
      | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT
      | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
      | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    constexpr VkPipelineStageFlags2 GfxOnly = GfxShader
      | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT
      | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT
      | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT
      | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
      | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
      | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
      | VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT
      | VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
    constexpr VkAccessFlags2 GfxAccess =
        VK_ACCESS_2_INDEX_READ_BIT
      | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT
      | VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT
      | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT
      | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
      | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT
      | VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT
      | VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT
      | VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;

    auto stages = [] (VkPipelineStageFlags2 s) {
      VkPipelineStageFlags2 r = s & ~GfxOnly;
      if (s & GfxShader)
        r |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      return r;
    };

    auto fix = [&] (auto& b) {
      b.srcStageMask  = stages(b.srcStageMask);
      b.dstStageMask  = stages(b.dstStageMask);
      b.srcAccessMask &= ~GfxAccess;
      b.dstAccessMask &= ~GfxAccess;
      if (!b.srcStageMask) b.srcAccessMask = 0u;
      if (!b.dstStageMask) b.dstAccessMask = 0u;
    };

    std::vector<VkMemoryBarrier2> mem(dependencyInfo->pMemoryBarriers,
      dependencyInfo->pMemoryBarriers + dependencyInfo->memoryBarrierCount);
    std::vector<VkBufferMemoryBarrier2> buf(dependencyInfo->pBufferMemoryBarriers,
      dependencyInfo->pBufferMemoryBarriers + dependencyInfo->bufferMemoryBarrierCount);
    std::vector<VkImageMemoryBarrier2> img(dependencyInfo->pImageMemoryBarriers,
      dependencyInfo->pImageMemoryBarriers + dependencyInfo->imageMemoryBarrierCount);

    for (auto& b : mem) fix(b);
    for (auto& b : buf) fix(b);
    for (auto& b : img) fix(b);

    VkDependencyInfo dep = *dependencyInfo;
    dep.pMemoryBarriers = mem.data();
    dep.pBufferMemoryBarriers = buf.data();
    dep.pImageMemoryBarriers = img.data();

    m_vkd->vkCmdPipelineBarrier2(getCmdBuffer(DxvkCmdBuffer::ExecBuffer), &dep);
  }

}
