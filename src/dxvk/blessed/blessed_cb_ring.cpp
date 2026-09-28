// blessed: cb-ring -- cs-thread side of the constant-buffer ring: applies a batch of renames in map order
#include "blessed_cb_ring.h"
#include "blessed_cb_mirror.h"

#include "../dxvk_context.h"

namespace dxvk {

  void DxvkContext::blessedRenameBuffers(
    const Rc<DxvkResourceAllocation>& block,
    const Rc<DxvkResourceAllocation>& mirror,
    const DxvkBlessedCbRename*        entries,
          size_t                      count) {
    // Same steps as invalidateBuffer's uniform-only fast path, per entry:
    // swap the storage, keep the previous storage alive until the gpu is
    // done with this command list, reset access tracking, dirty the bound
    // uniform buffers. The previous storage is tracked once per distinct
    // allocation within the batch: a repeat is the ring block itself (or
    // another allocation this command list already holds), so a second
    // reference would change nothing about its lifetime.
    DxvkResourceAllocation* lastTracked = nullptr;
    VkShaderStageFlags stages = 0u;

    // blessed: cb-mirror -- the range of `block` this batch's mirrored
    // entries touch, handed to blessedTrackCbMirrorDirty below. A buffer a
    // cpu hook reads (blessedCpuRead) stays on the host block, unmirrored
    // and unchanged: mirror is null whenever d3d11.blessedCbMirror is off,
    // so this is one extra predictable branch per entry when disabled.
    bool         mirrorDirty = false;
    VkDeviceSize mirrorLo    = 0u;
    VkDeviceSize mirrorHi    = 0u;

    for (size_t i = 0; i < count; i++) {
      DxvkBuffer* buffer = entries[i].buffer.ptr();
      bool cpuRead = buffer->blessedCpuRead();
      // blessed: vanilla-vol-async readiness fix (round two audit finding
      // 5) -- a rename that happens *inside* the async window (the
      // generate dispatch's own constants, or a chain step's) has no
      // earlier point left to flush its mirror copy into before the kick
      // (blessedVolAsyncBegin's early flush only covers renames from
      // before the window opened). Keep these on the host block instead,
      // the same fallback blessedCpuRead already uses: the async
      // dispatch then reads what the cpu just wrote directly, with the
      // ordinary binding barrier, no cross-queue copy to race.
      bool useMirror = mirror != nullptr && !cpuRead && !m_blessedVolAsyncActive;

      if (unlikely(m_blessedVolAsyncActive) && mirror != nullptr) // blessed: vol-async-3, counted
        m_blessedVolStats.hostRenames += 1u;

      // blessed: cb-mirror -- gated on this call's own mirror != nullptr, a
      // value already loaded for useMirror above: a call with mirroring
      // off (mirror always null) touches neither counter, no atomics.
      if (mirror != nullptr) {
        if (cpuRead)
          blessed::g_cbMirrorStats.cpuRead.fetch_add(1u, std::memory_order_relaxed);
        else
          blessed::g_cbMirrorStats.mirrored.fetch_add(1u, std::memory_order_relaxed);
      }

      Rc<DxvkResourceAllocation> prev = buffer->blessedAssignStorageRange(
        useMirror ? mirror : block, entries[i].offset);

      if (prev.ptr() != lastTracked) {
        lastTracked = prev.ptr();
        m_cmd->track(std::move(prev));
      }

      buffer->resetTracking();
      stages |= buffer->getShaderStages();

      if (useMirror) {
        VkDeviceSize lo = entries[i].offset;
        VkDeviceSize hi = lo + buffer->info().size;

        mirrorLo = mirrorDirty ? std::min(mirrorLo, lo) : lo;
        mirrorHi = mirrorDirty ? std::max(mirrorHi, hi) : hi;
        mirrorDirty = true;
      }
    }

    if (mirrorDirty)
      blessedTrackCbMirrorDirty(block, mirror, mirrorLo, mirrorHi);

    m_descriptorState.dirtyBuffers(stages);
  }

}
