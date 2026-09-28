// blessed: cb-mirror -- lightweight d3d11.blessedCbMirror engagement counters,
// dumped to the dxvk log every so many frames so an in-game run can be
// checked without a dedicated probe capture. A handful of relaxed atomic
// adds, touched only on the rename/flush paths mirroring itself is already
// behind a cached bool for.
#pragma once

#include <atomic>
#include <cstdint>

namespace dxvk::blessed {

  struct CbMirrorStats {
    // DxvkContext::blessedRenameBuffers, per entry, gated behind that call's
    // own mirror != nullptr (so a call with mirroring off touches neither --
    // one branch, already needed for the rename itself, and no atomics).
    std::atomic<uint64_t> mirrored{ 0u }; // pointed at the mirror
    std::atomic<uint64_t> cpuRead { 0u }; // excluded: DxvkBuffer::blessedCpuRead

    // The threaded front end's compact path: every compact CbRename
    // replayed while mirroring is on, and how many found nothing in
    // BlessedCbRing::findMirror (that rename fell back to the host block).
    // compactMiss should read 0; a nonzero rate with mirroring on is a bug,
    // not a tuning result.
    std::atomic<uint64_t> compact     { 0u };
    std::atomic<uint64_t> compactMiss { 0u };

    // DxvkContext::blessedFlushCbMirror.
    std::atomic<uint64_t> bytesCopied { 0u }; // total bytes, every copy
    std::atomic<uint64_t> copies      { 0u }; // individual per-block copies (bytesCopied / copies = bytes/copy)
    std::atomic<uint64_t> submissions { 0u }; // flush calls that copied >= 1 range (a vkCmdCopyBuffer batch)
    std::atomic<uint64_t> earlySplits { 0u }; // of those, triggered by a block retiring mid-chunk, not the chunk's own end (blessedFlushCbMirrorEarly)
  };

  extern CbMirrorStats g_cbMirrorStats;

  /**
   * \brief Logs the last \p periodFrames frames' counters, then resets them
   *
   * Game thread only (called from D3D11ImmediateContext::BlessedEndFrameCbRing,
   * once per Present). A no-op on every frame but the last of the window,
   * and silent if the window saw no renames at all (the ring is on but
   * nothing used it, e.g. loading screens).
   */
  void BlessedCbMirrorMaybeLog(uint64_t frame, uint32_t periodFrames = 600u);

}
