// blessed: cb-ring -- app-thread ring of persistently mapped blocks that Map(WRITE_DISCARD) on small dynamic cbuffers carves chunks from
#pragma once

#include <array>
#include <atomic>
#include <vector>

#include "../dxvk/dxvk_buffer.h"

#include "../util/sync/sync_signal.h"

namespace dxvk {

  /**
   * \brief One chunk handed out by \ref BlessedCbRing::alloc
   *
   * Plain data: everything the replay half needs to apply the rename,
   * so a later threaded front end can carry it in a record.
   */
  struct BlessedCbRingChunk {
    /// Ring block the chunk lives in. Kept alive by the ring.
    DxvkResourceAllocation* block = nullptr;
    /// Byte offset of the chunk inside the block
    VkDeviceSize            offset = 0u;
    /// Host pointer to the chunk
    void*                   mapPtr = nullptr;
    /// blessed: cb-mirror -- block's vram mirror, or null if mirroring is
    /// off or this block predates it. Kept alive by the ring.
    DxvkResourceAllocation* mirror = nullptr;
    /// blessed: cb-mirror -- true if getting this chunk retired the pool's
    /// previous block: the moment its dirty range becomes final. The
    /// caller uses this to flush that block's mirror copy early instead of
    /// waiting for the chunk's own end (BlessedFlushCbMirrorEarly).
    bool                     blockRetired = false;
  };

  /**
   * \brief Constant-buffer ring (d3d11.blessedCbRing)
   *
   * Game (app) thread only, and self-contained: nothing here reads or
   * writes d3d11 or dxvk context state, so a threaded front end can own
   * it on the game thread while a replay thread drives the context. The
   * replay half of a map is \c D3D11ImmediateContext::BlessedEmitCbRename.
   *
   * A pool of 1 MiB blocks in the memory the cached dynamic cbuffers use,
   * each one allocation, mapped for the ring's life. A map takes the next
   * aligned chunk of the current block with a bump pointer: no allocator
   * or allocation-cache call.
   *
   * A full block is retired with the game-side frame number. It is reused
   * only when both hold:
   * - the frame fence (signalled on the gpu after each frame's Present,
   *   see \ref endFrame) has passed the frame it was retired in. Every
   *   rename into the block was recorded before that Present, so all of
   *   them have been applied on the cs thread, and the gpu is done with
   *   every submission up to there.
   * - the ring holds the block's last reference. With every rename into
   *   it applied, the other possible holders are a buffer whose current
   *   storage is still a chunk of it (mapped once and not since), and a
   *   command list that tracked it when a buffer was renamed away from
   *   it, which lets go when the gpu finishes that submission.
   * Both reads are atomic loads, taken only when a block fills.
   *
   * Blocks never leave the pool while the ring lives. With every block
   * busy and the pool at its cap, \ref alloc fails for a while and the
   * caller takes the upstream DiscardSlice path. That path also renames
   * buffers away from the ring, so blocks come free again.
   */
  class BlessedCbRing {

  public:

    constexpr static VkDeviceSize BlockSize    = VkDeviceSize(1u) << 20;
    constexpr static VkDeviceSize MaxChunkSize = VkDeviceSize(16u) << 10;
    constexpr static uint32_t     MaxBlocks    = 96u;

    /// Maps that skip the ring after it found no free block at the cap
    constexpr static uint32_t     StallMaps    = 256u;

    BlessedCbRing();
    ~BlessedCbRing();

    BlessedCbRing             (const BlessedCbRing&) = delete;
    BlessedCbRing& operator = (const BlessedCbRing&) = delete;

    /**
     * \brief Sets the chunk alignment
     * \param [in] alignment Power of two, at least the device's
     *    minUniformBufferOffsetAlignment
     */
    void setAlignment(VkDeviceSize alignment) {
      m_alignment = alignment;
    }

    /**
     * \brief Hands out the next chunk
     *
     * \param [in] buffer The buffer being mapped. Only used to allocate
     *    a new block with its usage and memory properties.
     * \param [in] size Chunk size in bytes, at most \c MaxChunkSize
     * \param [out] chunk The chunk
     * \returns \c false if no block is free; use the upstream path
     */
    bool alloc(
            DxvkBuffer*           buffer,
            VkDeviceSize          size,
            BlessedCbRingChunk*   chunk) {
      // blessed: perf-halfrate -- device-local blocks unless a hook reads
      // this buffer on the cpu (see setDeviceLocal)
      Pool& pool = m_pools[(m_deviceLocal && !buffer->blessedCpuRead()) ? 1u : 0u];
      VkDeviceSize offset = pool.offset;

      if (likely(offset + size <= pool.end)) {
        pool.offset = (offset + size + m_alignment - 1u) & ~(m_alignment - 1u);

        chunk->block  = pool.block;
        chunk->offset = offset;
        chunk->mapPtr = pool.base + offset;
        chunk->mirror = pool.mirrorBlock; // blessed: cb-mirror
        return true;
      }

      return allocSlow(pool, buffer, size, chunk);
    }

    /**
     * rief blessed: perf-halfrate -- device-local ring blocks
     *
     * With d3d11.blessedCbRingDeviceLocal, a second pool of blocks lives
     * in DEVICE_LOCAL | HOST_VISIBLE memory (resizable bar), so the gpu
     * reads cbuffers from vram. Buffers a hook reads on the cpu
     * (DxvkBuffer::blessedCpuRead) stay in the cached pool: a cpu read
     * of write-combined vram is uncached and slow.
     */
    void setDeviceLocal(bool enable) {
      m_deviceLocal = enable;
    }

    /**
     * \brief blessed: cb-mirror -- a vram mirror of each cached block
     *
     * With d3d11.blessedCbMirror, every block in the cached pool gets a
     * same-size device-local mirror. Cbuffer bindings for a chunk point at
     * the mirror instead of the host block, unless a hook reads that
     * buffer on the cpu (DxvkBuffer::blessedCpuRead), which stays on the
     * host block, unmirrored. \c DxvkContext::blessedFlushCbMirror copies
     * each block's written range to its mirror on the transfer queue, once
     * per submission. Mutually pointless with setDeviceLocal (that pool's
     * blocks are already gpu-local); mirroring only ever applies to pool 0.
     */
    void setMirror(bool enable) {
      m_mirror = enable;
    }

    /**
     * \brief blessed: cb-mirror -- whether mirroring is on
     *
     * A plain bool, set once before the threaded front end's replay thread
     * exists and never written again: safe to read from any thread despite
     * not being atomic, same as \ref m_deviceLocal. Lets a caller skip
     * \ref findMirror entirely when mirroring is off.
     */
    bool mirrorEnabled() const {
      return m_mirror;
    }

    /**
     * \brief blessed: cb-mirror -- resolves a block's mirror
     *
     * The threaded front end's compact draw packet (FeOpCbRename) carries
     * only the host block, not its mirror -- adding it would cost 8 bytes
     * on every cb-ring map, mirrored or not. This is how its replay (see
     * blessed_threaded_record.cpp) resolves the mirror instead, off the
     * game thread that owns this ring.
     *
     * Safe from any thread: \ref m_mirrorTable is append-only, sized to
     * \ref MaxBlocks up front, and each slot is written in full before
     * \ref m_mirrorCount publishes it (release), so a concurrent reader
     * (acquire) never sees a half-written entry, and never touches
     * pool.blocks (which the game thread that owns this ring may be
     * resizing at the same time).
     *
     * \param [in] block A ring block, as carried by a chunk or a
     *    DxvkBlessedCbRename's block reference
     * \returns The mirror, or null if \p block predates mirroring, isn't
     *    from this ring's cached pool, or mirroring is off
     */
    DxvkResourceAllocation* findMirror(DxvkResourceAllocation* block) const {
      uint32_t count = m_mirrorCount.load(std::memory_order_acquire);

      for (uint32_t i = 0; i < count; i++) {
        if (m_mirrorTable[i].host == block)
          return m_mirrorTable[i].mirror;
      }

      return nullptr;
    }

    /**
     * \brief Ends the game-side frame
     *
     * Called once per Present, on the game thread. The caller must queue
     * a gpu signal of \ref frameFence to the returned value after all of
     * this frame's work (today: \c D3D11ImmediateContext::EndFrame).
     * \returns Number of the frame that just ended, from 1
     */
    uint64_t endFrame() {
      return ++m_frame;
    }

    /**
     * \brief Frame fence
     *
     * Reaches n once the gpu has finished frame n. Read on the game
     * thread, signalled from the cs thread's submissions.
     */
    const Rc<sync::Fence>& frameFence() const {
      return m_frameFence;
    }

  private:

    struct Block {
      Rc<DxvkResourceAllocation>  alloc;
      char*                       mapPtr      = nullptr;
      uint64_t                    retireFrame = 0u;
      // blessed: cb-mirror -- this block's vram mirror, or null
      Rc<DxvkResourceAllocation>  mirror;
    };

    // blessed: cb-mirror -- one findMirror table entry. Written once, by
    // publishMirror, before the game thread that owns this ring ever
    // shares the block with anyone else.
    struct MirrorEntry {
      DxvkResourceAllocation* host   = nullptr;
      DxvkResourceAllocation* mirror = nullptr;
    };

    std::array<MirrorEntry, MaxBlocks> m_mirrorTable;
    std::atomic<uint32_t>              m_mirrorCount { 0u };

    // blessed: cb-mirror -- appends a (host, mirror) pair to m_mirrorTable
    // for findMirror. Game thread only, called once per new mirrored block
    // (allocSlow), after both allocations are final.
    void publishMirror(DxvkResourceAllocation* host, DxvkResourceAllocation* mirror) {
      uint32_t slot = m_mirrorCount.load(std::memory_order_relaxed);
      m_mirrorTable[slot] = { host, mirror };
      m_mirrorCount.store(slot + 1u, std::memory_order_release);
    }

    // blessed: perf-halfrate -- one bump allocator and block pool per
    // memory kind: 0 the buffers' own (cached) memory, 1 device-local
    struct Pool {
      VkDeviceSize                offset    = 0u;
      VkDeviceSize                end       = 0u;
      char*                       base      = nullptr;
      DxvkResourceAllocation*     block     = nullptr;
      // blessed: cb-mirror -- current block's mirror (pool 0 only)
      DxvkResourceAllocation*     mirrorBlock = nullptr;
      uint32_t                    current   = 0u;

      uint32_t                    stall     = 0u;
      bool                        disabled  = false;

      std::vector<Block>          blocks;
      std::vector<uint32_t>       retired;
    };

    VkDeviceSize                m_alignment = 256u;
    bool                        m_deviceLocal = false;
    bool                        m_mirror      = false; // blessed: cb-mirror

    Pool                        m_pools[2];

    uint64_t                    m_frame     = 0u;
    Rc<sync::Fence>             m_frameFence;

    bool allocSlow(
            Pool&                 pool,
            DxvkBuffer*           buffer,
            VkDeviceSize          size,
            BlessedCbRingChunk*   chunk);

    bool findFreeBlock(Pool& pool, uint32_t* index);

    Rc<DxvkResourceAllocation> allocateBlock(Pool& pool, DxvkBuffer* buffer);

  };

}
