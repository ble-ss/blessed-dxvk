// blessed: cb-ring -- the constant-buffer ring: game-side chunk allocation, the replay-side batched rename, the frame fence
#include "blessed_cb_ring.h"

#include "d3d11_buffer.h"
#include "d3d11_context_imm.h"
#include "d3d11_device.h"

#include "../dxvk/blessed/blessed_cb_ring.h"
#include "../dxvk/blessed/blessed_cb_mirror.h"

#include "../util/util_blessed_probe.h"

namespace dxvk {

  BlessedCbRing::BlessedCbRing()
  : m_frameFence(new sync::Fence(0u)) {

  }


  BlessedCbRing::~BlessedCbRing() {
    // Blocks are released here; any buffer still pointing into one, or
    // a command list that tracked one, keeps its own reference.
  }


  bool BlessedCbRing::allocSlow(
          Pool&                 pool,
          DxvkBuffer*           buffer,
          VkDeviceSize          size,
          BlessedCbRingChunk*   chunk) {
    if (unlikely(pool.disabled))
      return false;

    if (pool.stall) {
      pool.stall -= 1u;
      return false;
    }

    blessed::CbRingScope blessedProbe_Advance(blessed::CbRingEvent::Advance);

    // Retire the current block, if any, with the frame it filled up in
    if (pool.block) {
      pool.blocks[pool.current].retireFrame = m_frame;
      pool.retired.push_back(pool.current);

      pool.block       = nullptr;
      pool.mirrorBlock = nullptr; // blessed: cb-mirror
      pool.base        = nullptr;
      pool.offset      = 0u;
      pool.end         = 0u;

      // blessed: cb-mirror -- its dirty range is final as of right now
      chunk->blockRetired = true;
    }

    uint32_t index = 0u;

    if (!findFreeBlock(pool, &index)) {
      if (pool.blocks.size() >= MaxBlocks) {
        pool.stall = StallMaps;
        return false;
      }

      blessed::CbRingScope blessedProbe_NewBlock(blessed::CbRingEvent::NewBlock);

      Block block;
      block.alloc = allocateBlock(pool, buffer);

      if (!block.alloc || !block.alloc->mapPtr()) {
        if (&pool == &m_pools[1]) {
          // blessed: perf-halfrate -- the cached pool takes over
          Logger::warn("d3d11.blessedCbRingDeviceLocal: failed to allocate a mapped device-local ring block, using cached blocks");
          pool.disabled = true;
          m_deviceLocal = false;
          return alloc(buffer, size, chunk);
        }

        Logger::warn("d3d11.blessedCbRing: failed to allocate a mapped ring block, ring disabled");
        pool.disabled = true;
        return false;
      }

      block.mapPtr = reinterpret_cast<char*>(block.alloc->mapPtr());

      // blessed: cb-mirror -- a device-local mirror of this block, filled
      // by DxvkContext::blessedFlushCbMirror at every submission. Pool 0
      // only: pool 1's blocks are already gpu-local (blessedCbRingDeviceLocal).
      if (&pool == &m_pools[0] && m_mirror) {
        block.mirror = buffer->blessedAllocateStorage(BlockSize,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (!block.mirror) {
          Logger::warn("d3d11.blessedCbMirror: failed to allocate a device-local mirror block, mirroring disabled");
          m_mirror = false;
        } else {
          // blessed: cb-mirror -- lets the threaded front end's compact
          // replay resolve this block's mirror (findMirror), off-thread
          publishMirror(block.alloc.ptr(), block.mirror.ptr());
        }
      }

      if (pool.blocks.empty()) {
        if (&pool == &m_pools[1]) {
          VkMemoryPropertyFlags props = block.alloc->getMemoryProperties();
          Logger::info(str::format("d3d11.blessedCbRingDeviceLocal: on, first block memory",
            (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " device-local" : " NOT device-local (no resizable bar type left?)",
            (props & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " host-cached" : "",
            " (VkMemoryPropertyFlags ", uint32_t(props), ")"));
        } else {
          Logger::info(str::format("d3d11.blessedCbRing: on, ", BlockSize >> 10u, " KiB blocks, ",
            MaxBlocks, " max, chunks aligned to ", m_alignment, " bytes"));

          if (m_mirror) {
            Logger::info(str::format("d3d11.blessedCbMirror: on, first mirror memory",
              (block.mirror->getMemoryProperties() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " device-local" : " NOT device-local (out of vram budget?)",
              " (VkMemoryPropertyFlags ", uint32_t(block.mirror->getMemoryProperties()), ")"));
          }
        }
      }

      index = uint32_t(pool.blocks.size());
      pool.blocks.push_back(std::move(block));
    }

    pool.current     = index;
    pool.block       = pool.blocks[index].alloc.ptr();
    pool.base        = pool.blocks[index].mapPtr;
    pool.mirrorBlock = pool.blocks[index].mirror.ptr(); // blessed: cb-mirror
    pool.offset      = 0u;
    pool.end         = BlockSize;

    // size <= MaxChunkSize < BlockSize, so this cannot fail again
    return alloc(buffer, size, chunk);
  }


  Rc<DxvkResourceAllocation> BlessedCbRing::allocateBlock(Pool& pool, DxvkBuffer* buffer) {
    if (&pool != &m_pools[1])
      return buffer->blessedAllocateStorage(BlockSize);

    // blessed: perf-halfrate -- the resizable-bar memory type; dxvk's
    // allocator drops DEVICE_LOCAL on its own if none is left
    return buffer->blessedAllocateStorage(BlockSize,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  }


  bool BlessedCbRing::findFreeBlock(Pool& pool, uint32_t* index) {
    // Oldest retired block first. A block retired in frame r took its
    // last rename before Present r + 1, which signals r + 1.
    uint64_t finished = m_frameFence->value();

    for (size_t i = 0; i < pool.retired.size(); i++) {
      const Block& block = pool.blocks[pool.retired[i]];

      if (block.retireFrame >= finished)
        continue;

      if (block.alloc->blessedUseCount() != 1u)
        continue;

      // blessed: cb-mirror -- a mirrored chunk's storage is the mirror, not
      // the host block, so reuse must wait on the mirror's refs too, or a
      // buffer still pointing at it would see its bytes overwritten by the
      // next round of chunks into the reused block
      if (block.mirror && block.mirror->blessedUseCount() != 1u)
        continue;

      *index = pool.retired[i];
      pool.retired.erase(pool.retired.begin() + i);
      return true;
    }

    return false;
  }


  bool D3D11ImmediateContext::BlessedMapCbRing(
          D3D11Buffer*                pResource,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    // Game half: a chunk, and the app-visible map pointer. Touches only
    // the ring and the D3D11Buffer, never context state.
    UINT bufferSize = pResource->Desc()->ByteWidth;
    BlessedCbRingChunk chunk;

    if (unlikely(!m_blessedCbRing.alloc(pResource->BlessedBufferPtr(), bufferSize, &chunk))) {
      blessed::CbRingScope blessedProbe_Fallback(blessed::CbRingEvent::Fallback);
      return false;
    }

    pResource->BlessedSetMapPtr(chunk.mapPtr);

    pMappedResource->pData      = chunk.mapPtr;
    pMappedResource->RowPitch   = bufferSize;
    pMappedResource->DepthPitch = bufferSize;

    // Replay half: the rename, as data
    BlessedEmitCbRename(pResource, chunk);

    // blessed: cb-mirror -- rare (about as often as a block fills, ~1 MiB
    // of cbuffer writes): flush pending mirror copies now and split, so
    // this copy's transfer-queue submission gets a head start instead of
    // landing in the same submission as the draws that wait on it.
    if (unlikely(chunk.blockRetired && m_blessedCbRing.mirrorEnabled()))
      BlessedFlushCbMirrorEarly();

    return true;
  }


  void D3D11ImmediateContext::BlessedEmitCbRename(
          D3D11Buffer*                pResource,
    const BlessedCbRingChunk&         chunk) {
    // Ordering: the rename sits in the cs stream exactly where the map
    // happened. It is appended to the previous rename command only if
    // that is still the last command in the chunk (any other command
    // resets m_csDataType), so a draw or copy recorded after this map
    // always runs after it, and one recorded before it runs before.
    if (m_csDataType == D3D11CmdType::BlessedCbRename && m_blessedCbRenameBlock == chunk.block) {
      void* entry = m_csChunk->pushData(m_csData, 1u);

      if (likely(entry)) {
        new (entry) DxvkBlessedCbRename { pResource->GetBuffer(), chunk.offset };
        return;
      }
    }

    blessed::CbRingScope blessedProbe_RenameCmd(blessed::CbRingEvent::RenameCmd);

    EmitCsCmd<DxvkBlessedCbRename>(D3D11CmdType::BlessedCbRename, 1u, [
      cBlock  = Rc<DxvkResourceAllocation>(chunk.block),
      cMirror = Rc<DxvkResourceAllocation>(chunk.mirror) // blessed: cb-mirror
    ] (DxvkContext* ctx, const DxvkBlessedCbRename* entries, size_t count) {
      ctx->blessedRenameBuffers(cBlock, cMirror, entries, count);
    });

    m_blessedCbRenameBlock = chunk.block;
    new (m_csData->first()) DxvkBlessedCbRename { pResource->GetBuffer(), chunk.offset };
  }


  void D3D11ImmediateContext::BlessedEndFrameCbRing() {
    // Game half ends the frame; replay half queues the gpu signal after
    // everything recorded so far, this frame's renames included.
    // blessed: threaded-fe-2 -- with Present recorded, the game side
    // counted this frame when it recorded the Present: a block it retires
    // meanwhile must not see a frame number behind its own renames
    uint64_t frame = m_blessedCbRingPresetFrame
      ? std::exchange(m_blessedCbRingPresetFrame, 0u)
      : m_blessedCbRing.endFrame();

    // blessed: cb-mirror -- a periodic engagement line in the dxvk log;
    // no-op on all but one frame in the window, and silent if the window
    // never mirrored anything (mirroring off, or nothing to mirror yet)
    blessed::BlessedCbMirrorMaybeLog(frame);

    EmitCs<false>([
      cFence = m_blessedCbRing.frameFence(),
      cFrame = frame
    ] (DxvkContext* ctx) {
      ctx->signal(cFence, cFrame);
    });
  }


  void D3D11ImmediateContext::BlessedFlushCbMirrorEarly() {
    // blessed: cb-mirror -- see DxvkContext::blessedFlushCbMirrorEarly.
    // Reached from the direct path (BlessedMapCbRing, above) and from the
    // threaded front end's replay (D3D11ThreadedContext::
    // ReplayCbMirrorEarlyFlush), both game-thread-adjacent callers that
    // don't touch DxvkContext directly.
    EmitCs<false>([] (DxvkContext* ctx) {
      ctx->blessedFlushCbMirrorEarly();
    });
  }

}
