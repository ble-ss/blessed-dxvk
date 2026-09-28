#pragma once

#include "../util/util_time.h"

#include "../util/sync/sync_signal.h"

#include "d3d11_context.h"
#include "d3d11_state_object.h"
#include "d3d11_video.h"

#include "blessed_cb_ring.h" // blessed: cb-ring

namespace dxvk {
  
  class D3D11Buffer;
  class D3D11CommonTexture;
  class D3D11ThreadedContext; // blessed: threaded-fe

  class D3D11ImmediateContext : public D3D11CommonContext<D3D11ImmediateContext> {
    friend class D3D11CommonContext<D3D11ImmediateContext>;
    friend class D3D11SwapChain;
    friend class D3D11VideoContext;
    friend class D3D11DXGIKeyedMutex;
    friend class D3D11ThreadedContext; // blessed: threaded-fe, the facade that replays into this context
    friend class BlessedMidframeSplit; // blessed: between-passes, needs ExecuteFlush
  public:
    
    D3D11ImmediateContext(
            D3D11Device*    pParent,
      const Rc<DxvkDevice>& Device);
    ~D3D11ImmediateContext();
    
    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID  riid,
            void**  ppvObject);

    HRESULT STDMETHODCALLTYPE GetData(
            ID3D11Asynchronous*         pAsync,
            void*                       pData,
            UINT                        DataSize,
            UINT                        GetDataFlags);
    
    void STDMETHODCALLTYPE Begin(
            ID3D11Asynchronous*         pAsync);
    
    void STDMETHODCALLTYPE End(
            ID3D11Asynchronous*         pAsync);
    
    void STDMETHODCALLTYPE Flush();
    
    void STDMETHODCALLTYPE Flush1(
            D3D11_CONTEXT_TYPE          ContextType,
            HANDLE                      hEvent);

    HRESULT STDMETHODCALLTYPE Signal(
            ID3D11Fence*                pFence,
            UINT64                      Value);
    
    HRESULT STDMETHODCALLTYPE Wait(
            ID3D11Fence*                pFence,
            UINT64                      Value);

    void STDMETHODCALLTYPE ExecuteCommandList(
            ID3D11CommandList*  pCommandList,
            BOOL                RestoreContextState);
    
    HRESULT STDMETHODCALLTYPE FinishCommandList(
            BOOL                RestoreDeferredContextState,
            ID3D11CommandList   **ppCommandList);
    
    HRESULT STDMETHODCALLTYPE Map(
            ID3D11Resource*             pResource,
            UINT                        Subresource,
            D3D11_MAP                   MapType,
            UINT                        MapFlags,
            D3D11_MAPPED_SUBRESOURCE*   pMappedResource);
    
    void STDMETHODCALLTYPE Unmap(
            ID3D11Resource*             pResource,
            UINT                        Subresource);
            
    void STDMETHODCALLTYPE SwapDeviceContextState(
            ID3DDeviceContextState*           pState,
            ID3DDeviceContextState**          ppPreviousState);

    void Acquire11on12Resource(
            ID3D11Resource*             pResource,
            VkImageLayout               SrcLayout);

    void Release11on12Resource(
            ID3D11Resource*             pResource,
            VkImageLayout               DstLayout);

    void SynchronizeCsThread(
            uint64_t                          SequenceNumber);

    D3D10Multithread& GetMultithread() {
        return m_multithread;
    }

    D3D10DeviceLock LockContext() {
      return m_multithread.AcquireLock();
    }

    void InjectCsChunk(
            DxvkCsQueue                 Queue,
            DxvkCsChunkRef&&            Chunk,
            bool                        Synchronize);

    template<typename Fn>
    void InjectCs(
            DxvkCsQueue                 Queue,
            Fn&&                        Command) {
      auto chunk = AllocCsChunk();
      chunk->push(std::move(Command));

      InjectCsChunk(Queue, std::move(chunk), false);
    }

  private:
    
    DxvkCsThread            m_csThread;
    uint64_t                m_csSeqNum = 0ull;

    uint32_t                m_mappedImageCount = 0u;

    Rc<sync::CallbackFence> m_submissionFence;
    uint64_t                m_submissionId = 0ull;
    DxvkSubmitStatus        m_submitStatus;

    uint64_t                m_flushSeqNum = 0ull;
    GpuFlushTracker         m_flushTracker;

    Rc<sync::Fence>         m_stagingBufferFence;

    VkDeviceSize            m_discardMemoryCounter = 0u;
    VkDeviceSize            m_discardMemoryOnFlush = 0u;

    // blessed: cb-ring -- see blessed_cb_ring.h; empty unless d3d11.blessedCbRing
    // blessed: threaded-fe-2 -- its own lines: with the threaded front end
    // the game thread bumps it while the front end writes the fields around
    alignas(CACHE_LINE_SIZE)
    BlessedCbRing           m_blessedCbRing;
    alignas(CACHE_LINE_SIZE)
    DxvkResourceAllocation* m_blessedCbRenameBlock = nullptr;
    // blessed: threaded-fe-2 -- a recorded Present's ring frame, counted
    // game side; the next EndFrame signals it instead of counting one
    uint64_t                m_blessedCbRingPresetFrame = 0u;

    D3D10Multithread        m_multithread;
    D3D11VideoContext       m_videoContext;

    Com<D3D11DeviceContextState, false> m_stateObject;

    D3DDestructionNotifier  m_destructionNotifier;

    std::string             m_flushReason;

    bool                    m_hasPendingUnresolvedPass = false;

    // blessed: flush-pass-end -- BLESSED_FLUSH_AT_PASS_END, cached at
    // construction; and whether render targets are bound (set at each
    // NotifyRenderPassBoundary, cleared at EndFrame), approximating "a
    // render pass is open" for ConsiderFlush. See blessed_flush_pass_end.h.
    bool                    m_blessedFlushPassEnd   = false;
    bool                    m_blessedRenderPassOpen = false;

    bool BlessedDeferFlush(GpuFlushType FlushType, uint64_t ChunkId, uint64_t CompletedSubmissionId);

    HRESULT MapBuffer(
            D3D11Buffer*                pResource,
            D3D11_MAP                   MapType,
            UINT                        MapFlags,
            D3D11_MAPPED_SUBRESOURCE*   pMappedResource);
    
    // blessed: cb-ring -- Map(WRITE_DISCARD) through the ring; false means
    // take the upstream path. Defined in blessed_cb_ring.cpp.
    bool BlessedMapCbRing(
            D3D11Buffer*                pResource,
            D3D11_MAPPED_SUBRESOURCE*   pMappedResource);

    void BlessedEmitCbRename(
            D3D11Buffer*                pResource,
      const BlessedCbRingChunk&         chunk);

    void BlessedEndFrameCbRing();

    // blessed: cb-mirror -- resolves a ring block's mirror; see
    // BlessedCbRing::findMirror. Private like the rest of the ring: reached
    // through D3D11ThreadedContext::ReplayFindCbMirror, same as
    // BlessedEmitCbRename goes through ReplayCbRename.
    DxvkResourceAllocation* BlessedFindCbMirror(DxvkResourceAllocation* block) {
      return m_blessedCbRing.findMirror(block);
    }

    // blessed: cb-mirror -- whether mirroring is on; see
    // BlessedCbRing::mirrorEnabled.
    bool BlessedCbMirrorEnabled() const {
      return m_blessedCbRing.mirrorEnabled();
    }

    // blessed: cb-mirror -- flushes pending mirror copies now and splits
    // the command list; see DxvkContext::blessedFlushCbMirrorEarly and the
    // BlessedMapCbRing call site. Defined in blessed_cb_ring.cpp.
    void BlessedFlushCbMirrorEarly();

    HRESULT MapImage(
            D3D11CommonTexture*         pResource,
            UINT                        Subresource,
            D3D11_MAP                   MapType,
            UINT                        MapFlags,
            D3D11_MAPPED_SUBRESOURCE*   pMappedResource);
    
    void UnmapImage(
            D3D11CommonTexture*         pResource,
            UINT                        Subresource);
    
    void ReadbackImageBuffer(
            D3D11CommonTexture*         pResource,
            UINT                        Subresource);

    void UpdateDirtyImageRegion(
            D3D11CommonTexture*         pResource,
            UINT                        Subresource,
      const D3D11_COMMON_TEXTURE_REGION* pRegion);

    void UpdateMappedBuffer(
            D3D11Buffer*                pDstBuffer,
            UINT                        Offset,
            UINT                        Length,
      const void*                       pSrcData,
            UINT                        CopyFlags);

    void SynchronizeDevice();

    void EndFrame(
            Rc<DxvkLatencyTracker>      LatencyTracker);
    
    bool WaitForResource(
      const DxvkPagedResource&          Resource,
            uint64_t                    SequenceNumber,
            D3D11_MAP                   MapType,
            UINT                        MapFlags);
    
    void EmitCsChunk(DxvkCsChunkRef&& chunk);

    void TrackTextureSequenceNumber(
            D3D11CommonTexture*         pResource,
            UINT                        Subresource);

    void TrackBufferSequenceNumber(
            D3D11Buffer*                pResource);

    uint64_t GetCurrentSequenceNumber();

    uint64_t GetPendingCsChunks();

    void ApplyDirtyNullBindings();

    void ConsiderFlush(
            GpuFlushType                FlushType);

    void ExecuteFlush(
            GpuFlushType                FlushType,
            HANDLE                      hEvent,
            BOOL                        Synchronize);

    void ThrottleAllocation();

    void ThrottleDiscard(
            VkDeviceSize                Size);

    void NotifyRenderPassBoundary(
            bool                        IsMultisampled);

    void NotifyResolve();

    void RequestFlush(
            D3D11_CONTEXT_TYPE          ContextType,
            HANDLE                      hEvent);

    DxvkStagingBufferStats GetStagingMemoryStatistics();

    static GpuFlushType GetMaxFlushType(
            D3D11Device*    pParent,
      const Rc<DxvkDevice>& Device);

  };
  
}
