// blessed: threaded-fe-2 -- stage 2: Present recorded on the game thread and replayed on the front end, with a one-frame cap
#include <cmath>
#include <cstring>

#include "blessed_threaded_context.h"

#include "d3d11_device.h"
#include "d3d11_swapchain.h"

#if defined(DXVK_ARCH_X86)
  #ifdef _MSC_VER
  #include <intrin.h>
  #else
  #include <x86intrin.h>
  #endif
#endif

namespace dxvk {

  /**
   * \brief A recorded Present
   *
   * The swap chain is held by a raw pointer: its destructor drains the
   * front end first, so no record can outlive it. Dirty rects follow the
   * record inline.
   */
  struct D3D11ThreadedContext::PresentRec {
    BlessedFeThunk          fn;
    D3D11ThreadedContext*   parent;
    D3D11SwapChain*         swapChain;
    uint64_t                seq;
    uint64_t                cbRingFrame;
    UINT                    syncInterval;
    UINT                    presentFlags;
    uint32_t                hasParams;
    uint32_t                hasScrollRect;
    uint32_t                hasScrollOffset;
    uint32_t                rectCount;
    RECT                    scrollRect;
    POINT                   scrollOffset;

    /// Dirty rects a record carries; more take the stage 1 path
    constexpr static uint32_t MaxRects = 64u;

    static size_t Size(uint32_t RectCount) {
      return (sizeof(PresentRec) + RectCount * sizeof(RECT) + 7u) & ~size_t(7u);
    }

    RECT* rects() { return reinterpret_cast<RECT*>(this + 1); }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<PresentRec*>(pRecord);

      DXGI_PRESENT_PARAMETERS params = { };
      params.DirtyRectsCount = rec->rectCount;
      params.pDirtyRects     = rec->rectCount ? rec->rects() : nullptr;
      params.pScrollRect     = rec->hasScrollRect ? &rec->scrollRect : nullptr;
      params.pScrollOffset   = rec->hasScrollOffset ? &rec->scrollOffset : nullptr;

      if (rec->cbRingFrame)
        D3D11ThreadedContext::ReplayCbRingFrame(ctx, rec->cbRingFrame);

      // The game side waits on this Present's result: publish one even
      // if the body throws
      uint64_t frameId = 0u;
      HRESULT hr = E_FAIL;

      try {
        hr = rec->swapChain->BlessedReplayPresent(rec->syncInterval,
          rec->presentFlags, rec->hasParams ? &params : nullptr, &frameId);
      } catch (const DxvkError& e) {
        Logger::err(str::format("d3d11.blessedThreadedPresent: ", e.message()));
      }

      rec->parent->FinishPresent(rec->seq, hr, frameId);
      return Size(rec->rectCount);
    }
  };


  void BlessedProbePresent(DxvkDevice* pDevice) {
    // Drives the 120-present window and, when a window completes, writes
    // one line to BLESSED_PROBE_DIR/probe-frames.jsonl
    DxvkStatCounters statCounters = pDevice->getStatCounters();

    blessed::CsSnapshot csSnapshot;
    csSnapshot.idleTicks     = statCounters.getCtr(DxvkStatCounter::CsIdleTicks);
    csSnapshot.chunkCount    = statCounters.getCtr(DxvkStatCounter::CsChunkCount);
    csSnapshot.chunkCmdCount = statCounters.getCtr(DxvkStatCounter::CsChunkCmdCount);
    csSnapshot.syncCount     = statCounters.getCtr(DxvkStatCounter::CsSyncCount);
    csSnapshot.syncTicks     = statCounters.getCtr(DxvkStatCounter::CsSyncTicks);

    blessed::onPresent(csSnapshot);
  }


  bool D3D11ThreadedContext::RecordsPresent(
          UINT                      PresentFlags,
    const DXGI_PRESENT_PARAMETERS*  pPresentParameters) const {
    if (!m_threadedPresent || (PresentFlags & DXGI_PRESENT_TEST))
      return false;

    // Odd parameters keep the stage 1 path rather than a bigger record
    return !pPresentParameters
        || (pPresentParameters->DirtyRectsCount <= PresentRec::MaxRects
         && (!pPresentParameters->DirtyRectsCount || pPresentParameters->pDirtyRects));
  }


  HRESULT D3D11ThreadedContext::RecordPresent(
          D3D11SwapChain*           pSwapChain,
          UINT                      SyncInterval,
          UINT                      PresentFlags,
    const DXGI_PRESENT_PARAMETERS*  pPresentParameters,
          uint64_t*                 pFrameId) {
    D3D10DeviceLock lock = LockContext();

    uint32_t rectCount = pPresentParameters ? pPresentParameters->DirtyRectsCount : 0u;
    uint64_t seq = ++m_presentSeq;

    auto rec = static_cast<PresentRec*>(AllocRecord(PresentRec::Size(rectCount)));
    rec->fn              = &PresentRec::Replay;
    rec->parent          = this;
    rec->swapChain       = pSwapChain;
    rec->seq             = seq;
    rec->syncInterval    = SyncInterval;
    rec->presentFlags    = PresentFlags;
    rec->hasParams       = pPresentParameters != nullptr;
    rec->hasScrollRect   = pPresentParameters && pPresentParameters->pScrollRect;
    rec->hasScrollOffset = pPresentParameters && pPresentParameters->pScrollOffset;
    rec->rectCount       = rectCount;
    rec->scrollRect      = rec->hasScrollRect ? *pPresentParameters->pScrollRect : RECT();
    rec->scrollOffset    = rec->hasScrollOffset ? *pPresentParameters->pScrollOffset : POINT();

    if (rectCount)
      std::memcpy(rec->rects(), pPresentParameters->pDirtyRects, rectCount * sizeof(RECT));

    // The cb ring's frame ends here, on the thread that retires its blocks
    // (blessed_cb_ring.h): EndFrame on the front end signals this number
    rec->cbRingFrame = m_cbRing ? m_ctx->m_blessedCbRing.endFrame() : 0u;

    Publish();
    PlaceFrontEnd();

    if (blessed::enabled())
      TakeSnapshot();

    // The cap: wait while the front end is more than one Present behind
    WaitForPresent(seq - 1u);

    // Newest result we know: this Present's own if the front end already
    // ran it (always in loopback), else the one before it, and then this
    // one's frame id is predicted as the next (it is unless it fails).
    // 0 means not known yet.
    uint64_t done = m_presentDone.load(std::memory_order_acquire);

    if (done >= seq) {
      const PresentResult& result = m_presentResults[seq % PresentSlots];
      *pFrameId = result.frameId;
      return result.hr;
    }

    const PresentResult& result = m_presentResults[(seq - 1u) % PresentSlots];
    *pFrameId = result.frameId ? result.frameId + 1u : 0u;
    return result.hr;
  }


  void D3D11ThreadedContext::WaitForPresent(uint64_t Target) {
    if (likely(m_presentDone.load(std::memory_order_acquire) >= Target))
      return;

    bool timed = blessed::active();
    int64_t t0 = timed ? blessed::stamp() : 0;

    // Spin briefly: the front end is usually just finishing the previous
    // frame's records. Then sleep; it sets the event when it is done.
    uint32_t spins = 0u;

    while (m_presentDone.load(std::memory_order_acquire) < Target) {
      if (spins < 2048u) {
        WakeFrontEnd();
        spins += 1u;
#if defined(DXVK_ARCH_X86)
        _mm_pause();
#endif
        continue;
      }

      m_presentWaiting.store(1u, std::memory_order_seq_cst);

      if (m_presentDone.load(std::memory_order_seq_cst) < Target)
        WaitForSingleObject(m_presentEvent, 1u);

      m_presentWaiting.store(0u, std::memory_order_relaxed);
      WakeFrontEnd();
    }

    if (timed)
      blessed::addFePresentWaitSample(blessed::stamp() - t0);
  }


  void D3D11ThreadedContext::FinishPresent(uint64_t Seq, HRESULT Hr, uint64_t FrameId) {
    // Front end: the result first, then the number that publishes it
    PresentResult& result = m_presentResults[Seq % PresentSlots];
    result.hr      = Hr;
    result.frameId = FrameId;

    m_presentDone.store(Seq, std::memory_order_seq_cst);

    if (m_presentWaiting.load(std::memory_order_seq_cst) && m_presentEvent)
      SetEvent(m_presentEvent);
  }


  HRESULT D3D11SwapChain::BlessedRecordPresent(
          D3D11ThreadedContext*     pFrontEnd,
          UINT                      SyncInterval,
          UINT                      PresentFlags,
    const DXGI_PRESENT_PARAMETERS*  pPresentParameters) {
    blessed::FeScope feScope(blessed::FeCall::Present);

    // The probe's frame boundary stays with the thread it times
    if (blessed::enabled())
      BlessedProbePresent(m_device.ptr());

    uint64_t frameId = 0u;
    HRESULT hr = pFrontEnd->RecordPresent(this, SyncInterval, PresentFlags, pPresentParameters, &frameId);

    // The latency sleep paces this thread, the one that samples input,
    // as upstream does after its Present: for the frame after this one
    if (hr == S_OK && m_latency && frameId)
      m_latency->sleepAndBeginFrame(frameId + 1u, std::abs(m_targetFrameRate));

    return hr;
  }


  HRESULT D3D11SwapChain::BlessedReplayPresent(
          UINT                      SyncInterval,
          UINT                      PresentFlags,
    const DXGI_PRESENT_PARAMETERS*  pPresentParameters,
          uint64_t*                 pFrameId) {
    // IsReplaying() is set, so this runs the upstream body
    HRESULT hr = Present(SyncInterval, PresentFlags, pPresentParameters);
    *pFrameId = m_frameId;
    return hr;
  }

}
