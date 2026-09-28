// blessed: threaded-fe -- the front end's machinery: ring, publish, drain, the replay thread, deferred releases, and the side interfaces
#include <cstdlib>
#include <string>

#include "blessed_crash_log.h"
#include "blessed_threaded_context.h"
#include "blessed_threaded_packet.h"
#include "blessed_threaded_fe_diag.h" // blessed: fe-getters

#include "d3d11_device.h"

#include "../util/util_env.h"

#if defined(DXVK_ARCH_X86)
  #ifdef _MSC_VER
  #include <intrin.h>
  #else
  #include <x86intrin.h>
  #endif
#endif

namespace dxvk {

  bool g_blessedDeferRelease = false;

  thread_local bool D3D11ThreadedContext::t_replaying = false;


  bool BlessedDeferRelease(
          D3D11Device*      pDevice,
          void*             pObject,
          BlessedReleaseFn  pfnRelease) {
    D3D11ThreadedContext* frontEnd = pDevice->BlessedFrontEnd();

    if (!frontEnd)
      return false;

    frontEnd->DeferRelease(pObject, pfnRelease);
    return true;
  }


  static inline void BlessedFeSpinPause() {
#if defined(DXVK_ARCH_X86)
    _mm_pause();
#endif
  }


  static inline void BlessedFeStoreFence() {
    // The engine may write mapped cbuffers with non-temporal stores, and
    // a release store does not order those on x86. Three hooks read the
    // mapped bytes on the front end at draw time.
#if defined(DXVK_ARCH_X86)
    _mm_sfence();
#endif
  }


  /// Pad record: sends the reader back to the start of the ring
  static size_t BlessedFePadReplay(D3D11ImmediateContext*, void*) {
    return 0u;
  }


  /// A batch of final releases, see DeferRelease
  struct BlessedFeReleaseRec {
    BlessedFeThunk  fn;
    uint32_t        count;
    uint32_t        pad;

    struct Entry {
      void*             object;
      BlessedReleaseFn  release;
    } entries[1];

    constexpr static uint32_t MaxEntries = 256u;

    static size_t Replay(D3D11ImmediateContext*, void* pRecord) {
      auto rec = static_cast<BlessedFeReleaseRec*>(pRecord);

      for (uint32_t i = 0; i < rec->count; i++)
        rec->entries[i].release(rec->entries[i].object);

      return (sizeof(BlessedFeReleaseRec) + sizeof(Entry) * (rec->count - 1u) + 7u) & ~size_t(7u);
    }
  };


  D3D11ThreadedContext::D3D11ThreadedContext(
          D3D11Device*            pParent,
          D3D11ImmediateContext*  pContext,
          bool                    Loopback)
  : D3D11DeviceChild<ID3D11DeviceContext4>(pParent),
    m_ctx               (pContext),
    m_loopback          (Loopback),
    m_threadedPresent   (pParent->GetOptions()->blessedThreadedPresent),
    m_cbRing            (pParent->GetOptions()->blessedCbRing),
    m_compact           (pParent->GetOptions()->blessedFrontEndCompact),
    m_publishEvery      (m_compact ? PublishEveryCompact : PublishEvery),
    m_multithread       (this, false, pParent->GetOptions()->enableContextLock),
    m_destructionNotifier(this),
    m_ext               (this),
    m_annotation        (this, pContext->m_annotation.GetStatus()) {
    m_ring = static_cast<char*>(_aligned_malloc(RingSize, CACHE_LINE_SIZE));

    if (!m_ring)
      throw DxvkError("d3d11.blessedThreadedFrontEnd: failed to allocate the ring");

    m_writeLimit = RingSize;

    m_shadow = new FeShadow();
    m_shadow->invalidate();
    m_verifyOmShadow = blessed::FeShadowVerifyEnabled(); // blessed: fe-getters
    // blessed: fe-getters -- off unless asked for: with the shadow on, 13 of 47
    // whiterun runs lost the device (c62-c65), against 0 of 69 with it off or
    // absent; it bought no fps (c62). the race is not found yet.
    m_noOmShadow = env::getEnvVar("BLESSED_FE_OM_SHADOW") != "1";

    m_releaseList.reserve(1024u);
    m_releaseTake.reserve(1024u);

    blessed::feInit();

    if (!m_loopback) {
      m_wakeEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

      if (!m_wakeEvent)
        throw DxvkError("d3d11.blessedThreadedFrontEnd: failed to create the wake event");

      m_presentEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

      if (!m_presentEvent)
        throw DxvkError("d3d11.blessedThreadedFrontEnd: failed to create the present event");

      m_thread = dxvk::thread([this] { FrontEndThread(); });
    }

    // blessed: fe-crash-2 -- the crash report dumps the ring and packet
    // state; registered last, so a throw above leaves nothing dangling
    BlessedCrashLogSetExtra(&D3D11ThreadedContext::WriteCrashForensics, this);

    Logger::info(str::format("d3d11.blessedThreadedFrontEnd: on, ",
      m_loopback ? "loopback (stage 0, replay on the recording thread)" : "threaded",
      m_threadedPresent ? ", Present on the front end (stage 2)" : ", Present drains (stage 1)",
      m_compact ? ", compact records (stage 3)" : "",
      ", ring ", RingSize >> 20u, " MiB"));
  }


  D3D11ThreadedContext::~D3D11ThreadedContext() {
    // Replay everything, including releases queued by destructors that
    // ran during the replay, before the real context goes away.
    if (!this_thread::isInModuleDetachment()) {
      do {
        Drain(blessed::FeDrain::Other);
      } while (m_releasePending.load(std::memory_order_acquire));
    }

    if (m_thread.joinable()) {
      m_stop.store(true, std::memory_order_release);
      SetEvent(m_wakeEvent);
      m_thread.join();
    }

    if (m_wakeEvent)
      CloseHandle(m_wakeEvent);

    if (m_presentEvent)
      CloseHandle(m_presentEvent);

    BlessedCrashLogClearExtra(this); // blessed: fe-crash-2

    delete m_shadow;

    _aligned_free(m_ring);
  }


  ULONG STDMETHODCALLTYPE D3D11ThreadedContext::AddRef() {
    // Upstream D3D11DeviceChild behaviour. The facade's own final release
    // must not go through the ring it owns.
    uint32_t refCount = m_refCount++;

    if (unlikely(!refCount)) {
      AddRefPrivate();
      GetParentInterface()->AddRef();
    }

    return refCount + 1;
  }


  ULONG STDMETHODCALLTYPE D3D11ThreadedContext::Release() {
    uint32_t refCount = --m_refCount;

    if (unlikely(!refCount)) {
      // Replay everything while our device reference still holds the
      // device: a destructor run by a release record may drop another
      // object's device reference, and the last one must never drop on
      // the front end thread (the device's destructor joins it).
      { D3D10DeviceLock lock = LockContext();
        Drain(blessed::FeDrain::Other);
      }

      auto* parent = GetParentInterface();
      ReleasePrivate();
      parent->Release();
    }

    return refCount;
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::QueryInterface(REFIID riid, void** ppvObject) {
    blessed::FeScope feScope(blessed::FeCall::QueryInterface);

    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(ID3D11DeviceChild)
     || riid == __uuidof(ID3D11DeviceContext)
     || riid == __uuidof(ID3D11DeviceContext1)
     || riid == __uuidof(ID3D11DeviceContext2)
     || riid == __uuidof(ID3D11DeviceContext3)
     || riid == __uuidof(ID3D11DeviceContext4)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(ID3D11VkExtContext)
     || riid == __uuidof(ID3D11VkExtContext1)) {
      *ppvObject = ref(&m_ext);
      return S_OK;
    }

    if (riid == __uuidof(ID3DUserDefinedAnnotation)
     || riid == __uuidof(IDXVKUserDefinedAnnotation)) {
      *ppvObject = ref(&m_annotation);
      return S_OK;
    }

    if (riid == __uuidof(ID3D10Multithread)) {
      *ppvObject = ref(&m_multithread);
      return S_OK;
    }

    if (riid == __uuidof(ID3DDestructionNotifier)) {
      *ppvObject = ref(&m_destructionNotifier);
      return S_OK;
    }

    if (riid == __uuidof(ID3D11VideoContext)) {
      // The video context drives the real context from the calling
      // thread; not wired through the front end.
      static bool s_warned = false;

      if (!std::exchange(s_warned, true))
        Logger::warn("d3d11.blessedThreadedFrontEnd: ID3D11VideoContext is not available with the threaded front end");

      return E_NOINTERFACE;
    }

    if (logQueryInterfaceError(__uuidof(ID3D11DeviceContext), riid)) {
      Logger::warn("D3D11ThreadedContext::QueryInterface: Unknown interface query");
      Logger::warn(str::format(riid));
    }

    return E_NOINTERFACE;
  }


  void* D3D11ThreadedContext::AllocRecordSlow(size_t Size) {
    // Every wait below can publish, and a publish can move queued
    // releases into the ring, so the write position is re-read each turn.
    CheckProducerThread(); // blessed: fe-crash-2, see AllocRecord

    for (;;) {
      uint64_t pos = m_writePos;
      uint64_t boundary = (pos | uint64_t(RingSize - 1u)) + 1u;
      uint64_t replayed = m_replayed.load(std::memory_order_acquire);

      if (pos + Size > boundary) {
        // The record would straddle the end: pad to it. Needs the rest
        // of this lap free, which it is once the reader is a lap behind.
        if (boundary > replayed + RingSize) {
          WaitForReplayed(boundary - RingSize);
          continue;
        }

        *reinterpret_cast<BlessedFeThunk*>(m_ring + (pos & (RingSize - 1u))) = &BlessedFePadReplay;
        m_writePos = boundary;
        continue;
      }

      if (pos + Size > replayed + RingSize) {
        WaitForReplayed(pos + Size - RingSize);
        continue;
      }

      m_writePos = pos + Size;
      m_records += 1u;
      UpdateWriteLimit(replayed);
      return m_ring + (pos & (RingSize - 1u));
    }
  }


  void D3D11ThreadedContext::UpdateWriteLimit(uint64_t Replayed) {
    // The fast path's single compare: stay inside this lap of the ring
    // and inside what the reader has freed.
    // blessed: fe-crash-2 -- "this lap" is the lap of the last byte
    // written, not of m_writePos. When the record just allocated ends
    // exactly on the ring's end, m_writePos is the next lap's first byte,
    // and a limit a whole lap further let PushOp keep growing the packet
    // that ends there: its ops went to the ring's start (pos & mask)
    // while the reader walked the packet on past the ring's end, into the
    // bytes after the allocation (the null CbRename, c43's five dumps).
    // Now the limit stops at that ring end: the next op or record takes
    // the slow path, which opens its packet at the ring's start. One
    // extra slow call per 4 MiB lap.
    uint64_t last = m_writePos ? m_writePos - 1u : 0u;
    uint64_t boundary = (last | uint64_t(RingSize - 1u)) + 1u;
    m_writeLimit = std::min(boundary, Replayed + RingSize);
  }


  void D3D11ThreadedContext::WaitForReplayed(uint64_t Target) {
    bool timed = blessed::active();
    int64_t n0 = blessed::g_feNestedTicks;
    int64_t t0 = timed ? blessed::stamp() : 0;

    Publish();

    if (!m_loopback) {
      uint32_t spins = 0u;

      while (m_replayed.load(std::memory_order_acquire) < Target) {
        WakeFrontEnd();
        BlessedFeSpinPause();

        if (++spins >= 4096u) {
          this_thread::yield();
          spins = 0u;
        }
      }
    }

    if (timed)
      blessed::addFeRingFullSample(blessed::stamp() - t0 - (blessed::g_feNestedTicks - n0));
  }


  void D3D11ThreadedContext::Publish() {
    // blessed: threaded-fe-2 -- the front end may read the open packet
    // from here on, so it takes no more ops
    uint64_t packetEnd = std::exchange(m_packetEnd, ~0ull);

    if (unlikely(m_releasePending.load(std::memory_order_relaxed)))
      FlushReleases();

    m_unpublished = 0u;

    uint64_t end = m_writePos;

    if (end == m_publishedLocal)
      return;

    // blessed: fe-crash-2 -- the crash report's publish log: plain stores
    // to producer-owned memory, no fence. A publish from a thread other
    // than the one recording (a device-side drain, a second caller) is
    // counted: the protocol has no lock and allows only one producer.
    DWORD thread = GetCurrentThreadId();

    if (unlikely(thread != m_producerThreadId))
      ReportForeignPublish(thread);

    FePublishEvent& event = m_publishLog[m_publishes & (PublishLogSize - 1u)];
    event.writePos      = end;
    event.prevPublished = m_publishedLocal;
    event.packetEnd     = packetEnd;
    event.packet        = m_packet;
    event.size          = m_packet ? m_packet->size : 0u;
    event.count         = m_packet ? m_packet->count : 0u;
    event.pending       = m_packetPending;
    event.thread        = thread;

    BlessedFeStoreFence();
    m_publishedLocal = end;
    m_published.store(end, std::memory_order_release);
    m_publishes += 1u;

    if (m_loopback) {
      // Stage 0: the recording thread is its own front end
      bool timed = blessed::active();
      int64_t t0 = timed ? blessed::stamp() : 0;

      uint64_t begin = m_replayed.load(std::memory_order_relaxed);
      m_replayed.store(ReplayRange(begin, end), std::memory_order_release);
      UpdateWriteLimit(end);

      if (timed)
        blessed::addFeReplaySample(blessed::stamp() - t0);
    } else {
      if (unlikely(blessed::active()))
        blessed::addFeOccupancy(end - m_replayed.load(std::memory_order_relaxed));

      WakeFrontEnd();
    }
  }


  void D3D11ThreadedContext::ReportProducerThreadChange(DWORD thread) {
    // blessed: fe-crash -- m_writePos/m_packet/m_packetPending have no
    // lock (m_multithread's is a no-op unless the app opts in), and
    // PlaceFrontEnd's own comment already notes the recording thread can
    // change (a loading screen hands over to the render thread). A clean,
    // sequential handoff is fine; two threads live at once here is not --
    // this logs it (see below for how often), with whether a packet
    // was still open (unpublished) across the switch, which is the
    // window a corrupted packet (an op whose fields never arrived) would
    // come from. The common (same-thread) case never reaches here: see
    // CheckProducerThread in blessed_threaded_context.h.
    if (likely(!m_producerThreadId)) {
      m_producerThreadId = thread;
      return;
    }

    // blessed: fe-crash-2 -- every switch is counted and kept in a small
    // log for the crash report. The line below used to print once per
    // process, which read as "one handoff" even if the recording thread
    // changed many times; now it prints the first eight switches, then
    // each power of two, so a ping-pong between two threads is visible.
    bool packetOpen = m_packet != nullptr && m_packetEnd != ~0ull;
    uint32_t n = ++m_threadSwitches;

    FeSwitchEvent& event = m_switchLog[(n - 1u) & (SwitchLogSize - 1u)];
    event.writePos  = m_writePos;
    event.publishes = m_publishes;
    event.packetEnd = m_packetEnd;
    event.from      = m_producerThreadId;
    event.to        = thread;

    if (n <= 8u || !(n & (n - 1u))) {
      Logger::err(str::format("d3d11.blessedFeCheck: recording thread switch #", n, ": thread ", thread,
        " after thread ", m_producerThreadId, " at write position ", m_writePos, " (publish ", m_publishes, "); packet ",
        packetOpen ? "was still open (unpublished) across the switch" : "was closed at the switch",
        packetOpen ? str::format(", size ", m_packet->size, ", count ", m_packet->count) : std::string()));
    }

    m_producerThreadId = thread;
  }


  void D3D11ThreadedContext::ReportForeignPublish(DWORD thread) {
    // blessed: fe-crash-2 -- Publish on a thread other than the one that
    // last recorded. Nothing recorded yet: adopt it as the recorder.
    if (!m_producerThreadId) {
      m_producerThreadId = thread;
      return;
    }

    uint32_t n = ++m_foreignPublishes;

    if (n <= 8u || !(n & (n - 1u))) {
      Logger::err(str::format("d3d11.blessedFeCheck: publish #", n, " from thread ", thread,
        " while thread ", m_producerThreadId, " records, at write position ", m_writePos,
        ", op reservation pending: ", m_packetPending));
    }
  }


  void* D3D11ThreadedContext::PushOpSlow(size_t Size) {
    // A new packet, opened with its first op reserved but not yet counted:
    // rec->size covers the header alone until Recorded() folds in
    // m_packetPending, once the caller has stored the op's fields (same
    // rule as the fast path in PushOp). AllocRecord may publish (and
    // close any packet) before it returns; this one opens after, so that
    // earlier publish can only ever see complete packets.
    CheckProducerThread();

    auto rec = static_cast<FePacketRec*>(AllocRecord(sizeof(FePacketRec) + Size));
    rec->fn    = &FePacketRec::Replay;
    rec->size  = uint32_t(sizeof(FePacketRec));
    rec->count = 0u;
    rec->mirrorEarlyFlush = false; // blessed: cb-mirror

    m_packet        = rec;
    m_packetEnd     = m_writePos;
    m_packetPending = uint32_t(Size);
    m_packetOps    += 1u;
    return rec + 1;
  }


  void D3D11ThreadedContext::InvalidateShadow() {
    m_shadow->invalidate();
  }


  void D3D11ThreadedContext::WakeFrontEnd() {
    // A store and a load when the front end is awake. A wake-up missed
    // here (the flag still in flight) costs at most the front end's 1 ms
    // sleep timeout, and every producer-side wait loop retries it.
    if (unlikely(m_sleeping.load(std::memory_order_relaxed))) {
      if (m_sleeping.exchange(0u, std::memory_order_acq_rel)) {
        SetEvent(m_wakeEvent);
        m_wakes += 1u;
      }
    }
  }


  void D3D11ThreadedContext::Drain(blessed::FeDrain Reason, blessed::FeCall Call) {
    // blessed: fe-getters -- independent of the probe, works in a noprobe
    // build; one bool check unless BLESSED_FE_DRAIN_STATS=1
    blessed::FeDrainStatsTick(Reason, Call);

    // Timed less any inline replay or ring-full wait inside it, which
    // have their own buckets
    bool timed = blessed::active();
    int64_t n0 = blessed::g_feNestedTicks;
    int64_t t0 = timed ? blessed::stamp() : 0;

    Publish();

    if (!m_loopback) {
      // Equal counters mean the front end is between records, outside
      // the context. Our later writes reach it through the next publish.
      uint64_t target = m_writePos;
      uint32_t spins = 0u;

      while (m_replayed.load(std::memory_order_acquire) != target) {
        WakeFrontEnd();
        BlessedFeSpinPause();

        if (++spins >= 4096u) {
          this_thread::yield();
          spins = 0u;
        }
      }
    }

    if (timed)
      blessed::addFeDrainSample(Reason, blessed::stamp() - t0 - (blessed::g_feNestedTicks - n0));
  }


  void D3D11ThreadedContext::DrainFromDevice(blessed::FeDrain Reason) {
    // blessed: threaded-fe-2 -- device code reached from a replayed
    // record (Present) already runs on the context's owner
    if (t_replaying)
      return;

    blessed::FeScope feScope(blessed::FeCall::DeviceInternal);
    D3D10DeviceLock lock = LockContext();
    Drain(Reason, blessed::FeCall::DeviceInternal);
  }


  void D3D11ThreadedContext::BeginPresent() {
    D3D10DeviceLock lock = LockContext();
    Drain(blessed::FeDrain::Present, blessed::FeCall::Present);
    PlaceFrontEnd();

    if (blessed::enabled())
      TakeSnapshot();
  }


  void D3D11ThreadedContext::TakeSnapshot() {
    blessed::FeSnapshot snapshot;
    snapshot.mode      = m_loopback ? 1u : 2u;
    snapshot.records   = m_records;
    snapshot.bytes     = m_writePos;
    snapshot.publishes = m_publishes;
    snapshot.releases  = m_releases;
    snapshot.idleUs    = m_idleUs.load(std::memory_order_relaxed);
    snapshot.wakes     = m_wakes;
    snapshot.folded    = m_folded;
    snapshot.packetOps = m_packetOps;

    blessed::setFeSnapshot(snapshot);
  }


  void D3D11ThreadedContext::DeferRelease(
          void*             pObject,
          BlessedReleaseFn  pfnRelease) {
    std::lock_guard<sync::Spinlock> lock(m_releaseLock);
    m_releaseList.push_back({ pObject, pfnRelease });
    m_releasePending.store(true, std::memory_order_release);
  }


  void D3D11ThreadedContext::FlushReleases() {
    // Producer only. Each release lands after everything recorded so far;
    // the app ordered its last use of the object before releasing it.
    // Recording below can publish again, which must not re-enter here.
    if (m_releaseTake.size())
      return;

    { std::lock_guard<sync::Spinlock> lock(m_releaseLock);
      m_releasePending.store(false, std::memory_order_relaxed);
      std::swap(m_releaseList, m_releaseTake);
    }

    size_t total = m_releaseTake.size();

    for (size_t first = 0u; first < total; first += BlessedFeReleaseRec::MaxEntries) {
      uint32_t count = uint32_t(std::min<size_t>(total - first, BlessedFeReleaseRec::MaxEntries));

      size_t size = AlignRecord(sizeof(BlessedFeReleaseRec)
        + sizeof(BlessedFeReleaseRec::Entry) * (count - 1u));

      auto rec = static_cast<BlessedFeReleaseRec*>(AllocRecord(size));
      rec->fn    = &BlessedFeReleaseRec::Replay;
      rec->count = count;
      rec->pad   = 0u;

      for (uint32_t i = 0; i < count; i++) {
        rec->entries[i].object  = m_releaseTake[first + i].object;
        rec->entries[i].release = m_releaseTake[first + i].release;
      }
    }

    m_releases += total;
    m_releaseTake.clear();
  }


  uint64_t D3D11ThreadedContext::ReplayRange(uint64_t Begin, uint64_t End) {
    uint64_t pos = Begin;
    uint64_t freed = Begin;

    // blessed: fe-crash-2 -- what this thread is replaying, for the crash
    // report; plain stores to a line only this thread writes
    m_feBegin = Begin;
    m_feEnd   = End;

    // A Present replayed below runs its upstream body, and device-side
    // drains inside it are no-ops: this thread owns the context now
    bool replaying = std::exchange(t_replaying, true);

    try {
      while (pos < End) {
        char* rec = m_ring + (pos & (RingSize - 1u));
        size_t size = (*reinterpret_cast<BlessedFeThunk*>(rec))(m_ctx, rec);

        pos = likely(size)
          ? pos + size
          : (pos | uint64_t(RingSize - 1u)) + 1u;

        // Hand space back to a producer waiting on a full ring
        if (unlikely(pos - freed >= (RingSize >> 4u)) && !m_loopback) {
          m_replayed.store(pos, std::memory_order_release);
          freed = pos;
        }
      }
    } catch (const DxvkError& e) {
      Logger::err(str::format("d3d11.blessedThreadedFrontEnd: replay failed, skipping ",
        End - pos, " bytes of records: ", e.message()));
      pos = End;
    }

    t_replaying = replaying;
    return pos;
  }


  void D3D11ThreadedContext::FrontEndThread() {
    env::setThreadName("dxvk-fe");

    uint64_t pos = 0u;
    bool probe = blessed::enabled();

    while (!m_stop.load(std::memory_order_acquire)) {
      uint64_t end = m_published.load(std::memory_order_acquire);

      if (likely(end != pos)) {
        pos = ReplayRange(pos, end);
        m_replayed.store(pos, std::memory_order_release);
        continue;
      }

      // Idle: spin about 50 us, then sleep until the producer wakes us
      auto t0 = probe ? dxvk::high_resolution_clock::now() : dxvk::high_resolution_clock::time_point();
      auto spinUntil = dxvk::high_resolution_clock::now() + std::chrono::microseconds(50);
      bool work = false;

      while (!work) {
        for (uint32_t i = 0; i < 256u && !work; i++) {
          BlessedFeSpinPause();
          work = m_published.load(std::memory_order_acquire) != pos
              || m_stop.load(std::memory_order_relaxed);
        }

        if (work || dxvk::high_resolution_clock::now() >= spinUntil)
          break;
      }

      if (!work) {
        m_sleeping.store(1u, std::memory_order_seq_cst);

        if (m_published.load(std::memory_order_seq_cst) == pos
         && !m_stop.load(std::memory_order_acquire))
          WaitForSingleObject(m_wakeEvent, 1u);

        m_sleeping.store(0u, std::memory_order_release);
      }

      if (probe) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
          dxvk::high_resolution_clock::now() - t0).count();
        m_idleUs.fetch_add(uint64_t(us), std::memory_order_relaxed);
      }
    }
  }


  ID3D11VkExtContext1* D3D11ThreadedContext::InnerExt() {
    return &m_ctx->m_contextExt;
  }


  // --- vk ext context: drain, then the real context's own ---

  ULONG STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::AddRef() {
    return m_parent->AddRef();
  }


  ULONG STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::Release() {
    return m_parent->Release();
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::QueryInterface(REFIID riid, void** ppvObject) {
    return m_parent->QueryInterface(riid, ppvObject);
  }


#define BLESSED_FE_EXT_DRAIN() \
  blessed::FeScope feScope(blessed::FeCall::ExtContext); \
  D3D10DeviceLock feLock = m_parent->LockContext(); \
  m_parent->Drain(blessed::FeDrain::ExtContext, blessed::FeCall::ExtContext)

  void STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::MultiDrawIndirect(
          UINT DrawCount, ID3D11Buffer* pBufferForArgs, UINT ByteOffsetForArgs, UINT ByteStrideForArgs) {
    BLESSED_FE_EXT_DRAIN();
    m_parent->InnerExt()->MultiDrawIndirect(DrawCount, pBufferForArgs, ByteOffsetForArgs, ByteStrideForArgs);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::MultiDrawIndexedIndirect(
          UINT DrawCount, ID3D11Buffer* pBufferForArgs, UINT ByteOffsetForArgs, UINT ByteStrideForArgs) {
    BLESSED_FE_EXT_DRAIN();
    m_parent->InnerExt()->MultiDrawIndexedIndirect(DrawCount, pBufferForArgs, ByteOffsetForArgs, ByteStrideForArgs);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::MultiDrawIndirectCount(
          UINT MaxDrawCount, ID3D11Buffer* pBufferForCount, UINT ByteOffsetForCount,
          ID3D11Buffer* pBufferForArgs, UINT ByteOffsetForArgs, UINT ByteStrideForArgs) {
    BLESSED_FE_EXT_DRAIN();
    m_parent->InnerExt()->MultiDrawIndirectCount(MaxDrawCount, pBufferForCount,
      ByteOffsetForCount, pBufferForArgs, ByteOffsetForArgs, ByteStrideForArgs);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::MultiDrawIndexedIndirectCount(
          UINT MaxDrawCount, ID3D11Buffer* pBufferForCount, UINT ByteOffsetForCount,
          ID3D11Buffer* pBufferForArgs, UINT ByteOffsetForArgs, UINT ByteStrideForArgs) {
    BLESSED_FE_EXT_DRAIN();
    m_parent->InnerExt()->MultiDrawIndexedIndirectCount(MaxDrawCount, pBufferForCount,
      ByteOffsetForCount, pBufferForArgs, ByteOffsetForArgs, ByteStrideForArgs);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::SetDepthBoundsTest(
          BOOL Enable, FLOAT MinDepthBounds, FLOAT MaxDepthBounds) {
    BLESSED_FE_EXT_DRAIN();
    m_parent->InnerExt()->SetDepthBoundsTest(Enable, MinDepthBounds, MaxDepthBounds);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::SetBarrierControl(
          UINT ControlFlags) {
    BLESSED_FE_EXT_DRAIN();
    m_parent->InnerExt()->SetBarrierControl(ControlFlags);
  }


  bool STDMETHODCALLTYPE D3D11ThreadedContext::ExtContext::LaunchCubinShaderNVX(
          IUnknown* hShader, uint32_t GridX, uint32_t GridY, uint32_t GridZ,
          const void* pParams, uint32_t ParamSize, void* const* pReadResources,
          uint32_t NumReadResources, void* const* pWriteResources, uint32_t NumWriteResources) {
    BLESSED_FE_EXT_DRAIN();
    return m_parent->InnerExt()->LaunchCubinShaderNVX(hShader, GridX, GridY, GridZ,
      pParams, ParamSize, pReadResources, NumReadResources, pWriteResources, NumWriteResources);
  }

#undef BLESSED_FE_EXT_DRAIN

}
