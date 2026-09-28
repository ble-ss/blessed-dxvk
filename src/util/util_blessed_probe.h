// blessed: render-thread + cs-thread timing probe, enabled via BLESSED_PROBE_DIR (unset = fully off)
#pragma once

#include <cstdint>

// blessed: -Dblessed_probe=false (meson) defines this 0, which drops every
// scope and the BLESSED_PROBE_CALL macro to a true no-op below -- no rdtsc,
// no globals, no addXSample calls compiled anywhere. A build with the
// default (true) is unchanged from before this option existed.
#ifndef BLESSED_PROBE_ENABLED
#define BLESSED_PROBE_ENABLED 1
#endif

#if BLESSED_PROBE_ENABLED

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "util_time.h"

#include "util_blessed_fe_census.h" // blessed: threaded-fe

namespace dxvk::blessed {

  // one bucket per instrumented d3d11 immediate-context entry point.
  // stage-specific methods (VSSetConstantBuffers, PSSetConstantBuffers, ...)
  // share a single bucket, as do the ...1 variants where noted.
  enum class Call : uint32_t {
    Map = 0,
    Unmap,
    UpdateSubresource,
    UpdateSubresource1,
    SetConstantBuffers,
    SetShaderResources,
    SetSamplers,
    SetShader,
    IASetVertexBuffers,
    IASetIndexBuffer,
    IASetInputLayout,
    IASetPrimitiveTopology,
    OMSetRenderTargets,          // includes OMSetRenderTargetsAndUnorderedAccessViews
    OMSetBlendState,
    OMSetDepthStencilState,
    RSSetState,
    RSSetViewports,
    RSSetScissorRects,
    Draw,
    DrawIndexed,
    DrawInstanced,
    DrawIndexedInstanced,
    Dispatch,
    CopyResource,
    CopySubresourceRegion,
    ClearRenderTargetView,
    ClearDepthStencilView,
    // sub-buckets inside DrawIndexed: already counted in DrawIndexed, so
    // app_in_dxvk_ms sums only the buckets before SubFirst
    SubFirst,
    DrawIndexedCascadeCheck = SubFirst,
    DrawIndexedSceneCapture,
    DrawIndexedGi,
    DrawIndexedCore,
    DrawIndexedPostHook,
    // blessed: hook-cpu -- finer split inside the two costliest sub-buckets
    // above. SceneCaptureResolve is everything in BlessedSceneCaptureDraw
    // after the selector match (layout/buffer checks, xform/campos read,
    // EmitCs) -- what's left of DrawIndexedSceneCapture once the om-generation
    // cache (see D3D11ContextStateOM::blessedOmGeneration) makes the selector
    // check itself nearly free for draws outside the wanted pass. GiPatch is
    // the same split for BlessedGi::OnDraw: everything after IsMainLitPassBound
    // returns true (mesh/albedo resolve, probe sample, cbuffer patch).
    DrawIndexedSceneCaptureResolve,
    DrawIndexedGiPatch,
    // blessed: hook-cpu-2 -- one level further down, all nested inside the
    // two buckets above. Static: the static path after the skinned routing
    // (checks, xform/campos reads, slices, EmitCs), and its slice + EmitCs
    // tail alone. Skinned: the whole skinned path (validation for every
    // skinned draw, incl. the ones after the mask pass that are counted but
    // never copied), and its staging tail alone (bones into the frame arena
    // + the small record; runs only for draws the tracer may keep).
    // blessed: scene-cs -- the static path's cbuffer reads, slices and
    // hand-off moved to the cs thread (they show in cs.busy_ms): scene_static
    // is now only the layout and D3D11-state checks, and scene_static_emit
    // stays in this list but no longer fires.
    DrawIndexedSceneStatic,
    DrawIndexedSceneStaticEmit,
    DrawIndexedSceneSkinned,
    DrawIndexedSceneSkinnedStage,
    // gi: mesh/albedo dedup (+ EmitCs on a miss), the two cbuffer reads
    // (vs World, ps CameraPosAdjust), the probe-grid sample (only when the
    // world position differs from the previous sampled draw's), and the
    // in-place ps b2 write (a plain store into the mapped slice, no alloc).
    DrawIndexedGiResolve,
    DrawIndexedGiCbRead,
    DrawIndexedGiSample,
    DrawIndexedGiWrite,
    // blessed: hook-cpu-2, round two -- inside scene_skinned_stage: taking
    // the frame's batch (n = batches per frame, should be 1; a high ns means
    // it was allocated, not recycled), record growth (n should be 0 once
    // warm), the four GetBufferSlice calls plus the b10 allocation ref, and
    // the record push_back. Round four dropped the bones buckets: the cpu
    // no longer reads the bones. Inside gi_sample: a cell-cache miss
    // (corner slots + validity worked out for a new cell).
    DrawIndexedSceneSkinnedAcquire,
    DrawIndexedSceneSkinnedGrow,
    DrawIndexedSceneSkinnedSlices,
    DrawIndexedSceneSkinnedPush,
    DrawIndexedGiSampleCell,
    // blessed: gi-cs -- what BlessedGi::OnDraw still does on the app thread
    // once the main lit pass is bound: the mesh/albedo note (gi_resolve,
    // nested) and the once-a-frame config checks. The probe sample and the
    // cbuffer reads/write moved to the cs thread (they show in cs.busy_ms);
    // gi_patch, gi_cbread, gi_sample, gi_write and gi_sample_cell stay in
    // this list but no longer fire.
    DrawIndexedGiEmit,
    Count
  };

  extern const char* const g_callNames[uint32_t(Call::Count)];

  // Map() bucketed by map type x bound resource kind.
  // blessed: cb-ring -- Ring: WRITE_DISCARD served by the constant-buffer ring
  enum class MapType : uint32_t { Discard = 0, NoOverwrite, Other, Ring, Count };

  // blessed: cb-ring -- events of the ring: a block switch, a new block
  // allocated (nested in the switch), a map that fell back upstream, and a
  // rename that opened a new cs command instead of joining the previous one
  enum class CbRingEvent : uint32_t { Advance = 0, NewBlock, Fallback, RenameCmd, Count };
  enum class BindKind : uint32_t { Constant = 0, Vertex, Index, Other, Count };

  BindKind classifyBindFlags(uint32_t bindFlags);

  // set once at dll load from BLESSED_PROBE_DIR. inline so a disabled probe
  // costs one load and a predicted branch per d3d11 call, not a call.
  extern const bool g_enabled;
  inline bool enabled() { return g_enabled; }

  // true while the current frame is being timed. BLESSED_PROBE_EVERY=N
  // times one frame in N (default 1), so the probe's own cost can be kept
  // out of the frames it is not timing. the scopes gate on this.
  extern bool g_active;
  inline bool active() { return g_active; }

  // scopes nested inside another scope (MapBuffer, DiscardSlice, EmitCsChunk,
  // the DrawIndexed.* sub-buckets) add their own stamps to the outer scope's
  // time. BLESSED_PROBE_NEST=0 turns them off, so top-level numbers are clean.
  extern const bool g_nested;
  inline bool nestedActive() { return g_active && g_nested; }

  // raw timestamp for the scopes: rdtsc (invariant tsc, ~8 ns), converted to
  // time once per window against the qpc clock. c5 found the old per-call
  // qpc read + two divisions + microsecond truncation cost 4.6 ms per frame
  // and reported every ~50 ns call as 0.
  inline int64_t stamp() { return int64_t(__rdtsc()); }

  void addCallSample(Call id, int64_t ticks);
  void addMapSample(MapType type, BindKind bind, int64_t ticks);
  void addDiscardSliceSample(int64_t ticks);
  void addCbRingSample(CbRingEvent event, int64_t ticks); // blessed: cb-ring
  // "emit_block" stall bucket only: EmitCsChunk is timed directly since
  // nothing else tracks it. The "sync_cs" bucket is derived instead from the
  // existing CsSyncCount/CsSyncTicks stat counters (see CsSnapshot below),
  // which already measure exactly the wait inside SynchronizeCsThread.
  void addEmitBlockSample(int64_t ticks);

  struct CsSnapshot {
    uint64_t idleTicks    = 0; // microseconds, DxvkStatCounter::CsIdleTicks
    uint64_t chunkCount   = 0; // DxvkStatCounter::CsChunkCount
    uint64_t chunkCmdCount= 0; // DxvkStatCounter::CsChunkCmdCount
    uint64_t syncCount    = 0; // DxvkStatCounter::CsSyncCount
    uint64_t syncTicks    = 0; // microseconds, DxvkStatCounter::CsSyncTicks
  };

  // RAII timing scopes for the immediate-context-only call sites that are not
  // templated (Map/MapBuffer/DiscardSlice/EmitCsChunk all live directly on
  // D3D11ImmediateContext, so there is no deferred-context instantiation to
  // keep zero-cost here; the enabled() branch is the only cost when off).
  struct MapScope {
    MapScope(MapType type, BindKind bind) : m_type(type), m_bind(bind) {
      if (nestedActive())
        m_t0 = stamp();
    }

    ~MapScope() {
      if (nestedActive())
        addMapSample(m_type, m_bind, stamp() - m_t0);
    }

    MapScope             (const MapScope&) = delete;
    MapScope& operator = (const MapScope&) = delete;

    // blessed: cb-ring -- a ring map that fell back upstream is a Discard
    void setType(MapType type) { m_type = type; }

  private:
    MapType  m_type;
    BindKind m_bind;
    int64_t  m_t0 = 0;
  };

  // blessed: cb-ring -- times one ring slow-path event (nested in Map)
  struct CbRingScope {
    explicit CbRingScope(CbRingEvent event) : m_event(event) {
      if (nestedActive())
        m_t0 = stamp();
    }

    ~CbRingScope() {
      if (nestedActive())
        addCbRingSample(m_event, stamp() - m_t0);
    }

    CbRingScope             (const CbRingScope&) = delete;
    CbRingScope& operator = (const CbRingScope&) = delete;

  private:
    CbRingEvent m_event;
    int64_t     m_t0 = 0;
  };

  struct DiscardSliceScope {
    DiscardSliceScope() {
      if (nestedActive())
        m_t0 = stamp();
    }

    ~DiscardSliceScope() {
      if (nestedActive())
        addDiscardSliceSample(stamp() - m_t0);
    }

    DiscardSliceScope             (const DiscardSliceScope&) = delete;
    DiscardSliceScope& operator = (const DiscardSliceScope&) = delete;

  private:
    int64_t m_t0 = 0;
  };

  struct EmitBlockScope {
    EmitBlockScope() {
      if (nestedActive())
        m_t0 = stamp();
    }

    ~EmitBlockScope() {
      if (nestedActive())
        addEmitBlockSample(stamp() - m_t0);
    }

    EmitBlockScope             (const EmitBlockScope&) = delete;
    EmitBlockScope& operator = (const EmitBlockScope&) = delete;

  private:
    int64_t m_t0 = 0;
  };

  // blessed: threaded-fe -- the front end's census and buckets. Written on
  // the game (recording) thread only; the front end thread keeps its own
  // counters and hands them over in FeSnapshot. See
  // d3d11/blessed_threaded_context.h for what each one measures.
  void feInit();
  void addFeCallSample(FeCall call, int64_t ticks, bool drained);
  void addFeDrainSample(FeDrain reason, int64_t ticks);
  void addFeRingFullSample(int64_t ticks);
  void addFeReplaySample(int64_t ticks);
  void addFePresentWaitSample(int64_t ticks); // blessed: threaded-fe-2
  void addFeOccupancy(uint64_t bytes);

  struct FeSnapshot {
    uint32_t mode      = 0; // 0 off, 1 loopback, 2 threaded
    uint64_t records   = 0; // cumulative from here on
    uint64_t bytes     = 0;
    uint64_t publishes = 0;
    uint64_t releases  = 0; // deferred final releases
    uint64_t idleUs    = 0; // front end thread idle (spinning or asleep)
    uint64_t wakes     = 0; // times the game thread woke a sleeping front end
    uint64_t folded    = 0; // blessed: threaded-fe-2, redundant binds not recorded
    uint64_t packetOps = 0; // blessed: threaded-fe-2, calls packed into draw packets
  };

  // cumulative counters, once per Present, before onPresent
  void setFeSnapshot(const FeSnapshot& fe);

  // ticks spent in drains, inline replay and ring-full waits so far; the
  // drain/replay/ring_full samples add to it, and FeScope subtracts what
  // piled up inside it, so the census holds each entry's own time
  extern int64_t g_feNestedTicks;

  // drains so far; an entry that drained ran dxvk's own method directly,
  // and its time goes to the direct bucket as well as the census
  extern uint64_t g_feDrainCount;

  // times one facade entry, less any drain, inline replay or ring-full
  // wait inside it (those have their own buckets)
  struct FeScope {
    explicit FeScope(FeCall call) : FeScope(call, true) { }

    // blessed: threaded-fe-2 -- On false counts nothing (a Present body
    // replayed on the front end thread)
    FeScope(FeCall call, bool On) : m_call(call), m_on(On && active()) {
      if (m_on) {
        m_nested0 = g_feNestedTicks;
        m_drains0 = g_feDrainCount;
        m_t0 = stamp();
      }
    }

    ~FeScope() {
      if (m_on) {
        addFeCallSample(m_call, stamp() - m_t0 - (g_feNestedTicks - m_nested0),
          g_feDrainCount != m_drains0);
      }
    }

    FeScope             (const FeScope&) = delete;
    FeScope& operator = (const FeScope&) = delete;

  private:
    FeCall   m_call;
    bool     m_on;
    int64_t  m_t0 = 0;
    int64_t  m_nested0 = 0;
    uint64_t m_drains0 = 0;
  };

  // call once per Present, from the app thread. Drives the 120-frame window
  // and writes one jsonl line to BLESSED_PROBE_DIR/probe-frames.jsonl when a
  // window completes. No-op when disabled.
  void onPresent(const CsSnapshot& cs);

  // RAII timing scope for a single call bucket. ImmediateOnly selects whether
  // this instantiation actually measures anything; pass
  // std::is_same_v<ContextType, D3D11ImmediateContext> from templated call
  // sites so deferred-context instantiations compile to nothing.
  template<bool ImmediateOnly>
  struct CallScope {
    explicit CallScope(Call) { }
  };

  template<>
  struct CallScope<true> {
    explicit CallScope(Call id) : m_id(id), m_on(id < Call::SubFirst ? active() : nestedActive()) {
      if (m_on)
        m_t0 = stamp();
    }

    ~CallScope() {
      if (m_on)
        addCallSample(m_id, stamp() - m_t0);
    }

    CallScope             (const CallScope&) = delete;
    CallScope& operator = (const CallScope&) = delete;

  private:
    Call    m_id;
    bool    m_on;
    int64_t m_t0 = 0;
  };

}

#define BLESSED_PROBE_CALL(ContextType, id) \
  ::dxvk::blessed::CallScope<std::is_same_v<ContextType, ::dxvk::D3D11ImmediateContext>> blessedProbe_##id(::dxvk::blessed::Call::id)

#else // !BLESSED_PROBE_ENABLED

// blessed: blessed_probe=false -- the same names direct call sites use
// (Map/Unmap/gi/scene_capture instantiate CallScope<true> and Call::*
// directly, not just through the macro), kept trivial so no call site needs
// an #ifdef of its own. No timers, no globals, no .cpp: this is the entire
// probe when the option is off.
#include "util_blessed_fe_census.h" // blessed: threaded-fe

namespace dxvk::blessed {

  enum class Call : uint32_t {
    Map = 0, Unmap, UpdateSubresource, UpdateSubresource1, SetConstantBuffers,
    SetShaderResources, SetSamplers, SetShader, IASetVertexBuffers,
    IASetIndexBuffer, IASetInputLayout, IASetPrimitiveTopology,
    OMSetRenderTargets, OMSetBlendState, OMSetDepthStencilState, RSSetState,
    RSSetViewports, RSSetScissorRects, Draw, DrawIndexed, DrawInstanced,
    DrawIndexedInstanced, Dispatch, CopyResource, CopySubresourceRegion,
    ClearRenderTargetView, ClearDepthStencilView,
    SubFirst,
    DrawIndexedCascadeCheck = SubFirst, DrawIndexedSceneCapture, DrawIndexedGi,
    DrawIndexedCore, DrawIndexedPostHook, DrawIndexedSceneCaptureResolve,
    DrawIndexedGiPatch, DrawIndexedSceneStatic, DrawIndexedSceneStaticEmit,
    DrawIndexedSceneSkinned, DrawIndexedSceneSkinnedStage,
    DrawIndexedGiResolve, DrawIndexedGiCbRead, DrawIndexedGiSample,
    DrawIndexedGiWrite, DrawIndexedSceneSkinnedAcquire,
    DrawIndexedSceneSkinnedGrow, DrawIndexedSceneSkinnedSlices,
    DrawIndexedSceneSkinnedPush, DrawIndexedGiSampleCell, DrawIndexedGiEmit,
    Count
  };

  // blessed: cb-ring -- Ring: WRITE_DISCARD served by the constant-buffer ring
  enum class MapType : uint32_t { Discard = 0, NoOverwrite, Other, Ring, Count };

  // blessed: cb-ring -- events of the ring: a block switch, a new block
  // allocated (nested in the switch), a map that fell back upstream, and a
  // rename that opened a new cs command instead of joining the previous one
  enum class CbRingEvent : uint32_t { Advance = 0, NewBlock, Fallback, RenameCmd, Count };
  enum class BindKind : uint32_t { Constant = 0, Vertex, Index, Other, Count };

  inline BindKind classifyBindFlags(uint32_t) { return BindKind::Other; }

  // constexpr, not just inline: guarantees every `if (active())`-style
  // guard at a call site (e.g. D3D11ImmediateContext::MapBuffer's
  // classification branch) folds to dead code at any optimization level,
  // not just after the optimizer chooses to see through a global load.
  constexpr bool enabled()      { return false; }
  constexpr bool active()       { return false; }
  constexpr bool nestedActive() { return false; }

  struct MapScope {
    MapScope(MapType, BindKind) { }
    MapScope             (const MapScope&) = delete;
    MapScope& operator = (const MapScope&) = delete;
    void setType(MapType) { }
  };

  struct CbRingScope {
    explicit CbRingScope(CbRingEvent) { }
    CbRingScope             (const CbRingScope&) = delete;
    CbRingScope& operator = (const CbRingScope&) = delete;
  };

  struct DiscardSliceScope {
    DiscardSliceScope() { }
    DiscardSliceScope             (const DiscardSliceScope&) = delete;
    DiscardSliceScope& operator = (const DiscardSliceScope&) = delete;
  };

  struct EmitBlockScope {
    EmitBlockScope() { }
    EmitBlockScope             (const EmitBlockScope&) = delete;
    EmitBlockScope& operator = (const EmitBlockScope&) = delete;
  };

  struct CsSnapshot {
    uint64_t idleTicks     = 0;
    uint64_t chunkCount    = 0;
    uint64_t chunkCmdCount = 0;
    uint64_t syncCount     = 0;
    uint64_t syncTicks     = 0;
  };

  inline void onPresent(const CsSnapshot&) { }

  // blessed: threaded-fe -- see above
  struct FeSnapshot {
    uint32_t mode = 0; uint64_t records = 0, bytes = 0, publishes = 0,
      releases = 0, idleUs = 0, wakes = 0, folded = 0, packetOps = 0;
  };

  inline void feInit() { }
  inline void addFeCallSample(FeCall, int64_t, bool) { }
  inline void addFeDrainSample(FeDrain, int64_t) { }
  inline void addFeRingFullSample(int64_t) { }
  inline void addFeReplaySample(int64_t) { }
  inline void addFePresentWaitSample(int64_t) { }
  inline void addFeOccupancy(uint64_t) { }
  inline void setFeSnapshot(const FeSnapshot&) { }
  inline int64_t stamp() { return 0; }
  inline int64_t g_feNestedTicks = 0;
  inline uint64_t g_feDrainCount = 0;

  struct FeScope {
    explicit FeScope(FeCall) { }
    FeScope(FeCall, bool) { }
    FeScope             (const FeScope&) = delete;
    FeScope& operator = (const FeScope&) = delete;
  };

  template<bool ImmediateOnly>
  struct CallScope {
    explicit CallScope(Call) { }
  };

}

#define BLESSED_PROBE_CALL(ContextType, id) do { } while (0)

#endif // BLESSED_PROBE_ENABLED
