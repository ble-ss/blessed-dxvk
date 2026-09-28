// blessed: ray-traced gi probe grid -- dxvk-side trace dispatch + cpu-visible readback
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "blessed_rt.h"

#include "../dxvk_buffer.h"
#include "../dxvk_device.h"

#include "../../util/sync/sync_spinlock.h"

namespace dxvk {

  class DxvkContext;
  class DxvkCommandList;
  class DxvkImage;
  class DxvkImageView;
  class DxvkSampler;
  class DxvkPipelineLayout;

  /**
   * \brief One frame's pre-zero lighting reading
   *
   * Captured by \c BlessedGi::PatchOnCs (d3d11 side, cs thread since
   * gi-cs) from the *same* ps b2 cbuffer the ambient patch overwrites --
   * c0/c1 (DirLightDirection, DirLightColor) and c11-c13
   * (DirectionalAmbient, read before this draw's patch lands) all live in
   * one pass-115 draw's cbuffer (gi-route.md section 2). Captured once per
   * frame (the first patched draw), handed straight to
   * \ref BlessedGiState::noteFrameLighting, and consumed once by the next
   * \ref BlessedGiState::runTrace.
   */
  struct BlessedGiFrameLighting {
    float sunDir[3]      = { 0.0f, 0.0f, 1.0f };
    float sunColor[3]    = { 1.0f, 1.0f, 1.0f };
    // row-major float3x4, row c = (a_x, a_y, a_z, a_const) -- vanilla's
    // DirectionalAmbient, captured before the ambient-kill patch runs.
    float ambientRow[12] = { };
    bool  valid          = false;
  };

  /**
   * \brief Gi v1: one mesh's diffuse texture + geometry identity, app -> cs thread
   *
   * Built by \c BlessedGi::OnDraw (d3d11 side) when the main lit pass draws
   * an indexed mesh with a bound ps t0 SRV, and only when that mesh's bound
   * texture differs from the one an app-thread-local dedup cache last saw
   * for it (see the .cpp) -- so this crosses to the cs thread once per
   * distinct mesh/texture pairing, not once per draw. \c vb/ib carry the
   * same byte-offset-baked-in convention as \ref BlessedSceneDraw, so
   * \c DxvkContext::blessedGiNoteMeshAlbedo can build the identical
   * BlessedScene::CacheKey addDraw would for the same mesh.
   */
  struct BlessedGiMeshAlbedoNote {
    DxvkBufferSlice vb;
    uint32_t        vbStride   = 0;
    VkFormat        vbFormat   = VK_FORMAT_UNDEFINED;
    DxvkBufferSlice ib;
    VkIndexType     indexType  = VK_INDEX_TYPE_UINT16;
    uint32_t        indexCount = 0;
    uint32_t        startIndex = 0;
    int32_t         baseVertex = 0;

    Rc<DxvkImageView> textureView;
    // blessed: false when the SRV's own format is already srgb-typed (the
    // hardware sampler linearizes it on read); true for a unorm-typed BC
    // diffuse texture, which this seat assumes is srgb-authored (Skyrim's
    // own convention) and decodes manually -- see blessed_gi_albedo.comp.
    bool              needsSrgbDecode = true;
  };

  /**
   * \brief gi-bounds: one draw's mesh, as the cs thread sees it bound
   *
   * The same fields BlessedScene::CacheKey and noteMeshAlbedo key a mesh
   * by: \c vbOffset already includes the position attribute's own offset,
   * \c ibOffset is the index binding's offset. Built by
   * \c BlessedGi::PatchOnCsBounds (d3d11 side) from the bound vb/ib and the
   * draw's own index range.
   */
  struct BlessedGiBoundsRequest {
    DxvkBuffer*  vb         = nullptr;
    VkDeviceSize vbOffset   = 0;
    uint32_t     vbStride   = 0;
    VkFormat     vbFormat   = VK_FORMAT_UNDEFINED;
    DxvkBuffer*  ib         = nullptr;
    VkDeviceSize ibOffset   = 0;
    VkIndexType  indexType  = VK_INDEX_TYPE_UINT16;
    uint32_t     indexCount = 0;
    uint32_t     startIndex = 0;
    int32_t      baseVertex = 0;
  };

  /**
   * \brief Traces a camera-scrolled probe grid each frame and serves cpu-side samples
   *
   * Owned by \c DxvkDevice next to \c BlessedRt/BlessedScene, created only
   * when \ref isEnabled() is true (BLESSED_GI=probes) and the device
   * supports ray query. Mirrors \c BlessedShadowObjects' shape but keeps
   * the "traced state" object itself device-owned (not a function-local
   * static) since \ref sampleAmbient must be reachable from the d3d11 app
   * thread via \c DxvkDevice::blessedGi().
   *
   * Threading (gi-cs): \ref noteFrameLighting, \ref runTrace,
   * \ref refreshCsCache and \ref sampleAmbient all run on the cs thread
   * only -- the lighting note and the samples from the gi ambient patch
   * that runs just before each patched draw is recorded
   * (BlessedGi::PatchOnCs, d3d11 side), the trace and the cache refresh
   * from the ordered end-of-frame chunk (see \c D3D11SwapChain::Present).
   * So none of them need a lock between each other. \c m_published is
   * still guarded by \ref m_publishLock; only the small log getters read
   * state from other threads.
   */
  class BlessedGiState {

  public:

    explicit BlessedGiState(DxvkDevice* device);
    ~BlessedGiState();

    BlessedGiState             (const BlessedGiState&) = delete;
    BlessedGiState& operator = (const BlessedGiState&) = delete;

    // Cheap, cached: true once BLESSED_GI=probes.
    static bool isEnabled();

    // cs thread: cheap value copy, no gpu work. See BlessedGiFrameLighting.
    void noteFrameLighting(const BlessedGiFrameLighting& lighting);

    /**
     * \brief Traces every probe for one frame and publishes a cpu-visible copy
     *
     * Called from \c DxvkContext::blessedRunGiTrace, itself invoked right
     * after \c blessedSceneEndFrame from the same ordered cs chunk (so
     * \p ctx's scene tlas for this frame is already current -- see the .cpp).
     * No-op (just logs once) if no lighting has been noted yet or the
     * scene has no tlas.
     */
    void runTrace(
            DxvkContext*         ctx,
      const Rc<DxvkCommandList>& cmd);

    /**
     * \brief Trilinearly samples the most recently published probe grid
     *
     * Cs thread only (gi-cs), against \ref m_csCache.
     *
     * \p absWorldPos is a true absolute world position (camera-relative
     * value from the draw's own vs b2 World translation, plus that draw's
     * own current camera position -- see blessed_gi.cpp's OnDraw, Probes branch).
     * Writes the row-major float3x4 DirectionalAmbient form (12 floats,
     * already in the shape \c BlessedGi::OnDraw can memcpy straight into
     * the cbuffer) to \p outRow. Returns \c false before the first trace
     * has published anything.
     */
    bool sampleAmbient(const float absWorldPos[3], float outRow[12]) const;

    // blessed: gi-bounds -- BLESSED_GI_SAMPLE=bounds (default origin)
    bool sampleBounds() const { return m_sampleBounds; }

    /**
     * \brief gi-bounds: a mesh's object-space aabb, or null while unknown
     *
     * Cs thread only (from \c BlessedGi::PatchOnCsBounds). Returns six
     * floats (min xyz, max xyz) once the bounds pass has read them back. On
     * the first miss it queues the mesh for a later \ref runTrace's bounds
     * pass (a few frames late is expected); a mesh whose buffers are
     * dynamic, out of range or come back degenerate stays null for good,
     * and the caller keeps the origin sample. Records no gpu work.
     * \p pSettled, if given, is set true when the answer can no longer
     * change (known or failed), false while the mesh is queued or in flight.
     */
    const float* lookupMeshBounds(const BlessedGiBoundsRequest& req, bool* pSettled = nullptr);

    /**
     * \brief gi-bounds: samples the grid over a draw's bounds instead of its origin
     *
     * Cs thread only. \p world3x4 is the draw's row-major camera-relative
     * World, \p camPosAbs the camera, \p bmin / \p bmax the object-space
     * aabb. Samples the centre and the eight corners (pulled toward the
     * centre by BLESSED_GI_BOUNDS_PULL), each trilinearly with every
     * corner probe's weight scaled by its openness, and averages the rows
     * of the points that had valid probes, each weighted by its own
     * openness. Returns \c false when no point had a valid probe.
     */
    bool sampleAmbientBounds(const float world3x4[12], const float camPosAbs[3],
      const float bmin[3], const float bmax[3], float outRow[12]) const;

    /**
     * \brief Gi v1: once-per-frame copy of the ready probe slot, cs thread
     *
     * gi-cs: called from \c DxvkContext::blessedRunGiTrace right after
     * \ref runTrace (the ordered end-of-frame chunk, whether or not the
     * trace ran), so *before* the next frame's draws reach the cs thread
     * and call \ref sampleAmbient. Does the one bulk memcpy out of the
     * (possibly write-combined, now HOST_CACHED) gpu-mapped ring slot into
     * \ref m_csCache, a plain cs-thread-owned vector -- so every later
     * \ref sampleAmbient call this frame is a handful of reads against
     * ordinary process memory, no lock, no gpu-mapped indirection. Lead's
     * fix (campaign c3: per-draw sampling of the mapped ring cost 4.1 ms),
     * moved from the app thread's present to the cs thread's frame boundary.
     */
    void refreshCsCache();

    /**
     * \brief gi-cs: bumped by every \ref refreshCsCache (cs thread)
     *
     * Lets cs-thread callers key per-frame memos (the sample memo, the
     * once-per-frame lighting capture) off the cache they were built from.
     */
    uint64_t csCacheGeneration() const { return m_csCacheGeneration; }

    // blessed: gi-bounds -- the grid window origin (absolute cell) of the
    // trace the cs cache holds; changes when the grid scrolls. Cs thread.
    const int32_t* csCacheOriginCell() const { return m_csCacheOrigin; }

    // BLESSED_GI_DEBUG=irradiance -- see blessed_gi.cpp's WritePatchRow callers.
    bool debugIrradiance() const { return m_debugIrradiance; }

    // blessed: gi v0/v1 a/b switch -- BLESSED_GI_V0=1 keeps v0's constant
    // albedo, -ray-direction hit normal, no backface invalidation, and no
    // sky-bounce term, while still running through the mode=probes pipeline.
    bool v0Mode() const { return m_v0Mode; }

    /**
     * \brief Gi v1: resolves (or queues) the albedo id for one texture
     *
     * Cs-thread only (called from DxvkContext::blessedGiNoteMeshAlbedo). A
     * cache hit (same DxvkImage already seen) returns immediately; a miss
     * allocates the next id, queues the view for \ref runTrace to actually
     * sample (never records gpu work here -- this may run mid the game's
     * own render pass, see the .cpp), and returns that id right away so the
     * mesh's instance can carry it this frame. The albedo buffer slot for a
     * newly allocated id already holds BLESSED_GI_ALBEDO (the whole buffer
     * is fill-cleared to that constant on creation), so a mesh sampled
     * before its texture's own dispatch has run just gets the same constant
     * v0 always used -- never black, never garbage.
     */
    uint32_t ensureAlbedo(const Rc<DxvkImageView>& view, bool needsSrgbDecode);

    // blessed: gi.jsonl fields the d3d11 side can't see on its own -- see
    // BlessedGi::OnPresent (src/d3d11/blessed_gi.cpp), which merges these
    // into the same line as its own patched-draws-per-frame count so the
    // seat only ever writes one gi.jsonl.
    uint32_t probesActive() const {
      return uint32_t(m_config.dimX) * uint32_t(m_config.dimY) * uint32_t(m_config.dimZ);
    }
    uint32_t raysPerProbe() const { return m_config.rays; }
    double   lastTraceGpuMs() const {
      std::lock_guard<sync::Spinlock> lock(m_publishLock);
      return m_lastTraceGpuMs;
    }

    // blessed: gi v1 -- albedo textures resolved so far (cs-thread value,
    // read from the app thread's OnPresent same as the rest of this block;
    // a single relaxed-ish read of a monotonically-growing counter is fine
    // for a 120-present log line).
    uint32_t albedoTexturesActive() const { return m_nextAlbedoId - 1u; }

    // blessed: gi-cs -- counted on the cs thread by refreshCsCache every
    // InvalidCountInterval refreshes, read from the app thread's log line.
    uint32_t countInvalidProbes() const {
      return m_invalidProbes.load(std::memory_order_relaxed);
    }

    // blessed: gi-bounds -- gi.jsonl fields, same threading as the above
    uint32_t meshesWithBounds() const {
      return m_meshesWithBounds.load(std::memory_order_relaxed);
    }

    float meanOpenness() const {
      return m_meanOpenness.load(std::memory_order_relaxed);
    }

  private:

    struct GridConfig {
      int32_t dimX = 32, dimY = 32, dimZ = 8;
      float   spacing = 256.0f;
      uint32_t rays = 16;
      float   albedo = 0.35f;
      float   sky = 1.0f;
      float   alpha = 0.05f;
      float   skyBounce = 0.5f;  // BLESSED_GI_SKYBOUNCE
      float   backface  = 0.25f; // BLESSED_GI_BACKFACE
    };

    // blessed: gi v1 -- SH coefficients (12 floats, unchanged layout/meaning)
    // plus one validity float per probe (see the class doc + runTrace):
    // >0.5 valid, temporally blended like everything else in this buffer so
    // a probe's validity doesn't flicker frame to frame.
    static constexpr uint32_t ProbeStride = 13;

    // blessed: gi-bounds -- the stride actually used: ProbeStride, or one
    // more (openness at [13]) under BLESSED_GI_SAMPLE=bounds
    uint32_t m_probeStride = ProbeStride;

    DxvkDevice* m_device;
    GridConfig  m_config;
    bool        m_debugIrradiance = false;
    bool        m_v0Mode = false; // BLESSED_GI_V0=1
    bool        m_flipWinding = false; // BLESSED_GI_WINDING=flip

    // blessed: the persistent, device-local SH accumulator -- read-modify-
    // write in place by the trace shader every frame, one invocation per
    // probe slot (no cross-invocation access, so no in-shader barriers).
    Rc<DxvkBuffer> m_probeBuffer;

    // blessed: 3-frame host-visible ring, copied into right after the
    // trace dispatch each frame -- see runTrace's comment on why a fence-
    // free "N-2 frames old" read is an accepted approximation here.
    static constexpr uint32_t RingSize = 3;
    Rc<DxvkBuffer> m_ring[RingSize];
    uint64_t       m_traceCount = 0;
    int32_t        m_ringOrigin[RingSize][3] = { }; // blessed: gi-bounds, per-slot window origin

    // blessed: cs-thread-only, no lock needed between these two (see the
    // class comment) -- consumed and cleared by the next runTrace.
    BlessedGiFrameLighting m_pendingLighting;

    // blessed: the app-thread-visible "where to read" state -- small and
    // cheap to copy under the lock; the float data behind ringPtr is read
    // outside it (once a frame, by refreshCsCache -- see the class doc).
    struct Published {
      const float* ringPtr    = nullptr;
      int32_t      dims[3]    = { 32, 32, 8 };
      float        spacing    = 256.0f;
      bool         valid      = false;
      int32_t      origin[3]  = { 0, 0, 0 }; // blessed: gi-bounds, the slot's trace window origin
    };

    mutable sync::Spinlock m_publishLock;
    Published              m_published;
    double                 m_lastTraceGpuMs = 0.0; // guarded by m_publishLock

    // blessed: gi v1 -- cs-thread-only since gi-cs (written by
    // refreshCsCache, read by sampleAmbient; both run on the cs thread
    // only, so no lock needed between them -- see refreshCsCache's doc
    // comment). ProbeStride floats per probe, same layout as m_ring.
    std::vector<float> m_csCache;
    int32_t             m_csCacheDims[3] = { 32, 32, 8 };
    float               m_csCacheSpacing = 256.0f;
    bool                m_csCacheValid   = false;
    uint64_t            m_csCacheGeneration = 0; // blessed: gi-cs
    int32_t             m_csCacheOrigin[3] = { 0, 0, 0 }; // blessed: gi-bounds

    // blessed: gi-cs -- the gi.jsonl probes_invalid field, recounted on the
    // cs thread every InvalidCountInterval cache refreshes (the log line is
    // written every 120 presents) and read from the app thread.
    static constexpr uint32_t InvalidCountInterval = 120;
    uint32_t              m_refreshesSinceCount = InvalidCountInterval;
    std::atomic<uint32_t> m_invalidProbes = { 0u };
    uint32_t countInvalidProbesNow() const;

    // blessed: gi-bounds -- mean openness over the whole grid, recounted
    // with the invalid count (0 when the stride carries no openness)
    std::atomic<float>    m_meanOpenness = { 0.0f };
    float meanOpennessNow() const;

    // blessed: hook-cpu-2 -- cs-thread-only (gi-cs) cell cache in front of
    // sampleAmbient: per grid cell (c0), the eight corner probes' offsets
    // into m_csCache and which of them are valid (sh[12] > 0.5), so
    // repeated samples in one cell skip the wrap/index math and the eight
    // validity loads. Direct-mapped; emptied by refreshCsCache, the only
    // place m_csCache changes, so an entry always matches the cache.
    struct CsCell {
      int32_t  key[3]    = { 0, 0, 0 };
      bool     used      = false;
      uint8_t  validMask = 0;
      uint32_t offset[8] = { };
    };

    static constexpr uint32_t CsCellCount = 64;
    mutable CsCell m_csCells[CsCellCount];

    const CsCell& lookupCsCell(const int32_t c0[3]) const;

    // blessed: gi v1 -- per-texture albedo table. m_albedoIndex/m_nextAlbedoId
    // and m_pendingAlbedoTextures are cs-thread-only (touched only from
    // ensureAlbedo, itself only ever called from the cs-thread lambda
    // DxvkContext::blessedGiNoteMeshAlbedo queues); the buffer/pipeline
    // objects are built lazily the first time runTrace has a pending texture
    // to sample. Id 0 is reserved: an instance with albedoId 0 (never
    // assigned by ensureAlbedo, which starts at 1) uses BLESSED_GI_ALBEDO --
    // see blessed_gi_trace.comp.
    static constexpr uint32_t MaxAlbedoTextures = 4096;

    std::unordered_map<DxvkImage*, uint32_t> m_albedoIndex;
    uint32_t                                 m_nextAlbedoId = 1;

    struct PendingAlbedoTexture {
      Rc<DxvkImageView> view;
      uint32_t          id;
      bool              needsSrgbDecode;
    };
    std::vector<PendingAlbedoTexture> m_pendingAlbedoTextures;

    Rc<DxvkBuffer>            m_albedoBuffer;
    Rc<DxvkSampler>           m_albedoSampler;
    const DxvkPipelineLayout* m_albedoLayout   = nullptr;
    VkPipeline                m_albedoPipeline = VK_NULL_HANDLE;

    // blessed: gi.jsonl gpu-ms accounting -- same timestamp ring-pair idiom
    // as BlessedShadowObjects, just one pair (one dispatch per frame, not
    // per pixel-draw). The d3d11 side owns gi.jsonl itself (\ref lastTraceGpuMs).
    VkQueryPool m_queryPool = VK_NULL_HANDLE;

    void ensureBuffers(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd);
    void collectTiming();

    // blessed: gi v1 -- lazily creates the albedo buffer/sampler/pipeline
    // (first texture only) and clear-fills the buffer to BLESSED_GI_ALBEDO.
    void ensureAlbedoObjects(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd);

    // blessed: gi v1 -- drains m_pendingAlbedoTextures, one dispatch per new
    // texture (see runTrace: called before the probe trace dispatch, always
    // outside any render pass by then). Cheap in the steady state: empty
    // most frames, since ensureAlbedo only queues on a genuinely new texture.
    void drainPendingAlbedoTextures(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd);

    // blessed: gi-bounds -- see lookupMeshBounds. Everything below is
    // cs-thread-only, touched from PatchOnCsBounds (lookup), runTrace
    // (dispatch) and refreshCsCache (poll), all on the cs thread.
    bool  m_sampleBounds = false; // BLESSED_GI_SAMPLE=bounds
    float m_openness     = 1.0f;  // BLESSED_GI_OPENNESS, 0..1
    float m_boundsPull   = 0.1f;  // BLESSED_GI_BOUNDS_PULL, 0..1

    // blessed: the same fields as BlessedScene::CacheKey, with the
    // buffers' cookies in place of their pointers: a freed buffer's
    // address can come back as a different mesh, a cookie never does, so
    // no entry needs to pin its buffers.
    struct BoundsKey {
      uint64_t    vbCookie   = 0;
      uint64_t    ibCookie   = 0;
      uint32_t    vbOffset   = 0;
      uint32_t    vbStride   = 0;
      VkFormat    vbFormat   = VK_FORMAT_UNDEFINED;
      uint32_t    ibOffset   = 0;
      VkIndexType indexType  = VK_INDEX_TYPE_UINT16;
      uint32_t    indexCount = 0;
      uint32_t    startIndex = 0;
      int32_t     baseVertex = 0;

      bool operator == (const BoundsKey& o) const {
        return vbCookie == o.vbCookie && ibCookie == o.ibCookie && vbOffset == o.vbOffset
            && vbStride == o.vbStride && vbFormat == o.vbFormat && ibOffset == o.ibOffset
            && indexType == o.indexType && indexCount == o.indexCount
            && startIndex == o.startIndex && baseVertex == o.baseVertex;
      }
    };

    struct BoundsKeyHash {
      size_t operator () (const BoundsKey& k) const;
    };

    enum class BoundsState : uint8_t { Queued, InFlight, Known, Failed };

    struct BoundsEntry {
      float       bounds[6]    = { };  // min xyz, max xyz
      BoundsState state        = BoundsState::Queued;
      uint32_t    slot         = 0;
      uint32_t    serial       = 0;
      uint64_t    dispatchedAt = 0; // m_traceCount after the dispatch
      uint64_t    lastUsed     = 0; // m_csCacheGeneration of the last lookup
    };

    struct BoundsJob {
      BoundsKey      key;
      Rc<DxvkBuffer> vb;
      Rc<DxvkBuffer> ib;
      VkDeviceSize   vbOffset    = 0;
      VkDeviceSize   ibOffset    = 0;
      uint32_t       vertexCount = 0;
    };

    static constexpr uint32_t MaxBoundsSlots       = 4096;
    static constexpr uint32_t MaxBoundsQueued      = 4096;
    static constexpr uint32_t MaxBoundsPerFrame    = 128;
    static constexpr uint32_t BoundsReadbackFrames = 3;
    static constexpr uint32_t BoundsGiveUpFrames   = 60;
    static constexpr uint32_t BoundsEvictInterval  = 600;
    static constexpr uint64_t BoundsEvictAfter     = 3600;

    std::unordered_map<BoundsKey, BoundsEntry, BoundsKeyHash> m_bounds;
    std::vector<BoundsJob>    m_boundsJobs;     // queued, not yet dispatched
    std::vector<BoundsKey>    m_boundsInFlight; // dispatched, awaiting readback
    std::vector<uint32_t>     m_boundsFreeSlots;
    uint32_t                  m_boundsSerial = 0;
    uint32_t                  m_refreshesSinceEvict = 0;
    Rc<DxvkBuffer>            m_boundsResults;  // host-visible, MaxBoundsSlots x 8 floats
    const DxvkPipelineLayout* m_boundsLayout   = nullptr;
    VkPipeline                m_boundsPipeline = VK_NULL_HANDLE;
    std::atomic<uint32_t>     m_meshesWithBounds = { 0u };

    void dispatchBoundsJobs(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd);
    void pollBounds();

  };

}
