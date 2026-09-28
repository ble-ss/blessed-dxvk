// blessed: turns the game's own static draws into blas/tlas for the next tracer stage
#pragma once

#include <atomic>
#include <fstream>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "blessed_rt.h"
#include "blessed_skin_debug.h" // blessed: skin-repair
#include "blessed_gi_instance.h" // blessed: gi v1 -- per-instance geometry/albedo record

#include "../dxvk_buffer.h"
#include "../dxvk_device.h"

#include "../../util/sync/sync_spinlock.h"

namespace dxvk {

  class DxvkContext;
  class DxvkCommandList;

  /**
   * \brief One captured draw, handed from the d3d11 app thread to the cs thread
   *
   * \c vb and \c ib already carry the byte offset of the position stream /
   * index buffer start baked in as the slice offset (see
   * \c DxvkBufferSlice::getSliceInfo, which resolves the buffer's current
   * storage lazily -- exactly what lets this cross the app/cs boundary
   * before a possible map-discard rename on the buffer has been applied).
   */
  struct BlessedSceneDraw {
    DxvkBufferSlice vb;
    uint32_t        vbStride    = 0;
    VkFormat        vbFormat    = VK_FORMAT_UNDEFINED;
    uint32_t        vertexCount = 0;

    DxvkBufferSlice ib;
    VkIndexType     indexType   = VK_INDEX_TYPE_UINT16;
    uint32_t        indexCount  = 0;
    uint32_t        startIndex  = 0;
    int32_t         baseVertex  = 0;

    // object-to-world, row-major 3x4 (3 rows of float4: xyz + translation w),
    // as BLESSED_SCENE_XFORM finds it in the bound vs cbuffer.
    float transform[12] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
    };

    bool  hasCamPos = false;
    float camPos[3] = { 0.0f, 0.0f, 0.0f };
  };

  /**
   * \brief One captured skinned (actor/creature) draw -- see fork-actor-skinning.md
   *
   * blessed: hook-cpu-2 -- no longer handed over one EmitCs per draw.
   * The app thread stages these in a \ref BlessedSceneSkinBatch, one per
   * frame, and the whole batch crosses to the cs thread with that frame's
   * endFrame. The bones (the bound b10 cbuffer, 3,840 bytes, 80 bones x
   * row-major float3x4) are not copied: \c bonesAllocation is the discard
   * allocation the draw was bound to, which a later discard never writes
   * again, and holding it keeps it from being reused. The skin dispatch
   * reads the bones from its device address; the cpu only reads them
   * (through \c bones, the allocation's persistent map) for the dedupe
   * compare and BLESSED_SCENE_SKIN_TRACE.
   */
  struct BlessedSceneSkinnedDraw {
    // position stream: posVb's own DxvkBufferSlice offset already carries
    // the attribute's byte offset within the vertex baked in (see
    // D3D11CommonContext::BlessedSceneCaptureSkinnedDraw), so the cs/shader
    // side only ever needs vertex*posStride on top of it -- same convention
    // BlessedSceneDraw::vb uses for POSITION0.
    DxvkBufferSlice posVb;
    uint32_t        posStride = 0;
    VkFormat        posFormat = VK_FORMAT_UNDEFINED;

    // BLENDINDICES0 stream, always R8G8B8A8_UNORM -- may share posVb's
    // underlying buffer/binding or be a wholly separate one (read per draw,
    // per the research doc: "position may sit in a separate vertex buffer
    // slot"); either way its own slice offset already carries its attribute
    // offset the same way posVb's does.
    DxvkBufferSlice idxVb;
    uint32_t        idxStride = 0;

    // BLENDWEIGHT0 stream, always R16G16B16A16_SFLOAT
    DxvkBufferSlice wtVb;
    uint32_t        wtStride = 0;

    uint32_t        vertexCount = 0;

    DxvkBufferSlice ib;
    VkIndexType     indexType  = VK_INDEX_TYPE_UINT16;
    uint32_t        indexCount = 0;
    uint32_t        startIndex = 0;
    int32_t         baseVertex = 0;

    // 80 bones x 3 rows x float4, row-major float3x4 per bone -- see
    // Skinned.hlsli's BonesBuffer (b10) in docs/research/no-b2-draws.md.
    // blessed: hook-cpu-2 -- bonesOffset is the byte offset of the bound
    // range in bonesAllocation (the constant offset); \c bones is null until
    // endFrame points it at mapPtr + bonesOffset.
    static constexpr uint32_t BonesFloats = 240u * 4u;
    Rc<DxvkResourceAllocation> bonesAllocation;
    uint32_t                   bonesOffset = 0;
    const float*               bones       = nullptr;

    // blessed: skin-repair -- the pivot the vanilla vertex shader subtracts
    // from the blended bone translation: its OWN bound b12 byte 640
    // (BLESSED_SCENE_SKIN_PIVOT, default vs:12:640). Not BLESSED_SCENE_CAMPOS,
    // which tags the tlas camera and may point at the pixel shader's b12.
    float pivot[3] = { 0.0f, 0.0f, 0.0f };

    // BLESSED_SCENE_SKIN_TRACE only: the BLESSED_SCENE_CAMPOS value (if set)
    // and the bones cbuffer's D3D11_USAGE, logged next to the pivot.
    float    tagCam[3]  = { 0.0f, 0.0f, 0.0f };
    bool     hasTagCam  = false;
    uint32_t bonesUsage = 0;

    // blessed: skin-v2 -- BlessedSceneCapture::CurrentPass() at draw time:
    // the depth-only pass this draw belongs to (see endFrame's skinnedPass)
    uint32_t pass = 0;
  };

  /**
   * \brief blessed: hook-cpu-2 -- one frame's skinned draws, staged on the app thread
   *
   * Filled by \c BlessedSceneCapture::StageSkinnedDraw between two
   * presents, moved whole into the endFrame cs command, and recycled
   * through \ref BlessedScene::acquireSkinBatch / \ref BlessedScene::recycleSkinBatch
   * so the vector's memory is reused frame to frame.
   */
  struct BlessedSceneSkinBatch {
    std::vector<BlessedSceneSkinnedDraw> draws;
    uint32_t                             lateDraws = 0; // after the mask pass: counted, never staged

    void clear() {
      draws.clear();
      lateDraws = 0;
    }
  };

  /// blessed: {tlas address, camera pos, instance count} for the most recently completed build
  struct BlessedSceneFrame {
    VkDeviceAddress tlasAddress   = 0;
    uint32_t        instanceCount = 0;
    float           camPos[3]     = { 0.0f, 0.0f, 0.0f };
    bool            valid         = false;

    // blessed: gi v1 -- this frame's per-instance geometry/albedo table
    // (BlessedGiInstanceInfo, one entry per tlas instance, same order the
    // tlas build used -- see BlessedScene::endFrame). Zero when no static
    // instance carried geometry info this frame (instanceInfoCount == 0);
    // the gi trace shader falls back to -ray-direction shading in that case.
    VkDeviceAddress instanceInfoAddress = 0;
    uint32_t        instanceInfoCount   = 0;
    Rc<DxvkBuffer>  instanceInfoBuffer; // keeps instanceInfoAddress valid -- track this alongside tlas

    // blessed: the tlas object itself, not just its raw address -- a
    // consumer (e.g. DxvkContext::blessedRunShadowPass) must track this Rc
    // against whichever command list actually reads tlasAddress via ray
    // query, so the tlas (and every blas it references, see
    // BlessedAccelStruct::setReferencedBlases) survives until that GPU
    // dispatch finishes, independent of this scene's own cache eviction.
    Rc<BlessedAccelStruct> tlas;
  };

  /// blessed: counters for BLESSED_SCENE_LOG, updated on the cs thread and
  /// read (relaxed) from the present thread every 120 presents -- see
  /// BlessedSceneCapture::OnPresent in src/d3d11/blessed_scene_capture.cpp.
  struct BlessedSceneStats {
    std::atomic<uint32_t> blasCount        { 0 };
    std::atomic<uint64_t> blasBytes        { 0 };
    std::atomic<uint32_t> buildsThisWindow { 0 };
    std::atomic<uint32_t> tlasInstances    { 0 };
    std::atomic<uint32_t> blasDroppedCap   { 0 };

    // blessed: actor-skinning
    std::atomic<uint32_t> skinnedBlasCount  { 0 }; // live cache entries (built or refit)
    std::atomic<uint32_t> skinnedRefits     { 0 }; // UPDATE-mode builds this window
    std::atomic<uint32_t> skinnedDroppedCap { 0 }; // BLESSED_SCENE_MAX_SKINNED exceeded

    // blessed: skin-repair
    std::atomic<uint32_t> skinnedShared     { 0 }; // draws that shared a mesh with an earlier one this frame (occurrence > 0)
    std::atomic<uint32_t> skinnedDuplicates { 0 }; // same mesh + same pose again this frame, skipped
    std::atomic<uint64_t> skinnedGpuNs      { 0 }; // skin dispatch + blas build/refit, summed over samples
    std::atomic<uint32_t> skinnedGpuSamples { 0 };

    // blessed: skin-v2
    std::atomic<uint32_t> skinnedOtherPass  { 0 }; // skinned draws dropped: not from the pass before the mask draw
    std::atomic<uint32_t> skinnedNoMask     { 0 }; // frames with skinned draws but no mask draw (largest pass used)
    std::atomic<uint32_t> skinnedRebuilds   { 0 }; // full builds of an existing entry (refit constraint changed)

    // blessed: hook-cpu-2
    std::atomic<uint32_t> skinnedBonesUnaddressable { 0 }; // b10 allocation without a device address or 16-byte alignment, dropped

    // blessed: gi v1 -- distinct meshes noteMeshAlbedo has ever resolved a
    // texture for (mirrors m_meshAlbedo.size(), read cross-thread by
    // BlessedGi::OnPresent for gi.jsonl's meshes_with_albedo -- m_meshAlbedo
    // itself is cs-thread-only, an unordered_map with no lock, so this
    // atomic exists purely to make that count safe to read from the app
    // thread, same reasoning as every other field in this struct).
    std::atomic<uint32_t> giMeshesWithAlbedo { 0 };
  };

  /**
   * \brief Blas cache + per-frame tlas builder for scene capture
   *
   * Owned by \ref DxvkDevice next to \ref BlessedRt, created only when
   * \c BlessedScene::isEnabled() is true (a selector env var is set) and
   * the device supports ray query. Every entry point below is cheap to
   * call when disabled: the device simply never creates one, and the one
   * call site outside this file (\c DxvkContext::blessedSceneAddDraw /
   * \c blessedSceneEndFrame) is a single cached-pointer null check.
   */
  class BlessedScene {

  public:

    explicit BlessedScene(DxvkDevice* device);
    ~BlessedScene();

    BlessedScene             (const BlessedScene&) = delete;
    BlessedScene& operator = (const BlessedScene&) = delete;

    // Cheap, cached: true once a selector env var is set. Read once by
    // DxvkDevice's constructor to decide whether to create this at all.
    static bool isEnabled();

    static uint32_t maxBlasPerFrame();

    // blessed: actor-skinning -- BLESSED_SCENE_MAX_SKINNED, default 128 (skin-v2).
    // Caps how many distinct skinned draws get a compute dispatch + blas
    // build/refit per frame; drops beyond it are counted in
    // BlessedSceneStats::skinnedDroppedCap.
    static uint32_t maxSkinnedPerFrame();

    /**
     * \brief Looks up/creates the blas cache entry for one draw and appends an instance
     *
     * Runs on the cs thread. A cache hit marks the entry used this frame
     * (see \ref endFrame's eviction) and just appends an instance; a miss
     * queues a blas build, up to \ref maxBlasPerFrame per frame -- drops
     * beyond that are counted in \ref BlessedSceneStats::blasDroppedCap.
     */
    void addDraw(
            DxvkContext*         ctx,
      const Rc<DxvkCommandList>& cmd,
      const BlessedSceneDraw&    draw);

    // blessed: hook-cpu-2 -- BLESSED_SCENE_SKIN_PASS=all (every depth-only
    // pass feeds the tlas). Read by the app thread too: without it, skinned
    // draws after the mask pass can never be chosen and are not copied.
    static bool skinAllPasses();

    // blessed: hook-cpu-2 -- any thread. A cleared batch, recycled when one
    // is free (keeps the bones arena's capacity), else a new one.
    static std::unique_ptr<BlessedSceneSkinBatch> acquireSkinBatch();

    // blessed: hook-cpu-2 -- any thread. Clears \p batch and keeps it for reuse.
    static void recycleSkinBatch(std::unique_ptr<BlessedSceneSkinBatch> batch);

    /**
     * \brief Builds this frame's queued blases and tlas
     *
     * Runs on the cs thread, emitted from \c D3D11SwapChain::Present
     * before the present flush. Builds every blas queued by \ref addDraw
     * this frame, then uploads the instance list and builds the tlas into
     * whichever of the two ping-pong slots isn't the one \ref currentFrame
     * is currently handing out. \c m_tlasWriteSlot (which slot that is) is
     * only ever mutated in here, so it only ever changes on the cs thread.
     *
     * blessed: skin-v2 -- \p skinnedPass is the depth-only pass right before
     * the frame's mask draw (0 = no mask draw seen). Only the staged skinned
     * draws of that pass are skinned and refit; without a mask draw, the
     * pass with the most skinned draws is used. All of them go through one
     * compute dispatch and one vkCmdBuildAccelerationStructuresKHR call.
     *
     * blessed: hook-cpu-2 -- \p skinBatch holds the frame's skinned draws
     * (null: none staged); endFrame borrows its storage and leaves it cleared.
     */
    void endFrame(
            DxvkContext*           ctx,
      const Rc<DxvkCommandList>&   cmd,
            uint32_t               skinnedPass,
            BlessedSceneSkinBatch* skinBatch);

    /**
     * \brief The last fully-built frame's tlas, camera position, and instance count
     *
     * Cs-thread-only: callers must be running inside a cs-thread command
     * (e.g. \c DxvkContext::blessedRunShadowPass, at dispatch time), never
     * eagerly on the d3d11 app thread at draw-record time. The spinlock
     * makes every individual read/write here safe regardless of caller
     * thread, but calling this early on the app thread and carrying the
     * result across to the cs thread (as \c BlessedShadow::OnDraw used to)
     * can still name a ping-pong slot \ref endFrame rebuilds out from under
     * it before the cs thread's turn to use it comes up -- that raced the
     * sun mask into flickering frame to frame in a static scene.
     */
    BlessedSceneFrame currentFrame() const;

    /**
     * \brief Tags the frame being recorded with the camera it was drawn from
     *
     * cs thread only. Called by the shadow pass with the mask pass's own
     * CameraPosAdjust (ps b12 byte 640), so the tlas built at this frame's
     * end and the next frame's shadow pass compare the same quantity (the
     * prepass's b12 does not hold it). Overrides any camera from addDraw.
     */
    void noteCamera(const float camPos[3]);

    /**
     * \brief Gi v1: records the albedo id a mesh's diffuse texture resolved to
     *
     * Runs on the cs thread (called from DxvkContext::blessedGiNoteMeshAlbedo,
     * itself reached via the app thread's EmitCs, same pattern as addDraw).
     * Builds the identical CacheKey addDraw would for the same mesh (same
     * vb/ib/ranges) and remembers albedoId against it, so a later addDraw
     * this frame or next frame's addDraw for the same mesh looks it up when
     * building that instance's \ref BlessedGiInstanceInfo. Independent of
     * cache eviction: a stale mesh's albedo entry just goes unread.
     */
    void noteMeshAlbedo(
            DxvkBuffer*  vb,
            uint32_t     vbOffset,
            uint32_t     vbStride,
            VkFormat     vbFormat,
            DxvkBuffer*  ib,
            uint32_t     ibOffset,
            VkIndexType  indexType,
            uint32_t     indexCount,
            uint32_t     startIndex,
            int32_t      baseVertex,
            uint32_t     albedoId);

    BlessedSceneStats& stats() { return m_stats; }

  private:

    struct CacheKey {
      DxvkBuffer* vb         = nullptr;
      uint32_t    vbOffset   = 0;
      uint32_t    vbStride   = 0;
      VkFormat    vbFormat   = VK_FORMAT_UNDEFINED;
      DxvkBuffer* ib         = nullptr;
      uint32_t    ibOffset   = 0;
      VkIndexType indexType  = VK_INDEX_TYPE_UINT16;
      uint32_t    indexCount = 0;
      uint32_t    startIndex = 0;
      int32_t     baseVertex = 0;

      bool operator == (const CacheKey& o) const {
        return vb == o.vb && vbOffset == o.vbOffset && vbStride == o.vbStride
            && vbFormat == o.vbFormat && ib == o.ib && ibOffset == o.ibOffset
            && indexType == o.indexType && indexCount == o.indexCount
            && startIndex == o.startIndex && baseVertex == o.baseVertex;
      }
    };

    struct CacheKeyHash {
      size_t operator () (const CacheKey& k) const {
        size_t h = std::hash<const void*>()(k.vb);
        auto mix = [&h] (size_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
        mix(k.vbOffset);
        mix(k.vbStride);
        mix(size_t(k.vbFormat));
        mix(std::hash<const void*>()(k.ib));
        mix(k.ibOffset);
        mix(size_t(k.indexType));
        mix(k.indexCount);
        mix(k.startIndex);
        mix(size_t(k.baseVertex));
        return h;
      }
    };

    struct CacheEntry {
      Rc<BlessedAccelStruct> blas;
      Rc<DxvkBuffer>      vb;   // pinned: outlives the blas built from it
      Rc<DxvkBuffer>      ib;   // pinned; null for non-indexed (unused today, all draws are indexed)
      uint32_t            vertexCount = 0; // vb size / stride from the vb's own binding, the blas build's maxVertex
      uint64_t            lastUsedFrame = 0;
      bool                queuedThisFrame = false;
    };

    struct PendingInstance {
      CacheKey key;
      float    transform[12];
      uint32_t instanceId;
      uint32_t albedoId = 0; // blessed: gi v1 -- see BlessedScene::noteMeshAlbedo
    };

    // blessed: actor-skinning -- keyed the same way as CacheKey (vb/ib/ranges,
    // per fork-actor-skinning.md's lifetime rule), but vb here is the
    // position stream specifically; idxVb/wtVb are pinned separately below
    // since they may be a different buffer entirely.
    // blessed: skin-v2 -- everything a refit (MODE_UPDATE) must keep equal
    // to the build it updates (VUIDs 03758-03767: geometry count and flags
    // are constant here; vertex format, maxVertex, index type, index data,
    // primitive count, primitive offset and first vertex are checked).
    // The index data is compared by device address: a renamed or re-bound
    // index buffer is treated as new index data.
    struct SkinnedBuildShape {
      uint32_t        maxVertex       = 0;
      VkIndexType     indexType       = VK_INDEX_TYPE_UINT16;
      VkDeviceAddress indexAddress    = 0;
      uint32_t        primitiveCount  = 0;
      uint32_t        primitiveOffset = 0;
      uint32_t        firstVertex     = 0;

      bool operator == (const SkinnedBuildShape& o) const {
        return maxVertex == o.maxVertex && indexType == o.indexType
            && indexAddress == o.indexAddress && primitiveCount == o.primitiveCount
            && primitiveOffset == o.primitiveOffset && firstVertex == o.firstVertex;
      }
    };

    struct SkinnedCacheEntry {
      Rc<BlessedAccelStruct> blas;          // built with ALLOW_UPDATE; refit in place while shape is unchanged
      Rc<DxvkBuffer>         posVb, idxVb, wtVb, ib; // pinned: this frame's draw buffers (re-set every frame)
      Rc<DxvkBuffer>         outPositions;  // camera-relative float3 per vertex, compute output / blas vertex input
      uint32_t               vertexCount     = 0;
      uint64_t                lastUsedFrame  = 0;
      SkinnedBuildShape       shape;                   // blessed: skin-v2, what blas was built from
      VkDeviceSize            updateScratchSize = 0;   // blessed: skin-v2, from the build's size query
    };

    // blessed: skin-repair -- one skinned entry per draw INSTANCE within a
    // frame: two actors in the same armor share vb/ib/ranges, so the mesh
    // key alone collapsed them onto one output buffer and one blas (and
    // their dispatches raced on it). occurrence = how many distinct poses of
    // this mesh were already queued earlier in the same frame.
    struct SkinnedKey {
      CacheKey mesh;
      uint32_t occurrence = 0;

      bool operator == (const SkinnedKey& o) const {
        return mesh == o.mesh && occurrence == o.occurrence;
      }
    };

    struct SkinnedKeyHash {
      size_t operator () (const SkinnedKey& k) const {
        size_t h = CacheKeyHash()(k.mesh);
        return h ^ (size_t(k.occurrence) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
      }
    };

    struct PendingSkinnedDraw {
      SkinnedKey              key;    // mesh.vb = posVb.buffer(), matching SkinnedCacheEntry's identity
      uint32_t                staged; // blessed: skin-v2, index into m_stagedSkinned
      uint32_t                instanceId;
    };

    DxvkDevice*                                     m_device;
    BlessedRt*                                      m_rt;

    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> m_cache;
    std::vector<CacheKey>                            m_pendingBuilds;
    std::vector<PendingInstance>                     m_pendingInstances;

    // blessed: gi v1 -- mesh (CacheKey) -> albedo id, written by
    // noteMeshAlbedo (from the main lit pass's own draw hook) and read by
    // addDraw when building this frame's PendingInstance. Never evicted
    // (small: one entry per distinct mesh, same order as m_cache); a mesh
    // that stops appearing just goes unread, same as a stale m_cache entry.
    std::unordered_map<CacheKey, uint32_t, CacheKeyHash> m_meshAlbedo;

    // blessed: actor-skinning
    std::unordered_map<SkinnedKey, SkinnedCacheEntry, SkinnedKeyHash> m_skinnedCache;
    std::vector<PendingSkinnedDraw>                  m_pendingSkinned;

    // blessed: skin-v2 -- every skinned draw of the frame, from any
    // depth-only pass; endFrame picks one pass and queues only its draws
    std::vector<BlessedSceneSkinnedDraw>             m_stagedSkinned;

    // blessed: skin-repair -- poses already queued this frame, per mesh
    // (cleared in endFrame); the vector index is the occurrence.
    // blessed: hook-cpu-2 -- each pose is its bones (a pointer into the
    // draw's mapped allocation), compared with memcmp only when a mesh
    // repeats, instead of an app-thread hash over every draw's bones.
    std::unordered_map<CacheKey, std::vector<const float*>, CacheKeyHash> m_skinnedPosesThisFrame;

    // blessed: skin-repair -- lazily created, see blessed_skin_debug.h
    std::unique_ptr<BlessedSkinTrace>                m_skinTrace;
    std::unique_ptr<BlessedSkinTimer>                m_skinTimer;

    uint64_t                                         m_frameIndex = 0;
    uint32_t                                         m_nextInstanceId = 0;

    bool                                              m_pendingHasCamPos = false;
    bool                                              m_needInputBarrier = false;
    float                                             m_pendingCamPos[3] = { 0.0f, 0.0f, 0.0f };

    Rc<DxvkBuffer>                                   m_scratch;
    Rc<DxvkBuffer>                                   m_skinScratch; // blessed: skin-v2, one region per batched skinned build

    // ping-pong: the tlas a pass reads via currentFrame() is never the one
    // endFrame is about to rebuild. Each slot is an Rc: replacing it (on
    // rebuild) or dropping it (on destruction) is safe the moment nothing
    // still needs to track it -- see DxvkCommandList::track() call sites
    // in endFrame() below.
    static constexpr uint32_t NumTlasSlots = 2;
    Rc<BlessedAccelStruct>                           m_tlas[NumTlasSlots];
    uint32_t                                         m_tlasWriteSlot = 0;

    // blessed: currentFrame() is called from the present/app thread while
    // endFrame() (cs thread) may be updating this -- cheap to just spinlock
    // this handful of fields (now including one Rc<>) rather than reach
    // for atomics per field.
    mutable sync::Spinlock                           m_frameLock;
    BlessedSceneFrame                                m_publishedFrame;

    BlessedSceneStats                                m_stats;

    // blessed: BLESSED_SCENE_TRACE=1 -- logs the first 200 blas builds'
    // raw build inputs to <BLESSED_PROBE_DIR>/scene-trace.log, so a bad
    // build (out-of-bounds vb/ib read -> device lost) can be found without
    // re-deriving it from validation layer output alone.
    uint32_t                                         m_traceCount = 0;
    std::ofstream                                    m_traceFile;
    bool                                              m_traceFileTried = false;

    VkDeviceAddress ensureScratch(DxvkContext* ctx,
      const Rc<DxvkCommandList>& cmd, VkDeviceSize size);

    // blessed: skin-v2 -- same as above, for any scratch buffer slot
    VkDeviceAddress ensureScratch(DxvkContext* ctx,
      const Rc<DxvkCommandList>& cmd, VkDeviceSize size, Rc<DxvkBuffer>& scratch);

    void maybeTraceBuild(
      const CacheKey&   key,
      const CacheEntry& entry,
      VkDeviceAddress   vbAddress,
      VkDeviceAddress   ibAddress,
      uint32_t          maxVertex,
      uint32_t          primitiveCount);

    // blessed: skin-v2 -- picks the pass (see endFrame) and queues its draws
    void selectSkinnedDraws(DxvkContext* ctx, uint32_t skinnedPass);

    // blessed: skin-v2 -- the old addSkinnedDraw body: dedup, cap, pin, queue
    void queueSkinnedDraw(DxvkContext* ctx, uint32_t stagedIndex);

    /**
     * \brief Actor-skinning: dispatches compute skinning + builds/refits blases for this frame's queue
     *
     * Called from \ref endFrame, before static instances are resolved.
     * Self-contained: records its own transfer->compute and compute->AS-build
     * barriers (newly-seen buffers only need the former once). Appends one
     * instance (identity transform -- the compute output is already
     * camera-relative) per successfully built/refit entry to \p instances /
     * \p referencedBlases, and counts builds into \p buildsThisFrame so
     * endFrame's blas-to-tlas barrier covers these too.
     */
    void processSkinnedDraws(
            DxvkContext*         ctx,
      const Rc<DxvkCommandList>& cmd,
      std::vector<VkAccelerationStructureInstanceKHR>& instances,
      std::vector<Rc<BlessedAccelStruct>>&             referencedBlases,
      std::vector<BlessedGiInstanceInfo>&              instanceInfos,
      uint32_t&                                        buildsThisFrame);

  };

}
