// blessed: scene capture -- see blessed_scene.h
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "blessed_scene.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"

#include "../../util/util_env.h"
#include "../../util/util_math.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"

#include <blessed_skin.h>

namespace dxvk {

  namespace {

    // blessed: BLESSED_SCENE_MAX_BLAS_PER_FRAME=<n>, default 64
    uint32_t ParseMaxBlasPerFrame() {
      std::string s = env::getEnvVar("BLESSED_SCENE_MAX_BLAS_PER_FRAME");

      if (s.empty())
        return 64u;

      uint32_t n = std::strtoul(s.c_str(), nullptr, 10);
      return n ? n : 64u;
    }

    // blessed: actor-skinning -- BLESSED_SCENE_MAX_SKINNED=<n>, default 128
    // (skin-v2: was 32, raised now that all draws share one dispatch and
    // one build call)
    uint32_t ParseMaxSkinnedPerFrame() {
      std::string s = env::getEnvVar("BLESSED_SCENE_MAX_SKINNED");

      if (s.empty())
        return 128u;

      uint32_t n = std::strtoul(s.c_str(), nullptr, 10);
      return n ? n : 128u;
    }

    // blessed: entries unused for this many frames are evicted from the blas cache.
    // Safe purely because dropping a cache entry's Rc<BlessedAccelStruct> no
    // longer destroys anything still in use: any tlas built from it already
    // holds its own Rc via BlessedAccelStruct::setReferencedBlases, and any
    // command list still recording against it holds one via cmd->track().
    constexpr uint64_t EvictAfterFrames = 120;

    // blessed: VK_FORMAT_R32G32B32A32_SFLOAT is not a valid AS build vertex
    // format (VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexFormat-03797).
    // D3D11CommonContext::BlessedSceneCaptureDraw accepts it as a position
    // stream format because plenty of vertex layouts pad positions to a
    // float4, so map it to its 3-component twin here; the build only reads
    // xyz and vertexStride (the real per-vertex byte stride) already skips
    // the w lane correctly. R16G16B16A16_SFLOAT is a valid AS format as-is.
    VkFormat ToAccelStructVertexFormat(VkFormat fmt) {
      switch (fmt) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        default: return fmt;
      }
    }

    // blessed: gi v1 -- CacheKey::vbFormat is always one of these two after
    // ToAccelStructVertexFormat (BlessedSceneCaptureDraw only ever accepts
    // R32G32B32(A32)_SFLOAT or R16G16B16A16_SFLOAT as a position stream, and
    // the former's A32 variant is already collapsed above); see
    // BlessedGiVertexFormat in blessed_gi_instance.h.
    uint32_t GiVertexFormatCode(VkFormat fmt) {
      return fmt == VK_FORMAT_R16G16B16A16_SFLOAT
        ? uint32_t(BlessedGiVertexFormat::Rgba16f)
        : uint32_t(BlessedGiVertexFormat::Rgb32f);
    }

    // blessed: BLESSED_SCENE_TRACE=1, see BlessedScene::maybeTraceBuild
    bool TraceEnabled() {
      static bool s_enabled = env::getEnvVar("BLESSED_SCENE_TRACE") == "1";
      return s_enabled;
    }

    constexpr uint32_t MaxTracedBuilds = 200;

    // blessed: actor-skinning -- position formats blessed_skin.comp understands
    // (see readPosition there); -1 for anything else (caller skips the draw).
    int32_t SkinPosFormatCode(VkFormat fmt) {
      switch (fmt) {
        case VK_FORMAT_R32G32B32_SFLOAT:      return 0;
        case VK_FORMAT_R32G32B32A32_SFLOAT:   return 1;
        case VK_FORMAT_R16G16B16A16_SFLOAT:   return 2;
        default:                              return -1;
      }
    }

    // blessed: skin-v2 -- one per skinned draw of the frame. Must match
    // SkinDraw in shaders/blessed_skin.comp exactly (scalar layout).
    struct BlessedSkinDrawRecord {
      uint64_t posBuf;
      uint64_t idxBuf;
      uint64_t wtBuf;
      uint64_t outBuf;
      uint64_t bonesBuf;    // blessed: hook-cpu-2, the draw's own b10 range (device address)
      float    pivot[3];
      uint32_t posStride;
      uint32_t posFormat;
      uint32_t idxStride;
      uint32_t wtStride;
      uint32_t vertexCount;
      uint32_t firstGroup;
      uint32_t reserved;    // keeps the size a multiple of 8
    };

    static_assert(sizeof(BlessedSkinDrawRecord) == 80,
      "must match the scalar layout of SkinDraw in blessed_skin.comp");

    // blessed: push data layout must match BlessedSkinPushData in
    // shaders/blessed_skin.comp exactly (scalar layout).
    struct BlessedSkinPushData {
      uint64_t table;       // BlessedSkinDrawRecord[drawCount]
      uint64_t groups;      // uint32_t[groupCount], workgroup -> record index
      uint32_t groupCount;
      uint32_t reserved;
    };

    static_assert(sizeof(BlessedSkinPushData) <= 128,
      "must fit the Vulkan-guaranteed minimum maxPushConstantsSize");

    /**
     * \brief Lazily-built compute-skin pipeline, process lifetime -- same model as BlessedShadowObjects
     */
    class BlessedSkinObjects {
    public:

      explicit BlessedSkinObjects(DxvkDevice* device)
      : m_device(device) {
        // blessed: no descriptors at all -- every buffer this shader touches
        // is addressed via GL_EXT_buffer_reference2 push-constant pointers,
        // the same convention as blessed_rt_selftest.comp.
        m_layout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlags(),
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedSkinPushData), 0u, nullptr);

        util::DxvkBuiltInShaderStage shader(blessed_skin, nullptr);
        m_pipeline = device->createBuiltInComputePipeline(m_layout, shader);
      }

      // blessed: skin-v2 -- one dispatch for every skinned draw of the frame
      void dispatch(
        const Rc<DxvkCommandList>&  cmd,
        const BlessedSkinPushData&  push) {
        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_layout,
          0u, nullptr, sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, push.groupCount, 1u, 1u);
      }

    private:

      DxvkDevice*               m_device;
      const DxvkPipelineLayout* m_layout   = nullptr;
      VkPipeline                m_pipeline = VK_NULL_HANDLE;
    };

    BlessedSkinObjects* SkinInstance(DxvkDevice* device) {
      static BlessedSkinObjects s_instance(device);
      return &s_instance;
    }

  }


  bool BlessedScene::isEnabled() {
    static bool s_enabled = [] {
      return !env::getEnvVar("BLESSED_SCENE_VS").empty()
          || env::getEnvVar("BLESSED_SCENE_RULE") == "depthonly";
    }();

    return s_enabled;
  }


  uint32_t BlessedScene::maxBlasPerFrame() {
    static uint32_t s_max = ParseMaxBlasPerFrame();
    return s_max;
  }


  uint32_t BlessedScene::maxSkinnedPerFrame() {
    static uint32_t s_max = ParseMaxSkinnedPerFrame();
    return s_max;
  }


  BlessedScene::BlessedScene(DxvkDevice* device)
  : m_device(device), m_rt(device->blessedRt()) {

  }


  BlessedScene::~BlessedScene() {

  }


  VkDeviceAddress BlessedScene::ensureScratch(DxvkContext* ctx,
    const Rc<DxvkCommandList>& cmd, VkDeviceSize size) {
    return ensureScratch(ctx, cmd, size, m_scratch);
  }


  VkDeviceAddress BlessedScene::ensureScratch(DxvkContext* ctx,
    const Rc<DxvkCommandList>& cmd, VkDeviceSize size, Rc<DxvkBuffer>& scratch) {
    VkDeviceSize alignment = m_device->properties().khrAccelerationStructure
      .minAccelerationStructureScratchOffsetAlignment;
    VkDeviceSize alignedSize = align(size, alignment);

    if (scratch == nullptr || scratch->info().size < alignedSize) {
      // blessed: dropping the old m_scratch Rc here is safe: whichever
      // command list(s) actually recorded builds against its address
      // already tracked it (see the cmd->track() call below, which every
      // caller of ensureScratch goes through), so it outlives this frame
      // if the GPU is still catching up.
      DxvkBufferCreateInfo bufferInfo = { };
      bufferInfo.size    = alignedSize;
      bufferInfo.usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      bufferInfo.stages  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      bufferInfo.access  = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      bufferInfo.debugName = "blessed scene scratch";

      scratch = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      ctx->ensureBufferAddress(scratch);
    }

    // blessed: keep the buffer alive until the GPU finishes whatever build
    // the caller is about to record against the address returned below.
    cmd->track(scratch);

    return scratch->getSliceInfo().gpuAddress;
  }


  void BlessedScene::addDraw(
          DxvkContext*         ctx,
    const Rc<DxvkCommandList>& cmd,
    const BlessedSceneDraw&    draw) {
    CacheKey key;
    key.vb         = draw.vb.buffer().ptr();
    key.vbOffset   = uint32_t(draw.vb.offset());
    key.vbStride   = draw.vbStride;
    key.vbFormat   = ToAccelStructVertexFormat(draw.vbFormat);
    key.ib         = draw.ib.buffer().ptr();
    key.ibOffset   = uint32_t(draw.ib.offset());
    key.indexType  = draw.indexType;
    key.indexCount = draw.indexCount;
    key.startIndex = draw.startIndex;
    key.baseVertex = draw.baseVertex;

    auto it = m_cache.find(key);

    if (it != m_cache.end()) {
      it->second.lastUsedFrame = m_frameIndex;
    } else {
      if (m_pendingBuilds.size() >= maxBlasPerFrame()) {
        m_stats.blasDroppedCap.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      CacheEntry entry;
      entry.vb             = draw.vb.buffer();
      entry.ib             = draw.ib.buffer();
      entry.vertexCount    = draw.vertexCount;
      entry.lastUsedFrame  = m_frameIndex;
      entry.queuedThisFrame = true;

      // blessed: the build reads these by raw device address, which
      // bypasses dxvk's usual per-draw resource tracking -- pin them so
      // they can't relocate, and keep the Rc<> in the cache entry so the
      // buffer itself can't die under a blas built from it.
      ctx->ensureBufferAddress(entry.vb);
      ctx->ensureBufferAddress(entry.ib);

      // blessed: newly seen buffers need a transfer-write -> build-read
      // barrier before their first blas build. recorded once in endFrame,
      // outside the game's render pass; addDraw records no commands, so it
      // never has to split the game's pass per draw.
      m_needInputBarrier = true;

      it = m_cache.emplace(key, std::move(entry)).first;
      m_pendingBuilds.push_back(key);
    }

    PendingInstance inst;
    inst.key = key;
    std::memcpy(inst.transform, draw.transform, sizeof(inst.transform));
    inst.instanceId = m_nextInstanceId++;

    // blessed: gi v1 -- 0 (unknown/default, the trace shader's BLESSED_GI_ALBEDO
    // fallback) unless the main lit pass's own draw hook already resolved
    // this mesh's diffuse texture this frame (or a previous one) -- see
    // noteMeshAlbedo.
    auto albedoIt = m_meshAlbedo.find(key);
    inst.albedoId = albedoIt != m_meshAlbedo.end() ? albedoIt->second : 0u;

    m_pendingInstances.push_back(inst);

    // blessed: same camera every draw this frame -- last write wins, which
    // is fine since it should never actually differ within one frame.
    if (draw.hasCamPos) {
      m_pendingHasCamPos = true;
      std::memcpy(m_pendingCamPos, draw.camPos, sizeof(m_pendingCamPos));
    }
  }


  void BlessedScene::noteMeshAlbedo(
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
          uint32_t     albedoId) {
    // blessed: gi v1 -- identical construction to addDraw's own CacheKey, so
    // the same mesh drawn depth-only (scene capture) and lit (gi's own hook)
    // hashes to the same entry.
    CacheKey key;
    key.vb         = vb;
    key.vbOffset   = vbOffset;
    key.vbStride   = vbStride;
    key.vbFormat   = ToAccelStructVertexFormat(vbFormat);
    key.ib         = ib;
    key.ibOffset   = ibOffset;
    key.indexType  = indexType;
    key.indexCount = indexCount;
    key.startIndex = startIndex;
    key.baseVertex = baseVertex;

    auto result = m_meshAlbedo.emplace(key, albedoId);
    if (result.second) {
      m_stats.giMeshesWithAlbedo.fetch_add(1, std::memory_order_relaxed);
    } else {
      result.first->second = albedoId;
    }
  }


  // blessed: hook-cpu-2 -- free skin batches. Two or three are live at any
  // time (one filling on the app thread, one or two queued for the cs
  // thread), so a small locked vector is plenty; touched twice per frame.
  // Kept up to 8, so a cs thread running several frames behind for a while
  // does not leave the app thread allocating fresh batches afterwards.
  namespace {
    sync::Spinlock                                      g_skinBatchLock;
    std::vector<std::unique_ptr<BlessedSceneSkinBatch>> g_skinBatchFree;
  }


  bool BlessedScene::skinAllPasses() {
    // blessed: BLESSED_SCENE_SKIN_PASS=all -- every depth-only pass, the
    // pre-skin-v2 behaviour, kept for a/b runs
    static const bool s_allPasses = env::getEnvVar("BLESSED_SCENE_SKIN_PASS") == "all";
    return s_allPasses;
  }


  std::unique_ptr<BlessedSceneSkinBatch> BlessedScene::acquireSkinBatch() {
    { std::lock_guard<sync::Spinlock> lock(g_skinBatchLock);

      if (!g_skinBatchFree.empty()) {
        std::unique_ptr<BlessedSceneSkinBatch> batch = std::move(g_skinBatchFree.back());
        g_skinBatchFree.pop_back();
        return batch;
      }
    }

    return std::make_unique<BlessedSceneSkinBatch>();
  }


  void BlessedScene::recycleSkinBatch(std::unique_ptr<BlessedSceneSkinBatch> batch) {
    if (!batch)
      return;

    batch->clear();

    std::lock_guard<sync::Spinlock> lock(g_skinBatchLock);

    if (g_skinBatchFree.size() < 8u)
      g_skinBatchFree.push_back(std::move(batch));
  }


  void BlessedScene::selectSkinnedDraws(DxvkContext* ctx, uint32_t skinnedPass) {
    if (m_stagedSkinned.empty())
      return;

    const bool s_allPasses = skinAllPasses();

    uint32_t chosen = 0;

    if (!s_allPasses) {
      // the pass with the most skinned draws, among passes at or before the
      // mask pass when one was seen. whiterun (skin-v2-1): the main prepass
      // holds the npcs (70 skinned draws), and a 4-draw depth-only pass sits
      // right before the mask; "the last pass" picked the small one.
      if (!skinnedPass)
        m_stats.skinnedNoMask.fetch_add(1, std::memory_order_relaxed);

      std::vector<std::pair<uint32_t, uint32_t>> counts;
      for (const BlessedSceneSkinnedDraw& draw : m_stagedSkinned) {
        if (skinnedPass && draw.pass > skinnedPass)
          continue;

        auto it = std::find_if(counts.begin(), counts.end(),
          [&draw] (const std::pair<uint32_t, uint32_t>& c) { return c.first == draw.pass; });

        if (it != counts.end())
          it->second++;
        else
          counts.push_back({ draw.pass, 1u });
      }

      uint32_t best = 0;
      for (const auto& c : counts) {
        if (c.second > best) {
          best   = c.second;
          chosen = c.first;
        }
      }
    }

    for (uint32_t i = 0; i < uint32_t(m_stagedSkinned.size()); i++) {
      if (s_allPasses || m_stagedSkinned[i].pass == chosen)
        queueSkinnedDraw(ctx, i);
      else
        m_stats.skinnedOtherPass.fetch_add(1, std::memory_order_relaxed);
    }
  }


  void BlessedScene::queueSkinnedDraw(DxvkContext* ctx, uint32_t stagedIndex) {
    const BlessedSceneSkinnedDraw& draw = m_stagedSkinned[stagedIndex];

    // blessed: keyed like the static cache (vb/ib/ranges) so an actor's
    // mesh gets one persistent entry refit frame to frame, independent of
    // that actor's changing bone pose -- see fork-actor-skinning.md's
    // lifetime rule.
    CacheKey key;
    key.vb         = draw.posVb.buffer().ptr();
    key.vbOffset   = uint32_t(draw.posVb.offset());
    key.vbStride   = draw.posStride;
    key.vbFormat   = draw.posFormat;
    key.ib         = draw.ib.buffer().ptr();
    key.ibOffset   = uint32_t(draw.ib.offset());
    key.indexType  = draw.indexType;
    key.indexCount = draw.indexCount;
    key.startIndex = draw.startIndex;
    key.baseVertex = draw.baseVertex;

    // blessed: skin-repair -- the same mesh key can be several actors (two
    // guards in one armor) or the same actor drawn again by another
    // depth-only pass. same pose again: nothing new, skip it. new pose: a
    // new occurrence, so it gets its own output buffer and its own blas.
    // blessed: hook-cpu-2 -- same test as the old bones hash, done exactly:
    // a byte compare against this mesh's earlier poses, so only a repeated
    // mesh pays anything (the old hash ran over every draw's bones).
    std::vector<const float*>& poses = m_skinnedPosesThisFrame[key];

    for (const float* pose : poses) {
      if (!std::memcmp(pose, draw.bones, sizeof(float) * BlessedSceneSkinnedDraw::BonesFloats)) {
        m_stats.skinnedDuplicates.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }

    if (m_pendingSkinned.size() >= maxSkinnedPerFrame()) {
      m_stats.skinnedDroppedCap.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    uint32_t occurrence = uint32_t(poses.size());
    poses.push_back(draw.bones);

    if (occurrence)
      m_stats.skinnedShared.fetch_add(1, std::memory_order_relaxed);

    // blessed: pin every buffer whose address will go into the compute
    // dispatch or the blas build -- same reasoning as the static path's
    // ensureBufferAddress calls in addDraw above.
    ctx->ensureBufferAddress(draw.posVb.buffer());
    ctx->ensureBufferAddress(draw.idxVb.buffer());
    ctx->ensureBufferAddress(draw.wtVb.buffer());
    ctx->ensureBufferAddress(draw.ib.buffer());

    PendingSkinnedDraw pending;
    pending.key.mesh       = key;
    pending.key.occurrence = occurrence;
    pending.staged         = stagedIndex;
    pending.instanceId     = m_nextInstanceId++;
    m_pendingSkinned.push_back(std::move(pending));
  }


  void BlessedScene::noteCamera(const float camPos[3]) {
    m_pendingHasCamPos = true;
    std::memcpy(m_pendingCamPos, camPos, sizeof(m_pendingCamPos));
  }


  void BlessedScene::endFrame(
          DxvkContext*           ctx,
    const Rc<DxvkCommandList>&   cmd,
          uint32_t               skinnedPass,
          BlessedSceneSkinBatch* skinBatch) {
    uint32_t buildsThisFrame = 0;

    // blessed: hook-cpu-2 -- take over the frame's staged skinned draws
    // (swap, so both vectors keep their capacity) and point each at its
    // bones through the allocation's persistent map. Each draw holds its
    // allocation, so the pointers hold for all of endFrame; nothing reads
    // through them unless a mesh repeats (dedupe) or the skin trace is on.
    if (skinBatch) {
      m_stagedSkinned.swap(skinBatch->draws);

      for (BlessedSceneSkinnedDraw& draw : m_stagedSkinned) {
        draw.bones = reinterpret_cast<const float*>(
          reinterpret_cast<const uint8_t*>(draw.bonesAllocation->mapPtr()) + draw.bonesOffset);
      }

      // skinned draws after the mask pass, dropped on the app thread without
      // staging: selectSkinnedDraws would have dropped them here
      if (skinBatch->lateDraws)
        m_stats.skinnedOtherPass.fetch_add(skinBatch->lateDraws, std::memory_order_relaxed);
    }

    // blessed: skin-v2 -- queue only the chosen pass's skinned draws
    selectSkinnedDraws(ctx, skinnedPass);

    // blessed: actor-skinning -- declared up here (not at their old spot
    // in step 2 below) so processSkinnedDraws (run after the static build
    // loop, before step 2) can append its own instances into the same
    // vectors that step 2 then adds the static ones to.
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    std::vector<Rc<BlessedAccelStruct>> referencedBlases;
    // blessed: gi v1 -- one entry per instance, same push order as
    // `instances` (processSkinnedDraws pushes a zeroed sentinel per skinned
    // instance; the static loop below pushes real geometry/albedo). Indexed
    // at trace time by rayQueryGetIntersectionInstanceIdEXT, the hardware
    // instance-array index -- not instanceCustomIndex.
    std::vector<BlessedGiInstanceInfo> instanceInfos;
    instances.reserve(m_pendingInstances.size() + m_pendingSkinned.size());
    referencedBlases.reserve(m_pendingInstances.size() + m_pendingSkinned.size());
    instanceInfos.reserve(m_pendingInstances.size() + m_pendingSkinned.size());

    // 0. newly seen game buffers: their writes (upload/transfer) must be
    // visible to the builds below. one barrier for all of them.
    if (m_needInputBarrier && !m_pendingBuilds.empty()) {
      VkMemoryBarrier2 toAsBuild = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      toAsBuild.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      toAsBuild.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      toAsBuild.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      toAsBuild.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

      VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      dep.memoryBarrierCount = 1u;
      dep.pMemoryBarriers    = &toAsBuild;

      // blessed: raw -- manual barrier, outside any render pass (endFrame's
      // caller ends the current pass first)
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
    }
    m_needInputBarrier = false;

    // 1. build every blas queued by this frame's addDraw() calls
    for (const CacheKey& key : m_pendingBuilds) {
      auto it = m_cache.find(key);
      if (it == m_cache.end())
        continue;

      VkAccelerationStructureGeometryTrianglesDataKHR triangles =
        { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
      triangles.vertexFormat             = key.vbFormat;
      triangles.vertexData.deviceAddress = it->second.vb->getSliceInfo().gpuAddress + key.vbOffset;
      triangles.vertexStride             = key.vbStride;
      triangles.maxVertex                = it->second.vertexCount > 0 ? it->second.vertexCount - 1u : 0u;
      triangles.indexType                = key.indexType;
      triangles.indexData.deviceAddress  = it->second.ib->getSliceInfo().gpuAddress + key.ibOffset;

      VkAccelerationStructureGeometryKHR geometry =
        { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
      geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.geometry.triangles = triangles;
      geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;

      VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
        { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
      buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
      buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.geometryCount = 1u;
      buildInfo.pGeometries   = &geometry;

      uint32_t primitiveCount = key.indexCount / 3u;

      VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
        { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
      // blessed: raw
      m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

      it->second.blas = m_rt->createAccelStruct(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo.accelerationStructureSize);

      buildInfo.dstAccelerationStructure  = it->second.blas->handle();
      buildInfo.scratchData.deviceAddress = ensureScratch(ctx, cmd, sizeInfo.buildScratchSize);

      VkAccelerationStructureBuildRangeInfoKHR rangeInfo = { };
      rangeInfo.primitiveCount = primitiveCount;
      rangeInfo.primitiveOffset = key.startIndex * (key.indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u);
      rangeInfo.firstVertex    = uint32_t(key.baseVertex);
      const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

      blessedBuildBarrier(cmd.ptr());
      cmd->cmdBuildAccelerationStructures(DxvkCmdBuffer::ExecBuffer, 1u, &buildInfo, &pRangeInfo);

      // blessed: keep this blas alive until this build finishes on the GPU.
      // The cache entry's own Rc (above) covers instance reuse across
      // frames; the tlas built below takes its own Rc too, via
      // setReferencedBlases, for as long as the tlas itself survives.
      cmd->track(it->second.blas);

      // blessed: BLESSED_SCENE_TRACE=1 -- see maybeTraceBuild
      maybeTraceBuild(key, it->second, triangles.vertexData.deviceAddress,
        triangles.indexData.deviceAddress, triangles.maxVertex, primitiveCount);

      it->second.queuedThisFrame = false;
      buildsThisFrame++;
    }

    // 1b. actor-skinning: compute-skin + build/refit this frame's queued
    // skinned draws, appending their instances into the vectors declared
    // above -- see processSkinnedDraws.
    processSkinnedDraws(ctx, cmd, instances, referencedBlases, instanceInfos, buildsThisFrame);

    if (buildsThisFrame > 0) {
      // blessed: blas write -> tlas read (instance references) plus ww on
      // scratch if the tlas build below reuses the buffer BlessedRt uses.
      VkMemoryBarrier2 blasToTlas = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      blasToTlas.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      blasToTlas.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      blasToTlas.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      blasToTlas.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
                               | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

      VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      dep.memoryBarrierCount = 1u;
      dep.pMemoryBarriers    = &blasToTlas;

      // blessed: raw
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
    }

    // 2. resolve this frame's static instances (now every blas above has a
    // real address) -- skinned instances were already appended above by
    // processSkinnedDraws.
    for (const PendingInstance& pending : m_pendingInstances) {
      auto it = m_cache.find(pending.key);
      if (it == m_cache.end() || it->second.blas == nullptr || !it->second.blas->handle())
        continue;

      VkAccelerationStructureInstanceKHR instance = { };
      // blessed: transform is row-major 3x4 (3 rows of float4), which is
      // exactly VkTransformMatrixKHR's own layout.
      std::memcpy(&instance.transform, pending.transform, sizeof(instance.transform));
      instance.instanceCustomIndex           = pending.instanceId & 0xFFFFFFu;
      instance.mask                          = 0xFFu;
      instance.flags                         = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      instance.accelerationStructureReference = it->second.blas->address();

      instances.push_back(instance);
      referencedBlases.push_back(it->second.blas);

      // blessed: gi v1 -- same index as the instance just pushed above
      // (rayQueryGetIntersectionInstanceIdEXT returns that array position).
      // ib/vb addresses already carry the CacheKey's own byte offsets, same
      // as the blas build's own triangles.vertexData/indexData above.
      BlessedGiInstanceInfo info;
      info.vbAddress            = it->second.vb->getSliceInfo().gpuAddress + pending.key.vbOffset;

      // blessed: gi v1 -- round the ib address down to a 4-byte boundary and
      // carry the remainder separately; see BlessedGiInstanceInfo::ibAddress.
      VkDeviceAddress ibRawAddress = it->second.ib->getSliceInfo().gpuAddress + pending.key.ibOffset;
      info.ibAddress            = ibRawAddress & ~VkDeviceAddress(3u);
      info.ibByteShift          = uint32_t(ibRawAddress & VkDeviceAddress(3u));
      info.vbStride             = pending.key.vbStride;
      info.vbFormatCode         = GiVertexFormatCode(pending.key.vbFormat);
      info.indexTypeCode        = pending.key.indexType == VK_INDEX_TYPE_UINT16 ? 0u : 1u;
      info.primitiveOffsetBytes = pending.key.startIndex * (pending.key.indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u);
      info.baseVertex           = pending.key.baseVertex;
      info.albedoId             = pending.albedoId;
      instanceInfos.push_back(info);

      // blessed: BLESSED_SCENE_TRACE=1 -- one frame's instance list, to
      // find oversized or misplaced meshes (firstlight: rays hit something
      // more than 2048 units away everywhere)
      if (m_traceFile.is_open() && m_frameIndex == 1500) {
        const float* t = pending.transform;
        auto rowNorm = [&] (int r) { return std::sqrt(t[4*r]*t[4*r] + t[4*r+1]*t[4*r+1] + t[4*r+2]*t[4*r+2]); };
        m_traceFile << str::format("inst[", instances.size() - 1, "] t=(", t[3], ",", t[7], ",", t[11],
          ") dist=", std::sqrt(t[3]*t[3] + t[7]*t[7] + t[11]*t[11]),
          " scale=(", rowNorm(0), ",", rowNorm(1), ",", rowNorm(2), ")",
          " verts=", it->second.vertexCount, " indices=", pending.key.indexCount,
          " vb=", pending.key.vbStride, "b fmt=", uint32_t(pending.key.vbFormat)) << std::endl;
      }
    }

    // blessed: BLESSED_SCENE_FREEZE=<frame> -- stop rebuilding the tlas after
    // that frame (diagnosis: does the tlas content cause the flicker?)
    static const uint64_t s_freezeAt = std::strtoull(env::getEnvVar("BLESSED_SCENE_FREEZE").c_str(), nullptr, 10);
    bool frozen = s_freezeAt && m_frameIndex > s_freezeAt && m_publishedFrame.valid;

    // 3. upload the instance list and build the tlas into the other ping-pong slot
    if (!instances.empty() && !frozen) {
      DxvkBufferCreateInfo instanceInfo = { };
      instanceInfo.size    = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
      instanceInfo.usage   = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      instanceInfo.stages  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      instanceInfo.access  = VK_ACCESS_2_SHADER_READ_BIT;
      instanceInfo.debugName = "blessed scene instances";

      // blessed: fresh allocation every frame; tracked below (cmd->track)
      // instead of the old fixed-depth m_retiredBuffers heuristic.
      Rc<DxvkBuffer> instanceBuffer = m_device->createBuffer(instanceInfo,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      std::memcpy(instanceBuffer->getSliceInfo().mapPtr, instances.data(), instanceInfo.size);
      ctx->ensureBufferAddress(instanceBuffer);

      // blessed: gi v1 -- instanceInfos is built 1:1 with instances above
      // (processSkinnedDraws' sentinel entries + the static loop's real
      // ones), so its device address, read by rayQueryGetIntersectionInstanceIdEXT
      // in the gi trace shader, indexes the same array position.
      Rc<DxvkBuffer> instanceInfoBuffer;
      VkDeviceAddress instanceInfoAddress = 0;

      if (!instanceInfos.empty()) {
        DxvkBufferCreateInfo giInfo = { };
        giInfo.size      = sizeof(BlessedGiInstanceInfo) * instanceInfos.size();
        giInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        giInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        giInfo.access    = VK_ACCESS_2_SHADER_READ_BIT;
        giInfo.debugName = "blessed gi instance info";

        instanceInfoBuffer = m_device->createBuffer(giInfo,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memcpy(instanceInfoBuffer->getSliceInfo().mapPtr, instanceInfos.data(), giInfo.size);
        ctx->ensureBufferAddress(instanceInfoBuffer);
        instanceInfoAddress = instanceInfoBuffer->getSliceInfo().gpuAddress;
      }

      Rc<BlessedAccelStruct> tlas = m_rt->recordBuildTlas(ctx, cmd,
        instanceBuffer->getSliceInfo().gpuAddress, uint32_t(instances.size()));

      // blessed: pin every blas this tlas references on the tlas object
      // itself -- tracking the tlas's Rc anywhere (this frame's cmd below,
      // or a later frame's shadow-pass dispatch via currentFrame()) then
      // transitively keeps these alive too, independent of cache eviction.
      // Also track each directly against this cmd: belt and suspenders for
      // this frame's own submission, cheap since they're already Rc'd.
      for (const Rc<BlessedAccelStruct>& blas : referencedBlases)
        cmd->track(blas);
      tlas->setReferencedBlases(std::move(referencedBlases));

      // blessed: tlas write -> whatever future shader stage the tracer
      // pass reads it from. ACCELERATION_STRUCTURE_READ_BIT_KHR is valid
      // at any shader stage per VK_KHR_ray_query; ALL_COMMANDS on the dst
      // side because this seat doesn't know which stage that pass runs on.
      VkMemoryBarrier2 tlasToRead = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      tlasToRead.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      tlasToRead.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      tlasToRead.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      tlasToRead.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

      VkDependencyInfo tlasDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      tlasDep.memoryBarrierCount = 1u;
      tlasDep.pMemoryBarriers    = &tlasToRead;

      // blessed: raw
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &tlasDep);

      // blessed: keep this frame's instance buffer alive until the tlas
      // build above finishes on the GPU -- replaces m_retiredBuffers.
      cmd->track(instanceBuffer);

      // blessed: gi v1 -- this one is read much later than the build (by a
      // future frame's gi trace dispatch via currentFrame()), so its Rc
      // lives in m_publishedFrame below rather than just this cmd.
      if (instanceInfoBuffer != nullptr)
        cmd->track(instanceInfoBuffer);

      m_tlasWriteSlot = (m_tlasWriteSlot + 1u) % NumTlasSlots;
      // blessed: dropping the old slot's Rc here is safe -- whichever
      // command list(s) last read it (this scene's own build, or a
      // shadow-pass dispatch) already tracked it themselves.
      m_tlas[m_tlasWriteSlot] = std::move(tlas);

      {
        std::lock_guard<sync::Spinlock> lock(m_frameLock);
        m_publishedFrame.tlasAddress   = m_tlas[m_tlasWriteSlot]->address();
        m_publishedFrame.instanceCount = uint32_t(instances.size());
        m_publishedFrame.camPos[0]     = m_pendingCamPos[0];
        m_publishedFrame.camPos[1]     = m_pendingCamPos[1];
        m_publishedFrame.camPos[2]     = m_pendingCamPos[2];
        m_publishedFrame.valid         = true;
        m_publishedFrame.tlas          = m_tlas[m_tlasWriteSlot];
        m_publishedFrame.instanceInfoAddress = instanceInfoAddress;
        m_publishedFrame.instanceInfoCount   = uint32_t(instanceInfos.size());
        m_publishedFrame.instanceInfoBuffer  = std::move(instanceInfoBuffer);
      }
    }

    // 4. evict cache entries nothing has drawn in a while. Safe: see the
    // EvictAfterFrames comment above.
    for (auto it = m_cache.begin(); it != m_cache.end(); ) {
      if (m_frameIndex - it->second.lastUsedFrame > EvictAfterFrames)
        it = m_cache.erase(it);
      else
        ++it;
    }

    // blessed: actor-skinning -- same idea. An actor that leaves the scene
    // stops calling addSkinnedDraw, so its entry's lastUsedFrame stops
    // advancing; dropping the entry's Rcs here frees its blas, output
    // buffer, and pinned vb/ib/wt buffers together.
    for (auto it = m_skinnedCache.begin(); it != m_skinnedCache.end(); ) {
      if (m_frameIndex - it->second.lastUsedFrame > EvictAfterFrames)
        it = m_skinnedCache.erase(it);
      else
        ++it;
    }

    // 5. stats for BLESSED_SCENE_LOG (see BlessedSceneStats)
    uint64_t blasBytes = 0;
    for (const auto& kv : m_cache)
      blasBytes += (kv.second.blas != nullptr && kv.second.blas->buffer() != nullptr)
        ? kv.second.blas->buffer()->info().size : 0u;

    m_stats.skinnedBlasCount.store(uint32_t(m_skinnedCache.size()), std::memory_order_relaxed);

    m_stats.blasCount.store(uint32_t(m_cache.size()), std::memory_order_relaxed);
    m_stats.blasBytes.store(blasBytes, std::memory_order_relaxed);
    m_stats.tlasInstances.store(uint32_t(instances.size()), std::memory_order_relaxed);
    m_stats.buildsThisWindow.fetch_add(buildsThisFrame, std::memory_order_relaxed);

    m_pendingBuilds.clear();
    m_pendingInstances.clear();
    m_pendingSkinned.clear();
    m_stagedSkinned.clear(); // blessed: skin-v2

    // blessed: hook-cpu-2 -- hand the (now empty) draw storage back to the batch
    if (skinBatch)
      m_stagedSkinned.swap(skinBatch->draws);

    m_skinnedPosesThisFrame.clear(); // blessed: skin-repair
    m_frameIndex++;
  }


  void BlessedScene::processSkinnedDraws(
          DxvkContext*         ctx,
    const Rc<DxvkCommandList>& cmd,
    std::vector<VkAccelerationStructureInstanceKHR>& instances,
    std::vector<Rc<BlessedAccelStruct>>&             referencedBlases,
    std::vector<BlessedGiInstanceInfo>&              instanceInfos,
    uint32_t&                                        buildsThisFrame) {
    // blessed: skin-repair -- BLESSED_SCENE_SKIN_TRACE, non-blocking; writes
    // skin-trace.log once the recorded copy has completed on the gpu
    if (unlikely(m_skinTrace != nullptr))
      m_skinTrace->poll();

    if (m_pendingSkinned.empty())
      return;

    if (!m_skinTimer) {
      m_skinTimer = std::make_unique<BlessedSkinTimer>(m_device,
        &m_stats.skinnedGpuNs, &m_stats.skinnedGpuSamples);
    }

    if (unlikely(!m_skinTrace && BlessedSkinTrace::enabled()))
      m_skinTrace = std::make_unique<BlessedSkinTrace>();

    // blessed: skin-v2 -- batched. 1. walk the queue once: create/resize each
    // entry, decide build or refit, and fill one draw record per job plus
    // the workgroup -> record map for the single dispatch below.
    struct SkinJob {
      const PendingSkinnedDraw* pending;
      SkinnedCacheEntry*        entry;   // stable: unordered_map never moves its nodes
      SkinnedBuildShape         shape;
      bool                      refit;
    };

    std::vector<SkinJob>               jobs;
    std::vector<BlessedSkinDrawRecord> records;
    std::vector<uint32_t>              groupMap;
    jobs.reserve(m_pendingSkinned.size());
    records.reserve(m_pendingSkinned.size());

    const uint32_t maxGroups = m_device->properties().core.properties.limits.maxComputeWorkGroupCount[0];

    for (const PendingSkinnedDraw& pending : m_pendingSkinned) {
      const BlessedSceneSkinnedDraw& draw = m_stagedSkinned[pending.staged];

      int32_t posFormat = SkinPosFormatCode(draw.posFormat);
      uint32_t primitiveCount = pending.key.mesh.indexCount / 3u;

      if (posFormat < 0 || draw.vertexCount == 0 || primitiveCount == 0)
        continue;

      // blessed: hook-cpu-2 -- the shader reads the bones straight from the
      // draw's b10 allocation. every dxvk buffer allocation carries
      // SHADER_DEVICE_ADDRESS (dxvk_memory.cpp's global buffer usage), and
      // a constant buffer range is 16-byte aligned (constantOffset * 16 on
      // an aligned allocation), which BonesBuf's buffer_reference_align
      // needs; anything else is dropped rather than read wrongly.
      VkDeviceAddress bonesAddress = draw.bonesAllocation->getBufferInfo().gpuAddress;

      if (!bonesAddress || ((bonesAddress + draw.bonesOffset) & 15u)) {
        m_stats.skinnedBonesUnaddressable.fetch_add(1, std::memory_order_relaxed);
        continue;
      }

      uint32_t groups = (draw.vertexCount + 63u) / 64u;

      if (groupMap.size() + groups > maxGroups) {
        m_stats.skinnedDroppedCap.fetch_add(1, std::memory_order_relaxed);
        continue;
      }

      SkinnedCacheEntry& entry = m_skinnedCache[pending.key];

      // (re)size the output buffer. a new buffer means a new blas too: the
      // old one stays alive through the tlas that references it.
      if (entry.outPositions == nullptr || entry.vertexCount != draw.vertexCount) {
        DxvkBufferCreateInfo outInfo = { };
        outInfo.size    = VkDeviceSize(draw.vertexCount) * sizeof(float) * 3u;
        outInfo.usage   = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                        | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT; // blessed: skin-repair, trace readback
        outInfo.stages  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        outInfo.access  = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        outInfo.debugName = "blessed skinned positions";

        entry.outPositions = m_device->createBuffer(outInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        ctx->ensureBufferAddress(entry.outPositions);

        entry.vertexCount = draw.vertexCount;
        entry.blas        = nullptr;
      }

      // this frame's draw buffers: pinned by the entry until the next frame
      // replaces them (and tracked on cmd below for this submission)
      entry.posVb = draw.posVb.buffer();
      entry.idxVb = draw.idxVb.buffer();
      entry.wtVb  = draw.wtVb.buffer();
      entry.ib    = draw.ib.buffer();
      entry.lastUsedFrame = m_frameIndex;

      SkinJob job;
      job.pending = &pending;
      job.entry   = &entry;
      job.shape.maxVertex       = draw.vertexCount - 1u;
      job.shape.indexType       = pending.key.mesh.indexType;
      job.shape.indexAddress    = entry.ib->getSliceInfo().gpuAddress + pending.key.mesh.ibOffset;
      job.shape.primitiveCount  = primitiveCount;
      job.shape.primitiveOffset = pending.key.mesh.startIndex
        * (pending.key.mesh.indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u);
      job.shape.firstVertex     = uint32_t(pending.key.mesh.baseVertex);

      // blessed: skin-v2 -- refit only onto a blas built from the same shape
      // (see SkinnedBuildShape); anything else gets a fresh blas and a full build
      job.refit = entry.blas != nullptr && entry.shape == job.shape;

      if (!job.refit && entry.blas != nullptr)
        m_stats.skinnedRebuilds.fetch_add(1, std::memory_order_relaxed);

      BlessedSkinDrawRecord record = { };
      record.posBuf      = draw.posVb.buffer()->getSliceInfo().gpuAddress + draw.posVb.offset();
      record.idxBuf      = draw.idxVb.buffer()->getSliceInfo().gpuAddress + draw.idxVb.offset();
      record.wtBuf       = draw.wtVb.buffer()->getSliceInfo().gpuAddress + draw.wtVb.offset();
      record.outBuf      = entry.outPositions->getSliceInfo().gpuAddress;
      record.bonesBuf    = bonesAddress + draw.bonesOffset;
      std::memcpy(record.pivot, draw.pivot, sizeof(record.pivot));
      record.posStride   = draw.posStride;
      record.posFormat   = uint32_t(posFormat);
      record.idxStride   = draw.idxStride;
      record.wtStride    = draw.wtStride;
      record.vertexCount = draw.vertexCount;
      record.firstGroup  = uint32_t(groupMap.size());

      groupMap.insert(groupMap.end(), groups, uint32_t(records.size()));
      records.push_back(record);
      jobs.push_back(job);
    }

    if (jobs.empty())
      return;

    // 2. one host-visible upload: the draw records, then the group map.
    // blessed: hook-cpu-2 -- no bone snapshots any more: each record points
    // at its draw's own b10 allocation, tracked on cmd here so it outlives
    // the dispatch. those bytes are host writes made before this command
    // list is submitted, to host-coherent memory, so the submit makes them
    // visible to the device (the same guarantee the old snapshot upload in
    // this buffer relied on); toCompute below orders the shader read.
    for (const SkinJob& job : jobs)
      cmd->track(m_stagedSkinned[job.pending->staged].bonesAllocation);

    VkDeviceSize recordsOffset = 0;
    VkDeviceSize groupsOffset  = align(recordsOffset + sizeof(BlessedSkinDrawRecord) * records.size(), VkDeviceSize(16));
    VkDeviceSize uploadBytes   = groupsOffset + sizeof(uint32_t) * groupMap.size();

    DxvkBufferCreateInfo uploadInfo = { };
    uploadInfo.size      = uploadBytes;
    uploadInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    uploadInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    uploadInfo.access    = VK_ACCESS_2_SHADER_READ_BIT;
    uploadInfo.debugName = "blessed skinned draws";

    Rc<DxvkBuffer> uploadBuffer = m_device->createBuffer(uploadInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ctx->ensureBufferAddress(uploadBuffer);

    uint8_t* uploadPtr = reinterpret_cast<uint8_t*>(uploadBuffer->getSliceInfo().mapPtr);

    std::memcpy(uploadPtr + recordsOffset, records.data(), sizeof(BlessedSkinDrawRecord) * records.size());
    std::memcpy(uploadPtr + groupsOffset, groupMap.data(), sizeof(uint32_t) * groupMap.size());

    VkDeviceAddress uploadAddress = uploadBuffer->getSliceInfo().gpuAddress;

    // blessed: newly-pinned game vb/ib/wt buffers need their writes visible
    // to the compute shader. skin-repair -- also write-after-read: this
    // frame's compute rewrites outPositions (read by last frame's blas
    // build), the refits below rewrite blases in place (read by every earlier
    // ray query through the tlas), and the builds reuse m_skinScratch. so the
    // source scope is everything before, and the destination covers the
    // builds too. one barrier, only on frames where something was skinned.
    VkMemoryBarrier2 toCompute = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    toCompute.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    toCompute.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    toCompute.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                            | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    toCompute.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT
                            | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
                            | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    VkDependencyInfo toComputeDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    toComputeDep.memoryBarrierCount = 1u;
    toComputeDep.pMemoryBarriers    = &toCompute;

    // blessed: raw
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &toComputeDep);

    // blessed: skin-repair -- skinned_gpu_ms spans the dispatch and the
    // blas builds/refits below
    uint32_t timing = m_skinTimer->begin(cmd);

    // 3. one dispatch for every skinned draw of the frame
    BlessedSkinPushData push = { };
    push.table      = uploadAddress + recordsOffset;
    push.groups     = uploadAddress + groupsOffset;
    push.groupCount = uint32_t(groupMap.size());

    SkinInstance(m_device)->dispatch(cmd, push);

    // 4. compute write (positions) -> AS build read, before the builds/refits
    // below consume those buffers as triangle geometry.
    VkMemoryBarrier2 toAsBuild = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    toAsBuild.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toAsBuild.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    toAsBuild.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    toAsBuild.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo toAsBuildDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    toAsBuildDep.memoryBarrierCount = 1u;
    toAsBuildDep.pMemoryBarriers    = &toAsBuild;

    // blessed: raw
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &toAsBuildDep);

    // blessed: skin-repair -- BLESSED_SCENE_SKIN_TRACE, first frame only
    if (unlikely(m_skinTrace != nullptr && m_skinTrace->wantsRecord(m_frameIndex))) {
      std::vector<BlessedSkinTraceInput> traceInputs;

      for (const SkinJob& job : jobs) {
        BlessedSkinTraceInput in;
        in.draw         = &m_stagedSkinned[job.pending->staged];
        in.outPositions = job.entry->outPositions;
        in.occurrence   = job.pending->key.occurrence;
        traceInputs.push_back(std::move(in));
      }

      m_skinTrace->record(m_device, ctx, cmd, traceInputs);
    }

    // 5. every build and refit in one vkCmdBuildAccelerationStructuresKHR.
    // each info gets its own scratch region; the dst blases are all
    // distinct (one entry per SkinnedKey, one job per entry per frame).
    const VkDeviceSize scratchAlign = m_device->properties().khrAccelerationStructure
      .minAccelerationStructureScratchOffsetAlignment;

    std::vector<VkAccelerationStructureGeometryKHR>          geometries(jobs.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR>    ranges;
    std::vector<VkDeviceSize>                                scratchOffsets;
    std::vector<const SkinJob*>                              built;
    buildInfos.reserve(jobs.size());
    ranges.reserve(jobs.size());
    scratchOffsets.reserve(jobs.size());
    built.reserve(jobs.size());

    VkDeviceSize scratchTotal = 0;

    for (size_t i = 0; i < jobs.size(); i++) {
      const SkinJob& job = jobs[i];
      SkinnedCacheEntry& entry = *job.entry;

      VkAccelerationStructureGeometryTrianglesDataKHR triangles =
        { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
      triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
      triangles.vertexData.deviceAddress = entry.outPositions->getSliceInfo().gpuAddress;
      triangles.vertexStride             = sizeof(float) * 3u;
      triangles.maxVertex                = job.shape.maxVertex;
      triangles.indexType                = job.shape.indexType;
      triangles.indexData.deviceAddress  = job.shape.indexAddress;

      VkAccelerationStructureGeometryKHR& geometry = geometries[i];
      geometry = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
      geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.geometry.triangles = triangles;
      geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;

      VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
        { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
      buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                              | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      buildInfo.mode          = job.refit
        ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
        : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.geometryCount = 1u;
      buildInfo.pGeometries   = &geometry;

      VkDeviceSize scratchSize = entry.updateScratchSize;

      if (!job.refit) {
        // blessed: size query only on a full build; the result depends on
        // flags and geometry counts alone, so the update size is kept for
        // every refit that follows
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
          { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        // blessed: raw
        m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(),
          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &job.shape.primitiveCount, &sizeInfo);

        // a fresh blas every full build: the one the last tlas references is
        // never rewritten under it (that tlas holds its own Rc)
        entry.blas = m_rt->createAccelStruct(
          VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo.accelerationStructureSize);
        entry.shape             = job.shape;
        entry.updateScratchSize = sizeInfo.updateScratchSize;
        scratchSize             = sizeInfo.buildScratchSize;
      }

      if (entry.blas == nullptr || !entry.blas->handle()) {
        entry.blas = nullptr;
        continue;
      }

      buildInfo.srcAccelerationStructure = job.refit ? entry.blas->handle() : VK_NULL_HANDLE;
      buildInfo.dstAccelerationStructure = entry.blas->handle();

      VkAccelerationStructureBuildRangeInfoKHR range = { };
      range.primitiveCount  = job.shape.primitiveCount;
      range.primitiveOffset = job.shape.primitiveOffset;
      range.firstVertex     = job.shape.firstVertex;

      scratchOffsets.push_back(scratchTotal);
      scratchTotal += align(std::max<VkDeviceSize>(scratchSize, 1u), scratchAlign);

      buildInfos.push_back(buildInfo);
      ranges.push_back(range);
      built.push_back(&job);
    }

    if (!buildInfos.empty()) {
      // blessed: one extra alignment step so the base can be rounded up
      // even if the allocation itself is less aligned than scratchAlign
      VkDeviceAddress scratchBase = align(
        ensureScratch(ctx, cmd, scratchTotal + scratchAlign, m_skinScratch), scratchAlign);

      std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs(ranges.size());

      for (size_t i = 0; i < buildInfos.size(); i++) {
        buildInfos[i].scratchData.deviceAddress = scratchBase + scratchOffsets[i];
        rangePtrs[i] = &ranges[i];
      }

      cmd->cmdBuildAccelerationStructures(DxvkCmdBuffer::ExecBuffer,
        uint32_t(buildInfos.size()), buildInfos.data(), rangePtrs.data());

      for (const SkinJob* job : built) {
        SkinnedCacheEntry& entry = *job->entry;
        cmd->track(entry.blas);

        if (job->refit)
          m_stats.skinnedRefits.fetch_add(1, std::memory_order_relaxed);
        buildsThisFrame++;

        VkAccelerationStructureInstanceKHR instance = { };
        // blessed: identity -- blessed_skin.comp already wrote camera-relative
        // positions (bone blend minus the pivot), so no per-instance
        // transform is needed here, unlike the static path's captured 3x4.
        instance.transform.matrix[0][0] = 1.0f;
        instance.transform.matrix[1][1] = 1.0f;
        instance.transform.matrix[2][2] = 1.0f;
        instance.instanceCustomIndex           = job->pending->instanceId & 0xFFFFFFu;
        instance.mask                          = 0xFFu;
        instance.flags                         = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = entry.blas->address();

        instances.push_back(instance);
        referencedBlases.push_back(entry.blas);

        // blessed: gi v1 -- sentinel (vbAddress 0, albedoId 0): skinned
        // actors don't carry per-texture albedo or a static geometry table
        // entry in this seat (their positions live in a compute output
        // buffer, not the vb/ib shape BlessedGiInstanceInfo describes). The
        // trace shader's zero-address check falls back to -ray-direction
        // shading and the BLESSED_GI_ALBEDO constant for these, same as v0.
        instanceInfos.push_back(BlessedGiInstanceInfo{});
      }
    }

    m_skinTimer->end(cmd, timing); // blessed: skin-repair

    // blessed: skin-v2 -- every buffer read or written above by raw device
    // address lives until this submission is done, even if its entry is
    // evicted or resized, or the game releases it, before the gpu gets here
    cmd->track(uploadBuffer);

    for (const SkinJob& job : jobs) {
      const BlessedSceneSkinnedDraw& draw = m_stagedSkinned[job.pending->staged];
      cmd->track(job.entry->outPositions);
      cmd->track(draw.posVb.buffer());
      cmd->track(draw.idxVb.buffer());
      cmd->track(draw.wtVb.buffer());
      cmd->track(draw.ib.buffer());
    }
  }


  BlessedSceneFrame BlessedScene::currentFrame() const {
    std::lock_guard<sync::Spinlock> lock(m_frameLock);
    return m_publishedFrame;
  }


  void BlessedScene::maybeTraceBuild(
    const CacheKey&   key,
    const CacheEntry& entry,
    VkDeviceAddress   vbAddress,
    VkDeviceAddress   ibAddress,
    uint32_t          maxVertex,
    uint32_t          primitiveCount) {
    if (!TraceEnabled() || m_traceCount >= MaxTracedBuilds)
      return;

    if (!m_traceFileTried) {
      m_traceFileTried = true;

      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
      if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        m_traceFile.open(dir + env::PlatformDirSlash + "scene-trace.log",
          std::ios::out | std::ios::app);
      }
    }

    if (!m_traceFile.is_open())
      return;

    m_traceCount++;

    m_traceFile << str::format("blas[", m_traceCount, "]",
      " vb_addr=", vbAddress, " vb_offset=", key.vbOffset,
      " vb_stride=", key.vbStride, " vb_format=", uint32_t(key.vbFormat),
      " vb_size=", entry.vb != nullptr ? entry.vb->info().size : VkDeviceSize(0),
      " ib_addr=", ibAddress, " ib_offset=", key.ibOffset,
      " ib_index_type=", uint32_t(key.indexType),
      " ib_size=", entry.ib != nullptr ? entry.ib->info().size : VkDeviceSize(0),
      " index_count=", key.indexCount, " first_index=", key.startIndex,
      " base_vertex=", key.baseVertex, " max_vertex=", maxVertex,
      " primitive_count=", primitiveCount, "\n");
    m_traceFile.flush();
  }

}
