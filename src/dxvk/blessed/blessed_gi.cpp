// blessed: ray-traced gi probe grid -- see blessed_gi.h
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "blessed_async.h" // blessed: async-compute
#include "blessed_gi.h"
#include "blessed_scene.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"

#include "../../util/util_blessed_probe.h" // blessed: hook-cpu-2
#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"

#include <blessed_gi_trace.h>
#include <blessed_gi_albedo.h>

namespace dxvk {

  namespace {

    // blessed: push data layout must match BlessedGiTracePushData in
    // shaders/blessed_gi_trace.comp exactly (scalar layout).
    struct BlessedGiTracePushData {
      uint64_t tlasAddress;
      uint64_t probeBufferAddress;
      uint64_t instanceInfoAddress; // gi v1 -- BlessedGiInstanceInfo[], may be 0
      uint64_t albedoBufferAddress; // gi v1 -- vec4[MaxAlbedoTextures], may be 0
      float    camPos[3];
      int32_t  originCell[3];
      int32_t  dims[3];
      float    spacing;
      float    rotAngle;
      float    sunDir[3];
      float    sunColor[3];
      float    albedo;   // BLESSED_GI_ALBEDO -- fallback for albedoId 0 / v0Mode
      float    skyMul;
      float    skyBounce; // gi v1 -- BLESSED_GI_SKYBOUNCE
      float    ambientRow[12];
      float    alpha;
      uint32_t raysPerProbe;
      uint32_t instanceInfoCount; // gi v1
      float    backfaceThreshold; // gi v1 -- BLESSED_GI_BACKFACE
      uint32_t v0Mode;            // gi v1 -- BLESSED_GI_V0=1
      uint32_t probeStride;       // gi-bounds -- 13, or 14 with openness
    };

    bool ParseInts3(const std::string& s, int32_t out[3]) {
      size_t pos = 0;
      for (uint32_t i = 0; i < 3; i++) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (tok.empty())
          return false;
        out[i] = std::atoi(tok.c_str());
        if (comma == std::string::npos) {
          if (i + 1 != 3)
            return false;
        } else {
          pos = comma + 1;
        }
      }
      return true;
    }

    int32_t FloorDiv(float value, float spacing) {
      return int32_t(std::floor(value / spacing));
    }

    int32_t WrapMod(int32_t x, int32_t n) {
      int32_t r = x % n;
      return r < 0 ? r + n : r;
    }

    // blessed: hook-cpu-2 -- int32_t(std::floor(x)) without the floorf call
    // msvc emits. Same result for every input: truncation is floor for
    // x >= 0 and for integral x; a negative fraction steps down one. The
    // INT32_MIN guard keeps NaN and out-of-range inputs at the value
    // cvttss2si gives the original expression.
    inline int32_t FloorToInt(float x) {
      int32_t i = int32_t(x);
      if (float(i) > x && i != INT32_MIN)
        i--;
      return i;
    }

    // blessed: gi v1 -- push data layout must match BlessedGiAlbedoPushData
    // in shaders/blessed_gi_albedo.comp exactly (scalar layout).
    struct BlessedGiAlbedoPushData {
      uint64_t albedosAddress;
      uint32_t samplerIndex;
      uint32_t albedoId;
      uint32_t needsSrgbDecode;
    };

  }


  bool BlessedGiState::isEnabled() {
    static bool s_enabled = env::getEnvVar("BLESSED_GI") == "probes";
    return s_enabled;
  }


  BlessedGiState::BlessedGiState(DxvkDevice* device)
  : m_device(device) {
    int32_t grid[3];
    if (ParseInts3(env::getEnvVar("BLESSED_GI_GRID"), grid)) {
      m_config.dimX = std::max(1, grid[0]);
      m_config.dimY = std::max(1, grid[1]);
      m_config.dimZ = std::max(1, grid[2]);
    }

    std::string spacingStr = env::getEnvVar("BLESSED_GI_SPACING");
    if (!spacingStr.empty())
      m_config.spacing = std::max(1.0f, std::strtof(spacingStr.c_str(), nullptr));

    std::string raysStr = env::getEnvVar("BLESSED_GI_RAYS");
    if (!raysStr.empty())
      m_config.rays = std::max(1u, uint32_t(std::strtoul(raysStr.c_str(), nullptr, 10)));

    std::string albedoStr = env::getEnvVar("BLESSED_GI_ALBEDO");
    if (!albedoStr.empty())
      m_config.albedo = std::strtof(albedoStr.c_str(), nullptr);

    std::string skyStr = env::getEnvVar("BLESSED_GI_SKY");
    if (!skyStr.empty())
      m_config.sky = std::strtof(skyStr.c_str(), nullptr);

    // blessed: gi v1
    std::string skyBounceStr = env::getEnvVar("BLESSED_GI_SKYBOUNCE");
    if (!skyBounceStr.empty())
      m_config.skyBounce = std::strtof(skyBounceStr.c_str(), nullptr);

    std::string backfaceStr = env::getEnvVar("BLESSED_GI_BACKFACE");
    if (!backfaceStr.empty())
      m_config.backface = std::strtof(backfaceStr.c_str(), nullptr);

    m_v0Mode = env::getEnvVar("BLESSED_GI_V0") == "1";
    m_flipWinding = env::getEnvVar("BLESSED_GI_WINDING") == "flip";

    m_debugIrradiance = env::getEnvVar("BLESSED_GI_DEBUG") == "irradiance";

    // blessed: gi-bounds -- origin (default) keeps the 13-float stride and
    // every sample exactly as before
    m_sampleBounds = env::getEnvVar("BLESSED_GI_SAMPLE") != "origin"; // blessed: bounds is the default (2026-09-23)

    if (m_sampleBounds) {
      m_probeStride = ProbeStride + 1u;

      std::string opennessStr = env::getEnvVar("BLESSED_GI_OPENNESS");
      if (!opennessStr.empty())
        m_openness = std::clamp(std::strtof(opennessStr.c_str(), nullptr), 0.0f, 1.0f);

      std::string pullStr = env::getEnvVar("BLESSED_GI_BOUNDS_PULL");
      if (!pullStr.empty())
        m_boundsPull = std::clamp(std::strtof(pullStr.c_str(), nullptr), 0.0f, 1.0f);

      Logger::info(str::format("BlessedGiState: sample=bounds openness=", m_openness,
        " pull=", m_boundsPull));
    }

    if (device->properties().core.properties.limits.timestampComputeAndGraphics) {
      VkQueryPoolCreateInfo queryInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
      queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
      queryInfo.queryCount = 2u;

      VkResult vr = device->vkd()->vkCreateQueryPool(
        device->handle(), &queryInfo, nullptr, &m_queryPool);

      if (vr != VK_SUCCESS) {
        Logger::warn("blessed: gi: failed to create timestamp query pool, timing disabled");
        m_queryPool = VK_NULL_HANDLE;
      }
    }

    Logger::info(str::format("BlessedGiState: enabled, grid=", m_config.dimX, "x", m_config.dimY, "x", m_config.dimZ,
      " spacing=", m_config.spacing, " rays=", m_config.rays, " albedo=", m_config.albedo, " sky=", m_config.sky,
      " skyBounce=", m_config.skyBounce, " backface=", m_config.backface, " v0=", m_v0Mode ? 1 : 0));
  }


  BlessedGiState::~BlessedGiState() {
    if (m_queryPool != VK_NULL_HANDLE)
      m_device->vkd()->vkDestroyQueryPool(m_device->handle(), m_queryPool, nullptr);
  }


  void BlessedGiState::noteFrameLighting(const BlessedGiFrameLighting& lighting) {
    // blessed: cs-thread-only (see the class comment) -- plain assignment.
    m_pendingLighting = lighting;
  }


  void BlessedGiState::ensureBuffers(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd) {
    if (m_probeBuffer != nullptr)
      return;

    VkDeviceSize probeCount = VkDeviceSize(m_config.dimX) * m_config.dimY * m_config.dimZ;
    VkDeviceSize totalBytes = probeCount * m_probeStride * sizeof(float);

    DxvkBufferCreateInfo probeInfo = { };
    probeInfo.size      = totalBytes;
    probeInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    probeInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    probeInfo.access    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    probeInfo.debugName = "blessed gi probes";

    m_probeBuffer = m_device->createBuffer(probeInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ctx->ensureBufferAddress(m_probeBuffer);

    for (uint32_t i = 0; i < RingSize; i++) {
      DxvkBufferCreateInfo ringInfo = { };
      ringInfo.size      = totalBytes;
      ringInfo.usage     = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      ringInfo.stages    = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      ringInfo.access    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      ringInfo.debugName = "blessed gi probe readback ring";

      // blessed: HOST_CACHED_BIT -- lead's fix, campaign c3: without it this
      // memory is write-combined on nvidia, and reading it back (even in
      // the single once-a-frame bulk copy refreshCsCache now does, let
      // alone the old per-draw reads) is expensive. See PROJECT.md.
      m_ring[i] = m_device->createBuffer(ringInfo,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
      std::memset(m_ring[i]->getSliceInfo().mapPtr, 0, size_t(totalBytes));
    }

    // blessed: zero the accumulator before the first trace ever reads it --
    // uninitialized device memory could be NaN/garbage, which the temporal
    // mix() would then never fully recover from.
    ctx->clearBuffer(m_probeBuffer, 0, totalBytes, 0u);
    cmd->track(m_probeBuffer, DxvkAccess::Write);
  }


  void BlessedGiState::collectTiming() {
    if (m_queryPool == VK_NULL_HANDLE || m_traceCount == 0)
      return;

    struct { uint64_t ts; uint64_t avail; } results[2] = { };

    VkResult vr = m_device->vkd()->vkGetQueryPoolResults(
      m_device->handle(), m_queryPool, 0u, 2u,
      sizeof(results), results, sizeof(results[0]),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (vr != VK_SUCCESS || !results[0].avail || !results[1].avail)
      return;

    double period = double(m_device->properties().core.properties.limits.timestampPeriod);
    double ms = double(results[1].ts - results[0].ts) * period / 1.0e6;

    std::lock_guard<sync::Spinlock> lock(m_publishLock);
    m_lastTraceGpuMs = ms;
  }


  void BlessedGiState::runTrace(
          DxvkContext*         ctx,
    const Rc<DxvkCommandList>& cmd) {
    if (!m_pendingLighting.valid)
      return; // no pass-115 draw seen yet this frame (e.g. a loading screen)

    BlessedScene* scene = m_device->blessedScene();
    if (!scene)
      return;

    BlessedSceneFrame frame = scene->currentFrame();
    if (!frame.valid || frame.tlas == nullptr)
      return; // no scene yet -- nothing to trace against

    ensureBuffers(ctx, cmd);

    // blessed: non-blocking readback of *last* frame's timestamp pair,
    // before this frame's queries reuse the same two slots.
    if (m_queryPool != VK_NULL_HANDLE && m_traceCount > 0)
      collectTiming();

    VkDeviceSize probeCount = VkDeviceSize(m_config.dimX) * m_config.dimY * m_config.dimZ;
    VkDeviceSize totalBytes = probeCount * m_probeStride * sizeof(float);

    // blessed: gi v1 -- one dispatch per newly-seen texture, if any are
    // queued (empty in v0Mode: OnDraw never sends a note then). Always safe
    // here: scene->endFrame (called just before blessedRunGiTrace, see
    // dxvk_context.cpp) already ended the game's render pass for its own AS
    // builds.
    drainPendingAlbedoTextures(ctx, cmd);

    // blessed: WAR/RAW -- wait for the previous frame's copy-out (transfer
    // read of m_probeBuffer) to finish before this frame's compute pass
    // writes it again; on the very first call this instead covers
    // ensureBuffers' own clearBuffer (transfer *write*), which the first
    // trace's read-modify-write must not race. Global memory barrier: this
    // pass runs at most once a frame, so the extra sync scope costs
    // nothing measurable.
    //
    // blessed: gi v1 -- also covers drainPendingAlbedoTextures' own compute
    // writes into m_albedoBuffer, which the dispatch below reads (RAW):
    // same-queue submission order alone does not make a compute write
    // visible to a later compute read without a barrier between them.
    VkMemoryBarrier2 preBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    preBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    preBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    preBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    preBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

    VkDependencyInfo preDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preDep.memoryBarrierCount = 1u;
    preDep.pMemoryBarriers    = &preBarrier;
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &preDep); // blessed: raw

    // blessed: gi-bounds -- the queued meshes' aabb pass, after the barrier
    // above (it also makes the game's own vb/ib uploads visible to compute
    // reads) and before the trace timestamps. Empty unless sample=bounds.
    if (!m_boundsJobs.empty())
      dispatchBoundsJobs(ctx, cmd);

    // blessed: async-compute -- from here to the ring copy, the trace records
    // into the async queue's buffer (BLESSED_ASYNC=1, pass "gi"). Everything
    // above (albedo drain, bounds jobs: game textures and vertex buffers)
    // stayed on graphics; the kick waits for it. Nothing on graphics waits
    // for the kick until the next frame's scene end frame (blessedAsyncSync):
    // the only reader is the cpu, two traces later, as before.
    bool async = ctx->blessedAsyncAvailable() && BlessedAsync::PassRequested(BlessedAsyncPass::Gi);

    if (async) {
      ctx->blessedAsyncBegin();

      // same scopes as preBarrier, now on the async queue: orders this trace
      // after the previous kick's ring copy (a barrier's first scope spans
      // earlier submissions on the same queue); the graphics-side writes
      // above reach it through the kick's semaphore wait.
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &preDep); // blessed: raw
    }

    if (m_queryPool != VK_NULL_HANDLE) {
      cmd->cmdResetQueryPool(DxvkCmdBuffer::ExecBuffer, m_queryPool, 0u, 2u);
      cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, 0u);
    }

    // blessed: golden-angle azimuth increment -- decorrelates the fixed
    // fibonacci-sphere ray pattern frame to frame for the temporal blend,
    // cheaply (one float, no gpu rng state).
    float rotAngle = std::fmod(float(m_traceCount) * 2.39996323f, 6.28318531f);

    int32_t originCell[3] = {
      FloorDiv(frame.camPos[0], m_config.spacing) - m_config.dimX / 2,
      FloorDiv(frame.camPos[1], m_config.spacing) - m_config.dimY / 2,
      FloorDiv(frame.camPos[2], m_config.spacing) - m_config.dimZ / 2,
    };

    static const DxvkPipelineLayout* s_layout = m_device->createBuiltInPipelineLayout(
      DxvkPipelineLayoutFlags(), VK_SHADER_STAGE_COMPUTE_BIT,
      sizeof(BlessedGiTracePushData), 0u, nullptr);
    static VkPipeline s_pipeline = [this] {
      util::DxvkBuiltInShaderStage stage(blessed_gi_trace, nullptr);
      return m_device->createBuiltInComputePipeline(s_layout, stage);
    }();

    BlessedGiTracePushData push = { };
    push.tlasAddress         = frame.tlasAddress;
    push.probeBufferAddress  = m_probeBuffer->getSliceInfo().gpuAddress;
    // blessed: gi v1 -- 0 when the scene had no static instances this frame
    // (or v0Mode never resolved any albedo, so m_albedoBuffer was never
    // created); the shader treats either as "no geometry/albedo table".
    push.instanceInfoAddress = m_v0Mode ? 0 : frame.instanceInfoAddress;
    push.instanceInfoCount   = m_v0Mode ? 0 : frame.instanceInfoCount;
    push.albedoBufferAddress = (!m_v0Mode && m_albedoBuffer != nullptr)
      ? m_albedoBuffer->getSliceInfo().gpuAddress : 0;
    std::memcpy(push.camPos, frame.camPos, sizeof(push.camPos));
    std::memcpy(push.originCell, originCell, sizeof(push.originCell));
    push.dims[0] = m_config.dimX;
    push.dims[1] = m_config.dimY;
    push.dims[2] = m_config.dimZ;
    push.spacing = m_config.spacing;
    push.rotAngle = rotAngle;
    std::memcpy(push.sunDir, m_pendingLighting.sunDir, sizeof(push.sunDir));
    std::memcpy(push.sunColor, m_pendingLighting.sunColor, sizeof(push.sunColor));
    push.albedo = m_config.albedo;
    push.skyMul = m_config.sky;
    push.skyBounce = m_config.skyBounce;
    std::memcpy(push.ambientRow, m_pendingLighting.ambientRow, sizeof(push.ambientRow));
    push.alpha = 0.05f; // blessed: fixed per the brief, not exposed as an env knob
    push.raysPerProbe = m_config.rays;
    push.backfaceThreshold = m_config.backface;
    push.v0Mode = (m_v0Mode ? 1u : 0u) | (m_flipWinding ? 2u : 0u);
    push.probeStride = m_probeStride; // blessed: gi-bounds

    cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s_pipeline);
    cmd->bindResources(DxvkCmdBuffer::ExecBuffer, s_layout,
      0u, nullptr, sizeof(push), &push);

    uint32_t groups = (uint32_t(probeCount) + 63u) / 64u;
    cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, groups, 1u, 1u);

    if (m_queryPool != VK_NULL_HANDLE) {
      cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_queryPool, 1u);
    }

    // blessed: RAW -- the copy must not start reading until the dispatch
    // above finishes writing.
    VkMemoryBarrier2 postBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    postBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    postBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo postDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.memoryBarrierCount = 1u;
    postDep.pMemoryBarriers    = &postBarrier;
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &postDep); // blessed: raw

    uint32_t writeSlot = uint32_t(m_traceCount % RingSize);
    std::memcpy(m_ringOrigin[writeSlot], originCell, sizeof(originCell)); // blessed: gi-bounds

    if (async) {
      // blessed: async-compute -- a raw copy; ctx->copyBuffer would record
      // the context's own barriers into the async buffer
      DxvkResourceBufferInfo srcInfo = m_probeBuffer->getSliceInfo();
      DxvkResourceBufferInfo dstInfo = m_ring[writeSlot]->getSliceInfo();

      VkBufferCopy2 region = { VK_STRUCTURE_TYPE_BUFFER_COPY_2 };
      region.srcOffset = srcInfo.offset;
      region.dstOffset = dstInfo.offset;
      region.size      = totalBytes;

      VkCopyBufferInfo2 copyInfo = { VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2 };
      copyInfo.srcBuffer   = srcInfo.buffer;
      copyInfo.dstBuffer   = dstInfo.buffer;
      copyInfo.regionCount = 1u;
      copyInfo.pRegions    = &region;
      cmd->cmdCopyBuffer(DxvkCmdBuffer::ExecBuffer, &copyInfo);

      // the cpu reads the ring (two traces later, no fence, as before)
      VkMemoryBarrier2 hostBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      hostBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      hostBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      hostBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
      hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

      VkDependencyInfo hostDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      hostDep.memoryBarrierCount = 1u;
      hostDep.pMemoryBarriers    = &hostBarrier;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &hostDep); // blessed: raw

      // the ring, the tlas and the instance table are tracked below, the
      // same as on the graphics path
      cmd->track(m_probeBuffer, DxvkAccess::Write);

      if (m_albedoBuffer != nullptr)
        cmd->track(m_albedoBuffer, DxvkAccess::Read);

      ctx->blessedAsyncEnd();
    } else {
      ctx->copyBuffer(m_ring[writeSlot], 0, m_probeBuffer, 0, totalBytes);
    }

    cmd->track(m_ring[writeSlot], DxvkAccess::Write);
    cmd->track(frame.tlas);

    // blessed: gi v1 -- this dispatch just read instanceInfoAddress; keep the
    // buffer it points into alive for exactly this submission, same
    // reasoning as tracking frame.tlas above.
    if (frame.instanceInfoBuffer != nullptr)
      cmd->track(frame.instanceInfoBuffer);

    // blessed: publish the slot from *two* traces ago -- a full ring-size
    // margin of gpu-completion slack beyond the one-frame pipeline latency
    // that already exists (this trace runs at the end of the frame that
    // produced it; the earliest reader is next frame's draws).
    // No fence backs this -- see the class comment and the seat report.
    if (m_traceCount >= 2) {
      uint32_t readySlot = uint32_t((m_traceCount + RingSize - 2) % RingSize);
      std::lock_guard<sync::Spinlock> lock(m_publishLock);
      m_published.ringPtr = reinterpret_cast<const float*>(m_ring[readySlot]->getSliceInfo().mapPtr);
      m_published.dims[0] = m_config.dimX;
      m_published.dims[1] = m_config.dimY;
      m_published.dims[2] = m_config.dimZ;
      m_published.spacing = m_config.spacing;
      m_published.valid   = true;
      std::memcpy(m_published.origin, m_ringOrigin[readySlot], sizeof(m_published.origin)); // blessed: gi-bounds
    }

    m_traceCount++;
    m_pendingLighting.valid = false; // consumed
  }


  void BlessedGiState::refreshCsCache() {
    Published pub;
    {
      std::lock_guard<sync::Spinlock> lock(m_publishLock);
      pub = m_published;
    }

    // blessed: gi-cs -- every refresh starts a new generation, including
    // the one that finds nothing to copy (per-frame memos key off this)
    m_csCacheGeneration++;

    // blessed: gi-bounds -- read back the bounds dispatched a few traces ago
    if (m_sampleBounds)
      pollBounds();

    if (!pub.valid || pub.ringPtr == nullptr) {
      m_csCacheValid = false;
      return;
    }

    size_t count = size_t(pub.dims[0]) * size_t(pub.dims[1]) * size_t(pub.dims[2]) * m_probeStride; // blessed: gi-bounds, was ProbeStride
    if (m_csCache.size() != count)
      m_csCache.resize(count);

    // blessed: the one gpu-mapped read per frame -- see the class doc on
    // \ref refreshCsCache. Every later sampleAmbient call this frame reads
    // m_csCache instead, plain process memory.
    std::memcpy(m_csCache.data(), pub.ringPtr, count * sizeof(float));

    m_csCacheDims[0] = pub.dims[0];
    m_csCacheDims[1] = pub.dims[1];
    m_csCacheDims[2] = pub.dims[2];
    m_csCacheSpacing = pub.spacing;
    std::memcpy(m_csCacheOrigin, pub.origin, sizeof(m_csCacheOrigin)); // blessed: gi-bounds
    m_csCacheValid   = true;

    // blessed: hook-cpu-2 -- the cell cache describes the old data
    for (CsCell& cell : m_csCells)
      cell.used = false;

    // blessed: gi-cs -- the log's probes_invalid, recounted here (the cs
    // thread owns m_csCache) once per log window instead of per log read
    if (++m_refreshesSinceCount >= InvalidCountInterval) {
      m_refreshesSinceCount = 0;
      m_invalidProbes.store(countInvalidProbesNow(), std::memory_order_relaxed);

      if (m_sampleBounds) // blessed: gi-bounds
        m_meanOpenness.store(meanOpennessNow(), std::memory_order_relaxed);
    }
  }


  const BlessedGiState::CsCell& BlessedGiState::lookupCsCell(const int32_t c0[3]) const {
    uint32_t h = uint32_t(c0[0]) * 73856093u ^ uint32_t(c0[1]) * 19349663u ^ uint32_t(c0[2]) * 83492791u;
    CsCell& cell = m_csCells[h % CsCellCount];

    if (cell.used && cell.key[0] == c0[0] && cell.key[1] == c0[1] && cell.key[2] == c0[2])
      return cell;

    // blessed: gi-cs -- no probe scope here any more: this runs on the cs
    // thread now, and the probe's call counters are app-thread only.

    int32_t slots[3][2];
    for (int a = 0; a < 3; a++) {
      slots[a][0] = WrapMod(c0[a],     m_csCacheDims[a]);
      slots[a][1] = WrapMod(c0[a] + 1, m_csCacheDims[a]);
    }

    cell.validMask = 0;

    for (int dz = 0; dz < 2; dz++) {
      for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
          int64_t linear = int64_t(slots[0][dx]) + int64_t(slots[1][dy]) * m_csCacheDims[0]
                          + int64_t(slots[2][dz]) * m_csCacheDims[0] * m_csCacheDims[1];

          uint32_t corner = uint32_t(dz * 4 + dy * 2 + dx);
          cell.offset[corner] = uint32_t(linear * m_probeStride);

          // same test sampleAmbient always made: sh[12] <= 0.5f is invalid
          if (!(m_csCache[cell.offset[corner] + 12] <= 0.5f))
            cell.validMask |= uint8_t(1u << corner);
        }
      }
    }

    cell.key[0] = c0[0];
    cell.key[1] = c0[1];
    cell.key[2] = c0[2];
    cell.used   = true;
    return cell;
  }


  uint32_t BlessedGiState::countInvalidProbesNow() const {
    if (!m_csCacheValid)
      return 0;

    uint32_t total = uint32_t(m_csCacheDims[0]) * uint32_t(m_csCacheDims[1]) * uint32_t(m_csCacheDims[2]);
    uint32_t invalid = 0;

    for (uint32_t i = 0; i < total; i++) {
      if (m_csCache[size_t(i) * m_probeStride + 12] <= 0.5f)
        invalid++;
    }

    return invalid;
  }


  bool BlessedGiState::sampleAmbient(const float absWorldPos[3], float outRow[12]) const {
    // blessed: gi v1 -- cs-thread-only since gi-cs, no lock (see refreshCsCache's doc).
    if (!m_csCacheValid)
      return false;

    float base[3];
    int32_t c0[3];
    float frac[3];
    for (int a = 0; a < 3; a++) {
      base[a] = absWorldPos[a] / m_csCacheSpacing - 0.5f;
      c0[a]   = FloorToInt(base[a]); // blessed: hook-cpu-2, was int32_t(std::floor(...))
      frac[a] = base[a] - float(c0[a]);
    }

    float acc[12] = { };
    float weightSum = 0.0f;

    // blessed: hook-cpu-2 -- corner offsets and validity from the cell
    // cache (see CsCell); weights and accumulation order are unchanged
    const CsCell& cell = lookupCsCell(c0);

    for (int dz = 0; dz < 2; dz++) {
      float wz = dz ? frac[2] : (1.0f - frac[2]);
      for (int dy = 0; dy < 2; dy++) {
        float wy = dy ? frac[1] : (1.0f - frac[1]);
        for (int dx = 0; dx < 2; dx++) {
          float wx = dx ? frac[0] : (1.0f - frac[0]);
          float weight = wx * wy * wz;
          if (weight <= 0.0f)
            continue;

          uint32_t corner = uint32_t(dz * 4 + dy * 2 + dx);
          const float* sh = m_csCache.data() + cell.offset[corner];

          // blessed: gi v1 -- probes-inside-walls fix. sh[12] is this
          // probe's temporally-blended validity (>0.5 valid, see
          // blessed_gi_trace.comp); an invalid probe contributes nothing,
          // and its weight is redistributed over whichever corners *are*
          // valid (renormalized below). All 8 invalid -> weightSum stays 0
          // -> return false -> BlessedGi::PatchOnCs leaves vanilla's own
          // ambient untouched, exactly the brief's fallback.
          if (!(cell.validMask & (1u << corner)))
            continue; // blessed: hook-cpu-2, sh[12] <= 0.5f as of the cell's lookup

          weightSum += weight;
          for (int i = 0; i < 12; i++)
            acc[i] += sh[i] * weight;
        }
      }
    }

    if (weightSum <= 0.0f)
      return false;

    float invWeight = 1.0f / weightSum;
    for (int i = 0; i < 12; i++)
      outRow[i] = acc[i] * invWeight;

    return true;
  }


  uint32_t BlessedGiState::ensureAlbedo(const Rc<DxvkImageView>& view, bool needsSrgbDecode) {
    // blessed: cs-thread only -- see the header doc. Never records gpu work:
    // this may run mid the game's own render pass (called from the same
    // per-draw EmitCs lambda addDraw/noteMeshAlbedo run from).
    if (view == nullptr)
      return 0;

    DxvkImage* image = view->image();
    auto it = m_albedoIndex.find(image);
    if (it != m_albedoIndex.end())
      return it->second;

    if (m_nextAlbedoId >= MaxAlbedoTextures) {
      // blessed: cap reached -- stay silent past this point (logged once
      // would need its own dedup); the mesh just keeps BLESSED_GI_ALBEDO.
      return 0;
    }

    uint32_t id = m_nextAlbedoId++;
    m_albedoIndex.emplace(image, id);
    m_pendingAlbedoTextures.push_back({ view, id, needsSrgbDecode });
    return id;
  }


  void BlessedGiState::ensureAlbedoObjects(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd) {
    if (m_albedoBuffer != nullptr)
      return;

    VkDeviceSize totalBytes = VkDeviceSize(MaxAlbedoTextures) * sizeof(float) * 4u;

    DxvkBufferCreateInfo bufferInfo = { };
    bufferInfo.size      = totalBytes;
    bufferInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    bufferInfo.access    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    bufferInfo.debugName = "blessed gi albedo table";

    m_albedoBuffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ctx->ensureBufferAddress(m_albedoBuffer);

    // blessed: fill every slot to BLESSED_GI_ALBEDO, repeated across all 4
    // lanes of every vec4 -- an id whose texture dispatch hasn't landed yet
    // (or never will, past MaxAlbedoTextures) reads back exactly what v0
    // always used: never black, never garbage.
    uint32_t pattern;
    std::memcpy(&pattern, &m_config.albedo, sizeof(pattern));
    ctx->clearBuffer(m_albedoBuffer, 0, totalBytes, pattern);
    cmd->track(m_albedoBuffer, DxvkAccess::Write);

    DxvkSamplerKey samplerInfo = { };
    samplerInfo.setFilter(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST);
    samplerInfo.setAddressModes(
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    samplerInfo.setLodRange(0.0f, 16.0f, 0.0f); // deep enough for any real mip chain
    samplerInfo.setUsePixelCoordinates(false);
    m_albedoSampler = m_device->createSampler(samplerInfo);

    static const std::array<DxvkDescriptorSetLayoutBinding, 1> bindings = {{
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
    }};

    m_albedoLayout = m_device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
      VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedGiAlbedoPushData),
      uint32_t(bindings.size()), bindings.data());

    util::DxvkBuiltInShaderStage stage(blessed_gi_albedo, nullptr);
    m_albedoPipeline = m_device->createBuiltInComputePipeline(m_albedoLayout, stage);
  }


  void BlessedGiState::drainPendingAlbedoTextures(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd) {
    if (m_pendingAlbedoTextures.empty())
      return;

    ensureAlbedoObjects(ctx, cmd);

    for (const PendingAlbedoTexture& tex : m_pendingAlbedoTextures) {
      // blessed: read-after-read only (the game's own diffuse textures are
      // never gpu-written after upload) -- no image transition needed
      // before sampling, unlike blessed_shadow.cpp's TransitionForCompute
      // (which handles a cycling attachment, not the case here). The
      // view's own descriptor already carries whatever layout it samples
      // in for the game's own draws.
      DxvkDescriptorWrite descriptor = { };
      descriptor.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      descriptor.descriptor     = tex.view->getDescriptor();

      BlessedGiAlbedoPushData push = { };
      push.albedosAddress   = m_albedoBuffer->getSliceInfo().gpuAddress;
      push.samplerIndex     = m_albedoSampler->getDescriptor().samplerIndex;
      push.albedoId         = tex.id;
      push.needsSrgbDecode  = tex.needsSrgbDecode ? 1u : 0u;

      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_albedoPipeline);
      cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_albedoLayout,
        1u, &descriptor, sizeof(push), &push);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, 1u, 1u, 1u);

      cmd->track(tex.view->image(), DxvkAccess::Read);
    }

    // blessed: the probe trace dispatch that follows reads m_albedoBuffer --
    // see runTrace's preBarrier, extended to cover this write.
    m_pendingAlbedoTextures.clear();
  }

}
