// blessed: BLESSED_SHADOW_SOFT=1 -- soft, temporally-accumulated ray-traced
// sun shadows. See blessed_soft_shadow.h.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "blessed_soft_shadow.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"
#include "../dxvk_device.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"

#include <blessed_soft_shadow_trace.h>
#include <blessed_soft_shadow_filter.h>

namespace dxvk {

  namespace {

    // blessed: matches BlessedSoftShadowConstants in both .comp files,
    // byte-for-byte (scalar layout: tight packing, no vec3->vec4 padding,
    // same convention as BlessedShadowPushData in blessed_shadow.cpp)
    struct BlessedSoftShadowConstants {
      float    invViewProj[16];
      float    prevViewProj[16];
      float    camDelta[3];
      float    sunDir[3];
      float    reprojDelta[3];
      float    farDepthValue;
      float    sunHalfAngleRad;
      uint32_t spp;
      uint32_t frameIndex;
      uint32_t historyValid;
    };

    // blessed: matches BlessedSoftShadowPush in both .comp files -- shared
    // by the trace and filter dispatches, some fields meaningless in one
    // or the other (see the comments on each .comp file's copy)
    struct BlessedSoftShadowPush {
      uint64_t constantsAddress;
      uint64_t tlasAddress;
      uint64_t debugAddress;
      uint32_t width;
      uint32_t height;
      uint32_t samplerIndex;
      uint32_t tlasValid;
      uint32_t debugMode;
    };

    static_assert(sizeof(BlessedSoftShadowPush) <= 128,
      "must fit the Vulkan-guaranteed minimum push constant budget");

    // blessed: general 4x4 inverse (cofactors, in double). Layout-agnostic:
    // the inverse of a transpose is the transpose of the inverse, so a
    // row-major cbuffer matrix comes back row-major. False if singular
    // (e.g. a zeroed matrix from a failed cbuffer read).
    bool InvertMatrix(const float in[16], float out[16]) {
      double m[16];
      for (uint32_t i = 0; i < 16; i++)
        m[i] = double(in[i]);

      double inv[16];
      inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
      inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
      inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
      inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
      inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
      inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
      inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
      inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
      inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
      inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
      inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
      inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
      inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
      inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
      inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
      inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

      double det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
      if (!std::isfinite(det) || std::abs(det) < 1.0e-30)
        return false;

      double invDet = 1.0 / det;
      for (uint32_t i = 0; i < 16; i++)
        out[i] = float(inv[i] * invDet);
      return true;
    }

    // blessed: hlsl mul(M, v) on the row-major bytes (M[r*4+c])
    void MulRowMajor(const float m[16], const double v[4], double out[4]) {
      for (uint32_t r = 0; r < 4; r++)
        out[r] = m[r*4+0]*v[0] + m[r*4+1]*v[1] + m[r*4+2]*v[2] + m[r*4+3]*v[3];
    }

  }


  BlessedSoftShadowState::BlessedSoftShadowState(DxvkDevice* device)
  : m_device(device) {
    // blessed: bilinear, clamp-to-edge -- see the header comment on
    // m_linearSampler
    DxvkSamplerKey samplerInfo = { };
    samplerInfo.setFilter(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST);
    samplerInfo.setAddressModes(
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    samplerInfo.setUsePixelCoordinates(false);
    m_linearSampler = device->createSampler(samplerInfo);

    static const std::array<DxvkDescriptorSetLayoutBinding, 3> traceBindings = {{
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
    }};

    m_traceLayout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
      VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedSoftShadowPush),
      uint32_t(traceBindings.size()), traceBindings.data());

    util::DxvkBuiltInShaderStage traceShader(blessed_soft_shadow_trace, nullptr);
    m_tracePipeline = device->createBuiltInComputePipeline(m_traceLayout, traceShader);

    static const std::array<DxvkDescriptorSetLayoutBinding, 2> filterBindings = {{
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
    }};

    m_filterLayout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
      VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedSoftShadowPush),
      uint32_t(filterBindings.size()), filterBindings.data());

    util::DxvkBuiltInShaderStage filterShader(blessed_soft_shadow_filter, nullptr);
    m_filterPipeline = device->createBuiltInComputePipeline(m_filterLayout, filterShader);

    m_reprojGame = env::getEnvVar("BLESSED_SHADOW_REPROJ") == "game";
    Logger::info(str::format("blessed: soft shadow: reprojection = ", m_reprojGame ? "game" : "own"));
  }


  void BlessedSoftShadowState::ensureSized(const Rc<DxvkCommandList>& cmd, uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height)
      return;

    DxvkImageCreateInfo info = { };
    info.type        = VK_IMAGE_TYPE_2D;
    info.format      = VK_FORMAT_R16G16B16A16_SFLOAT;
    info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    info.extent      = { width, height, 1u };
    info.numLayers   = 1u;
    info.mipLevels   = 1u;
    info.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    info.stages      = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    info.access      = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    info.tiling      = VK_IMAGE_TILING_OPTIMAL;
    info.layout      = VK_IMAGE_LAYOUT_GENERAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    DxvkImageViewKey sampledViewInfo = { };
    sampledViewInfo.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    sampledViewInfo.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
    sampledViewInfo.format     = VK_FORMAT_R16G16B16A16_SFLOAT;
    sampledViewInfo.layout     = VK_IMAGE_LAYOUT_GENERAL;
    sampledViewInfo.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    sampledViewInfo.mipIndex   = 0u;
    sampledViewInfo.mipCount   = 1u;
    sampledViewInfo.layerIndex = 0u;
    sampledViewInfo.layerCount = 1u;

    DxvkImageViewKey storageViewInfo = sampledViewInfo;
    storageViewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;

    for (uint32_t i = 0; i < 2; i++) {
      m_accumImage[i] = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      m_accumSampledView[i] = m_accumImage[i]->createView(sampledViewInfo);
      m_accumStorageView[i] = m_accumImage[i]->createView(storageViewInfo);

      // blessed: one-time UNDEFINED -> GENERAL transition. These images
      // are ours alone (never touched outside this class), so once they
      // land in GENERAL they stay there for their whole lifetime -- a
      // storage write and a sampled read of the same layout is exactly
      // what GENERAL is for.
      VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      barrier.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
      barrier.srcAccessMask       = 0;
      barrier.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barrier.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image               = m_accumImage[i]->handle();
      barrier.subresourceRange    = m_accumImage[i]->getAvailableSubresources();

      VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      dep.imageMemoryBarrierCount = 1u;
      dep.pImageMemoryBarriers    = &barrier;

      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
      m_accumImage[i]->trackLayout(m_accumImage[i]->getAvailableSubresources(), VK_IMAGE_LAYOUT_GENERAL);
      cmd->track(m_accumImage[i], DxvkAccess::Write);
    }

    m_width  = width;
    m_height = height;
    m_curr   = 0u;
    m_historyValid = false;
    m_havePrev = false;

    Logger::info(str::format("blessed: soft shadow: (re)allocated history at ", width, "x", height));
  }


  void BlessedSoftShadowState::dispatch(
          DxvkContext*                ctx,
    const Rc<DxvkCommandList>&        cmd,
    const BlessedShadowDispatchArgs&  args,
          uint32_t                    debugMode,
          uint64_t                    debugAddress) {
    ensureSized(cmd, args.width, args.height);

    uint32_t readIdx  = m_curr;
    uint32_t writeIdx = 1u - m_curr;

    DxvkBufferCreateInfo constantsInfo = { };
    constantsInfo.size      = sizeof(BlessedSoftShadowConstants);
    constantsInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    constantsInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    constantsInfo.access    = VK_ACCESS_2_SHADER_READ_BIT;
    constantsInfo.debugName = "blessed soft shadow constants";

    Rc<DxvkBuffer> constantsBuffer = m_device->createBuffer(constantsInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ctx->ensureBufferAddress(constantsBuffer);

    // blessed: the history in m_accumImage[readIdx] was written by the
    // previous dispatch, from its depth, its CameraViewProjInverse and its
    // CameraPosAdjust. Reprojecting with exactly those (inverted) is right
    // by construction, whatever the game means by "previous", however many
    // mask draws a frame has, and whether taa is on. The game's own
    // CameraPreviousViewProjUnjittered stays available for a/b.
    float ownPrevViewProj[16] = { };
    float ownDelta[3] = {
      args.camPosNow[0] - m_prevCamPos[0],
      args.camPosNow[1] - m_prevCamPos[1],
      args.camPosNow[2] - m_prevCamPos[2] };
    bool ownValid = m_havePrev && InvertMatrix(m_prevInvViewProj, ownPrevViewProj);

    BlessedSoftShadowConstants* k =
      reinterpret_cast<BlessedSoftShadowConstants*>(constantsBuffer->getSliceInfo().mapPtr);
    std::memcpy(k->invViewProj, args.invViewProj, sizeof(k->invViewProj));
    std::memcpy(k->camDelta, args.camDelta, sizeof(k->camDelta));
    std::memcpy(k->sunDir, args.sunDir, sizeof(k->sunDir));

    if (m_reprojGame) {
      std::memcpy(k->prevViewProj, args.prevViewProj, sizeof(k->prevViewProj));
      std::memcpy(k->reprojDelta, args.camReprojDelta, sizeof(k->reprojDelta));
    } else {
      std::memcpy(k->prevViewProj, ownPrevViewProj, sizeof(k->prevViewProj));
      std::memcpy(k->reprojDelta, ownDelta, sizeof(k->reprojDelta));
    }

    k->farDepthValue  = args.farDepthValue;
    k->sunHalfAngleRad = args.sunHalfAngleRad;
    k->spp            = args.spp;
    k->frameIndex     = m_frameIndex;
    k->historyValid   = (m_historyValid && (m_reprojGame || ownValid)) ? 1u : 0u;

    if (debugMode == 8u && ownValid && (m_frameIndex % 120u) == 0u)
      logReprojCompare(args, ownPrevViewProj, ownDelta);

    BlessedSoftShadowPush push = { };
    push.constantsAddress = constantsBuffer->getSliceInfo().gpuAddress;
    push.tlasAddress   = args.tlasAddress;
    push.debugAddress  = 0ull;
    push.width         = args.width;
    push.height        = args.height;
    push.samplerIndex  = m_linearSampler->getDescriptor().samplerIndex;
    push.tlasValid     = args.tlasValid ? 1u : 0u;
    push.debugMode     = debugMode;

    std::array<DxvkDescriptorWrite, 3> traceDescriptors = { };
    traceDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    traceDescriptors[0].descriptor     = args.depthView->getDescriptor();
    traceDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    traceDescriptors[1].descriptor     = m_accumSampledView[readIdx]->getDescriptor();
    traceDescriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    traceDescriptors[2].descriptor     = m_accumStorageView[writeIdx]->getDescriptor();

    cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
    cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_traceLayout,
      uint32_t(traceDescriptors.size()), traceDescriptors.data(), sizeof(push), &push);

    uint32_t groupsX = (args.width  + 7u) / 8u;
    uint32_t groupsY = (args.height + 7u) / 8u;
    cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, groupsX, groupsY, 1u);

    // blessed: the filter pass reads what the trace pass just wrote --
    // needs its writes visible before the next dispatch's reads, no
    // layout change (both stay in GENERAL, see ensureSized)
    VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1u;
    dep.pMemoryBarriers    = &barrier;
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

    push.constantsAddress = 0ull;
    push.tlasAddress      = 0ull;
    push.debugAddress     = debugAddress;

    std::array<DxvkDescriptorWrite, 2> filterDescriptors = { };
    filterDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    filterDescriptors[0].descriptor     = m_accumSampledView[writeIdx]->getDescriptor();
    filterDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    filterDescriptors[1].descriptor     = args.outputView->getDescriptor();

    cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_filterPipeline);
    cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_filterLayout,
      uint32_t(filterDescriptors.size()), filterDescriptors.data(), sizeof(push), &push);
    cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, groupsX, groupsY, 1u);

    cmd->track(m_accumImage[readIdx], DxvkAccess::Read);
    cmd->track(m_accumImage[writeIdx], DxvkAccess::Write);
    cmd->track(constantsBuffer, DxvkAccess::Read);

    m_curr = writeIdx;
    m_historyValid = true;
    m_frameIndex++;

    std::memcpy(m_prevInvViewProj, args.invViewProj, sizeof(m_prevInvViewProj));
    std::memcpy(m_prevCamPos, args.camPosNow, sizeof(m_prevCamPos));
    m_havePrev = true;
  }


  void BlessedSoftShadowState::logReprojCompare(
    const BlessedShadowDispatchArgs&  args,
    const float                       ownPrevViewProj[16],
    const float                       ownDelta[3]) const {
    // blessed: five probe points on this frame's screen, pushed far into the
    // scene (ndc z near the far value), reconstructed with this frame's
    // matrix, then projected into last frame's screen twice: with our own
    // previous matrix and with the game's. "motion" is how far the point
    // moved on screen since last dispatch (the camera turn, in pixels);
    // "game-vs-own" is how far the game's matrix puts it from ours. With a
    // still camera both are ~0; if game-vs-own tracks motion while turning,
    // the game's matrix is not last dispatch's camera.
    static const double probes[5][2] = {
      { 0.0, 0.0 }, { -0.6, -0.6 }, { 0.6, -0.6 }, { -0.6, 0.6 }, { 0.6, 0.6 } };

    double z = args.farDepthValue > 0.5f ? 0.999 : 0.001;
    double w = double(args.width), h = double(args.height);

    double maxMotion = 0.0, maxGameVsOwn = 0.0;
    bool gameBehind = false;

    for (uint32_t i = 0; i < 5; i++) {
      double ndc[4] = { probes[i][0], probes[i][1], z, 1.0 };
      double p[4];
      MulRowMajor(args.invViewProj, ndc, p);
      if (std::abs(p[3]) < 1.0e-12)
        continue;

      double own[4] = { p[0] / p[3] + ownDelta[0], p[1] / p[3] + ownDelta[1], p[2] / p[3] + ownDelta[2], 1.0 };
      double game[4] = { p[0] / p[3] + args.camReprojDelta[0], p[1] / p[3] + args.camReprojDelta[1], p[2] / p[3] + args.camReprojDelta[2], 1.0 };

      double co[4], cg[4];
      MulRowMajor(ownPrevViewProj, own, co);
      MulRowMajor(args.prevViewProj, game, cg);
      if (co[3] <= 0.0)
        continue;
      if (cg[3] <= 0.0) {
        gameBehind = true;
        continue;
      }

      double ox = (co[0] / co[3] - ndc[0]) * 0.5 * w, oy = (co[1] / co[3] - ndc[1]) * 0.5 * h;
      double gx = (cg[0] / cg[3] - co[0] / co[3]) * 0.5 * w, gy = (cg[1] / cg[3] - co[1] / co[3]) * 0.5 * h;
      maxMotion    = std::max(maxMotion,    std::sqrt(ox * ox + oy * oy));
      maxGameVsOwn = std::max(maxGameVsOwn, std::sqrt(gx * gx + gy * gy));
    }

    Logger::info(str::format("blessed: soft shadow reproj[", m_frameIndex, "]: motion px=", maxMotion,
      " game-vs-own px=", maxGameVsOwn, gameBehind ? " (game matrix puts a probe behind the camera)" : "",
      " delta own=(", ownDelta[0], ",", ownDelta[1], ",", ownDelta[2],
      ") app=(", args.camReprojDelta[0], ",", args.camReprojDelta[1], ",", args.camReprojDelta[2], ")"));
  }

}
