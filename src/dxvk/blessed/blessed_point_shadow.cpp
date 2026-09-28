// blessed: ray-traced point-light shadow pass (tier a) -- dxvk-side compute dispatch. See blessed_point_shadow.h.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "blessed_point_shadow.h"
#include "blessed_scene.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"
#include "../dxvk_device.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/util_time.h"
#include "../../util/log/log.h"

#include <blessed_point_shadow.h>
#include <blessed_point_soft_trace.h>
#include <blessed_point_soft_filter.h>

namespace dxvk {

  namespace {

    // blessed: must match BlessedPointShadowPushData in
    // shaders/blessed_point_shadow.comp (scalar layout, 128 bytes)
    struct BlessedPointShadowPushData {
      float    invViewProj[16];
      uint64_t tlasAddress;
      uint64_t debugAddress;
      float    camDelta[3];
      float    lightPos[3];
      float    radius;
      float    proxyRadius;
      float    farDepthValue;
      uint32_t extent;
      uint32_t samplerIndex;
      uint32_t flags;
    };

    static_assert(sizeof(BlessedPointShadowPushData) == 128,
      "must match the 128-byte layout blessed_point_shadow.comp expects");

    // 4 lights a frame at most, double-buffered: 8 pairs
    constexpr uint32_t PointTimestampPairs = 8u;

    // blessed: same barrier pair as the sun pass (blessed_shadow.cpp), kept
    // local so this file stands alone -- see the caveat there about images
    // without unified layouts.
    VkImageLayout PointTransition(
      const Rc<DxvkCommandList>&   cmd,
      const Rc<DxvkImageView>&     view,
            VkPipelineStageFlags2  srcStage,
            VkAccessFlags2         srcAccess,
            VkPipelineStageFlags2  dstStage,
            VkAccessFlags2         dstAccess,
            bool                   toCompute,
            VkImageLayout          restoreLayout) {
      DxvkImage* image = view->image();
      bool unified = image->hasUnifiedLayout();
      VkImageLayout general = image->pickLayout(VK_IMAGE_LAYOUT_GENERAL);
      // blessed: state-audit -- the tracked layout, not the default one: a
      // pass suspended by endCurrentPass leaves its attachments in their
      // attachment layout. Unified-layout images never get here.
      VkImageLayout tracked = unified ? general : image->queryLayout(image->getAvailableSubresources());

      if (tracked == VK_IMAGE_LAYOUT_MAX_ENUM)
        tracked = image->info().layout;

      VkImageLayout oldLayout = toCompute ? tracked : general;
      VkImageLayout newLayout = toCompute ? general : restoreLayout;

      VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      barrier.srcStageMask        = srcStage;
      barrier.srcAccessMask       = srcAccess;
      barrier.dstStageMask        = dstStage;
      barrier.dstAccessMask       = dstAccess;
      barrier.oldLayout           = oldLayout;
      barrier.newLayout           = newLayout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image               = image->handle();
      barrier.subresourceRange    = image->getAvailableSubresources();

      VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      dep.imageMemoryBarrierCount = 1u;
      dep.pImageMemoryBarriers    = &barrier;

      // blessed: raw barrier on an image dxvk still owns (the game's mask / depth)
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

      if (!unified)
        image->trackLayout(image->getAvailableSubresources(), newLayout);

      return oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? image->info().layout : oldLayout;
    }


    // blessed: BLESSED_POINT_SOFT -- byte-for-byte BlessedPointSoftConstants
    // in blessed_point_soft_{trace,filter}.comp
    struct BlessedPointSoftConstants {
      float    invViewProj[16];
      float    prevViewProj[16];
      float    camDelta[3];
      float    lightPos[3];
      float    reprojDelta[3];
      float    radius;
      float    proxyRadius;
      float    lightRadius;
      float    farDepthValue;
      uint32_t spp;
      uint32_t frameIndex;
      uint32_t channel;
      uint32_t historyValid;
      uint32_t width;
      uint32_t height;
      uint32_t tlasValid;
      uint32_t debugMode;
    };

    struct BlessedPointSoftPush {
      uint64_t constantsAddress;
      uint64_t tlasAddress;
      uint64_t debugAddress;
      uint32_t samplerIndex;
      uint32_t channelMask;
    };

    static_assert(sizeof(BlessedPointSoftPush) == 32, "must match BlessedPointSoftPush in the .comp files");

    // blessed: a light that moved more than this (units, after the camera
    // shift) since last frame restarts its channel's history -- the engine
    // may hand a channel to a different light
    constexpr float PointHistoryJump = 16.0f;

    // blessed: general 4x4 inverse in double, layout-agnostic (a copy of
    // blessed_soft_shadow.cpp's, kept local so neither file depends on the other)
    bool PointInvert4(const float in[16], float out[16]) {
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

      for (uint32_t i = 0; i < 16; i++)
        out[i] = float(inv[i] / det);
      return true;
    }


    /**
     * blessed: soft point shadows, one history for all four channels.
     * Two ping-pong pairs, rgba16f, resident in GENERAL: vis (one component
     * per mask channel) and pos (the frame's camera-relative surfaces / 64,
     * written identically by every light). The pairs flip once per app
     * frame, so every light of a frame reads the same history. A channel's
     * history counts only if that channel was traced last frame, by a light
     * in about the same place. Reprojection is the sun's "own" path
     * (shadow-slide): the inverse of last frame's CameraViewProjInverse and
     * last frame's CameraPosAdjust, then the same 3D rejection.
     */
    class BlessedPointSoftState {
    public:

      explicit BlessedPointSoftState(DxvkDevice* device)
      : m_device(device) {
        DxvkSamplerKey samplerInfo = { };
        samplerInfo.setFilter(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST);
        samplerInfo.setAddressModes(
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        samplerInfo.setUsePixelCoordinates(false);
        m_sampler = device->createSampler(samplerInfo);

        static const std::array<DxvkDescriptorSetLayoutBinding, 5> traceBindings = {{
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        }};

        m_traceLayout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedPointSoftPush),
          uint32_t(traceBindings.size()), traceBindings.data());

        util::DxvkBuiltInShaderStage traceShader(blessed_point_soft_trace, nullptr);
        m_tracePipeline = device->createBuiltInComputePipeline(m_traceLayout, traceShader);

        static const std::array<DxvkDescriptorSetLayoutBinding, 3> filterBindings = {{
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        }};

        m_filterLayout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedPointSoftPush),
          uint32_t(filterBindings.size()), filterBindings.data());

        util::DxvkBuiltInShaderStage filterShader(blessed_point_soft_filter, nullptr);
        m_filterPipeline = device->createBuiltInComputePipeline(m_filterLayout, filterShader);
      }

      void dispatch(
              DxvkContext*                      ctx,
        const Rc<DxvkCommandList>&              cmd,
        const BlessedPointShadowDispatchArgs&   args,
              uint64_t                          debugAddress) {
        ensureSized(cmd, args.width, args.height);

        // blessed: a new app frame flips the pairs; what the last frame
        // wrote becomes the history every light of this frame reads
        if (args.frameId != m_frameId) {
          bool consecutive = m_frameId != 0u && args.frameId == m_frameId + 1u;
          m_writtenLast = consecutive ? m_writtenThis : 0u;
          std::memcpy(m_lightLast, m_lightThis, sizeof(m_lightLast));
          m_havePrev = consecutive && m_haveFrameCam;
          std::memcpy(m_prevInvViewProj, m_frameInvViewProj, sizeof(m_prevInvViewProj));
          std::memcpy(m_prevCamPos, m_frameCamPos, sizeof(m_prevCamPos));
          m_writtenThis  = 0u;
          m_haveFrameCam = false;
          m_curr         = 1u - m_curr;
          m_frameId      = args.frameId;
          m_frameIndex++;
        }

        if (!m_haveFrameCam) {
          std::memcpy(m_frameInvViewProj, args.invViewProj, sizeof(m_frameInvViewProj));
          std::memcpy(m_frameCamPos, args.camPosNow, sizeof(m_frameCamPos));
          m_haveFrameCam = true;
        }

        uint32_t channel = 0u;
        while (channel < 3u && !(args.channelMask & (1u << channel)))
          channel++;

        float delta[3] = {
          args.camPosNow[0] - m_prevCamPos[0],
          args.camPosNow[1] - m_prevCamPos[1],
          args.camPosNow[2] - m_prevCamPos[2] };

        float prevViewProj[16] = { };
        bool valid = m_havePrev && (m_writtenLast & (1u << channel))
          && PointInvert4(m_prevInvViewProj, prevViewProj);

        if (valid) {
          float dx = args.lightPos[0] + delta[0] - m_lightLast[channel][0];
          float dy = args.lightPos[1] + delta[1] - m_lightLast[channel][1];
          float dz = args.lightPos[2] + delta[2] - m_lightLast[channel][2];
          valid = dx * dx + dy * dy + dz * dz <= PointHistoryJump * PointHistoryJump;
        }

        m_writtenThis |= 1u << channel;
        std::memcpy(m_lightThis[channel], args.lightPos, sizeof(m_lightThis[channel]));

        DxvkBufferCreateInfo constantsInfo = { };
        constantsInfo.size      = sizeof(BlessedPointSoftConstants);
        constantsInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        constantsInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constantsInfo.access    = VK_ACCESS_2_SHADER_READ_BIT;
        constantsInfo.debugName = "blessed point soft constants";

        Rc<DxvkBuffer> constants = m_device->createBuffer(constantsInfo,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ctx->ensureBufferAddress(constants);

        auto* k = reinterpret_cast<BlessedPointSoftConstants*>(constants->getSliceInfo().mapPtr);
        std::memcpy(k->invViewProj, args.invViewProj, sizeof(k->invViewProj));
        std::memcpy(k->prevViewProj, prevViewProj, sizeof(k->prevViewProj));
        std::memcpy(k->camDelta, args.camDelta, sizeof(k->camDelta));
        std::memcpy(k->lightPos, args.lightPos, sizeof(k->lightPos));
        std::memcpy(k->reprojDelta, delta, sizeof(k->reprojDelta));
        k->radius        = args.radius;
        k->proxyRadius   = args.proxyRadius;
        k->lightRadius   = args.lightRadius;
        k->farDepthValue = args.farDepthValue;
        k->spp           = args.spp;
        k->frameIndex    = m_frameIndex;
        k->channel       = channel;
        k->historyValid  = valid ? 1u : 0u;
        k->width         = args.width;
        k->height        = args.height;
        k->tlasValid     = args.tlasValid ? 1u : 0u;
        k->debugMode     = args.debugMode;

        uint32_t readIdx  = 1u - m_curr;
        uint32_t writeIdx = m_curr;

        // blessed: the previous light's trace/filter wrote these images
        ComputeBarrier(cmd);

        BlessedPointSoftPush push = { };
        push.constantsAddress = constants->getSliceInfo().gpuAddress;
        push.tlasAddress      = args.tlasAddress;
        push.debugAddress     = 0ull;
        push.samplerIndex     = m_sampler->getDescriptor().samplerIndex;
        push.channelMask      = args.channelMask;

        std::array<DxvkDescriptorWrite, 5> traceDescriptors = { };
        traceDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        traceDescriptors[0].descriptor     = args.depthView->getDescriptor();
        traceDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        traceDescriptors[1].descriptor     = m_visSampled[readIdx]->getDescriptor();
        traceDescriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        traceDescriptors[2].descriptor     = m_posSampled[readIdx]->getDescriptor();
        traceDescriptors[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        traceDescriptors[3].descriptor     = m_visStorage[writeIdx]->getDescriptor();
        traceDescriptors[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        traceDescriptors[4].descriptor     = m_posStorage[writeIdx]->getDescriptor();

        uint32_t groupsX = (args.width  + 7u) / 8u;
        uint32_t groupsY = (args.height + 7u) / 8u;

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_traceLayout,
          uint32_t(traceDescriptors.size()), traceDescriptors.data(), sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, groupsX, groupsY, 1u);

        ComputeBarrier(cmd);

        push.tlasAddress  = 0ull;
        push.debugAddress = debugAddress;

        std::array<DxvkDescriptorWrite, 3> filterDescriptors = { };
        filterDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        filterDescriptors[0].descriptor     = m_visSampled[writeIdx]->getDescriptor();
        filterDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        filterDescriptors[1].descriptor     = m_posSampled[writeIdx]->getDescriptor();
        filterDescriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        filterDescriptors[2].descriptor     = args.outputView->getDescriptor();

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_filterPipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_filterLayout,
          uint32_t(filterDescriptors.size()), filterDescriptors.data(), sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, groupsX, groupsY, 1u);

        for (uint32_t i = 0; i < 2; i++) {
          cmd->track(m_vis[i], DxvkAccess::Write);
          cmd->track(m_pos[i], DxvkAccess::Write);
        }
        cmd->track(constants, DxvkAccess::Read);
      }

    private:

      DxvkDevice*               m_device;
      Rc<DxvkSampler>           m_sampler;
      const DxvkPipelineLayout* m_traceLayout    = nullptr;
      VkPipeline                m_tracePipeline  = VK_NULL_HANDLE;
      const DxvkPipelineLayout* m_filterLayout   = nullptr;
      VkPipeline                m_filterPipeline = VK_NULL_HANDLE;

      uint32_t m_width  = 0u;
      uint32_t m_height = 0u;

      std::array<Rc<DxvkImage>, 2>     m_vis;
      std::array<Rc<DxvkImage>, 2>     m_pos;
      std::array<Rc<DxvkImageView>, 2> m_visSampled, m_visStorage;
      std::array<Rc<DxvkImageView>, 2> m_posSampled, m_posStorage;
      uint32_t m_curr = 0u;    // this frame's write slot; 1 - m_curr is the history

      uint64_t m_frameId    = 0u;
      uint32_t m_frameIndex = 0u;

      bool     m_haveFrameCam = false;
      float    m_frameInvViewProj[16] = { };
      float    m_frameCamPos[3] = { };

      bool     m_havePrev = false;
      float    m_prevInvViewProj[16] = { };
      float    m_prevCamPos[3] = { };

      uint32_t m_writtenThis = 0u;
      uint32_t m_writtenLast = 0u;
      float    m_lightThis[4][3] = { };
      float    m_lightLast[4][3] = { };

      static void ComputeBarrier(const Rc<DxvkCommandList>& cmd) {
        VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &barrier;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
      }

      void initLayout(const Rc<DxvkCommandList>& cmd, const Rc<DxvkImage>& image) {
        // blessed: one-time UNDEFINED -> GENERAL; ours alone, it stays there
        VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image->handle();
        barrier.subresourceRange    = image->getAvailableSubresources();

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1u;
        dep.pImageMemoryBarriers    = &barrier;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

        image->trackLayout(image->getAvailableSubresources(), VK_IMAGE_LAYOUT_GENERAL);
        cmd->track(image, DxvkAccess::Write);
      }

      void ensureSized(const Rc<DxvkCommandList>& cmd, uint32_t width, uint32_t height) {
        if (width == m_width && height == m_height)
          return;

        DxvkImageCreateInfo info = { };
        info.type          = VK_IMAGE_TYPE_2D;
        info.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
        info.sampleCount   = VK_SAMPLE_COUNT_1_BIT;
        info.extent        = { width, height, 1u };
        info.numLayers     = 1u;
        info.mipLevels     = 1u;
        info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        info.stages        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        info.access        = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        info.layout        = VK_IMAGE_LAYOUT_GENERAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        DxvkImageViewKey sampledKey = { };
        sampledKey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
        sampledKey.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
        sampledKey.format     = VK_FORMAT_R16G16B16A16_SFLOAT;
        sampledKey.layout     = VK_IMAGE_LAYOUT_GENERAL;
        sampledKey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
        sampledKey.mipIndex   = 0u;
        sampledKey.mipCount   = 1u;
        sampledKey.layerIndex = 0u;
        sampledKey.layerCount = 1u;

        DxvkImageViewKey storageKey = sampledKey;
        storageKey.usage = VK_IMAGE_USAGE_STORAGE_BIT;

        for (uint32_t i = 0; i < 2; i++) {
          m_vis[i] = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
          m_pos[i] = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
          m_visSampled[i] = m_vis[i]->createView(sampledKey);
          m_visStorage[i] = m_vis[i]->createView(storageKey);
          m_posSampled[i] = m_pos[i]->createView(sampledKey);
          m_posStorage[i] = m_pos[i]->createView(storageKey);
          initLayout(cmd, m_vis[i]);
          initLayout(cmd, m_pos[i]);
        }

        m_width       = width;
        m_height      = height;
        m_frameId     = 0u;
        m_havePrev    = false;
        m_writtenThis = 0u;
        m_writtenLast = 0u;

        Logger::info(str::format("blessed: point soft shadows: history at ", width, "x", height,
          " (4 x rgba16f, ", (uint64_t(width) * height * 8u * 4u) >> 20, " MiB)"));
      }
    };


    class BlessedPointShadowObjects {
    public:

      explicit BlessedPointShadowObjects(DxvkDevice* device)
      : m_device(device) {
        DxvkSamplerKey samplerInfo = { };
        samplerInfo.setFilter(VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST);
        samplerInfo.setAddressModes(
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        samplerInfo.setUsePixelCoordinates(false);
        m_sampler = device->createSampler(samplerInfo);

        static const std::array<DxvkDescriptorSetLayoutBinding, 2> bindings = {{
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        }};

        m_layout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedPointShadowPushData),
          uint32_t(bindings.size()), bindings.data());

        util::DxvkBuiltInShaderStage shader(blessed_point_shadow, nullptr);
        m_pipeline = device->createBuiltInComputePipeline(m_layout, shader);

        if (device->properties().core.properties.limits.timestampComputeAndGraphics) {
          VkQueryPoolCreateInfo queryInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
          queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
          queryInfo.queryCount = PointTimestampPairs * 2u;

          if (device->vkd()->vkCreateQueryPool(device->handle(), &queryInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
            Logger::warn("blessed: point shadows: no timestamp query pool, timing disabled");
            m_queryPool = VK_NULL_HANDLE;
          }
        }
      }

      void dispatch(
              DxvkContext*                      ctx,
        const Rc<DxvkCommandList>&              cmd,
        const BlessedPointShadowDispatchArgs&   args) {
        if (args.width == 0u || args.height == 0u || args.width > 0xFFFFu || args.height > 0xFFFFu)
          return;

        // blessed: a new app-thread frame closes the previous one's timing
        if (args.frameId != m_lastFrameId) {
          if (m_lastFrameId != 0u)
            endFrame(args.frameId - m_lastFrameId);
          m_lastFrameId = args.frameId;
        }

        Rc<DxvkBuffer> debugBuffer;

        if (args.dump) {
          DxvkBufferCreateInfo info = { };
          info.size      = 16ull + VkDeviceSize(args.width) * args.height * sizeof(uint32_t);
          info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
          info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          info.access    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
          info.debugName = "blessed point shadow debug";

          debugBuffer = m_device->createBuffer(info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
          std::memset(debugBuffer->getSliceInfo().mapPtr, 0, size_t(info.size));
          ctx->ensureBufferAddress(debugBuffer);
        }

        VkImageLayout depthRestore = PointTransition(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
          true, VK_IMAGE_LAYOUT_UNDEFINED);

        VkImageLayout outputRestore = PointTransition(cmd, args.outputView,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          true, VK_IMAGE_LAYOUT_UNDEFINED);

        BlessedPointShadowPushData push = { };
        std::memcpy(push.invViewProj, args.invViewProj, sizeof(push.invViewProj));
        push.tlasAddress   = args.tlasAddress;
        push.debugAddress  = debugBuffer != nullptr ? debugBuffer->getSliceInfo().gpuAddress : 0ull;
        std::memcpy(push.camDelta, args.camDelta, sizeof(push.camDelta));
        std::memcpy(push.lightPos, args.lightPos, sizeof(push.lightPos));
        push.radius        = args.radius;
        push.proxyRadius   = args.proxyRadius;
        push.farDepthValue = args.farDepthValue;
        push.extent        = args.width | (args.height << 16);
        push.samplerIndex  = m_sampler->getDescriptor().samplerIndex;
        push.flags         = (args.channelMask & 0xFu)
                           | (args.tlasValid ? 0x10u : 0u)
                           | ((args.debugMode & 0xFFu) << 8);

        uint32_t queryBase = beginTiming(cmd);

        // blessed: the soft path serves the real output and its own hist
        // view; every other debug view stays on the hard path (one ray, no
        // history)
        if (args.softEnabled && (args.debugMode == 0u || args.debugMode == 7u)) {
          if (m_soft == nullptr)
            m_soft = std::make_unique<BlessedPointSoftState>(m_device);

          m_soft->dispatch(ctx, cmd, args, push.debugAddress);
        } else {
          std::array<DxvkDescriptorWrite, 2> descriptors = { };
          descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
          descriptors[0].descriptor     = args.depthView->getDescriptor();
          descriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
          descriptors[1].descriptor     = args.outputView->getDescriptor();

          cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
          cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_layout,
            uint32_t(descriptors.size()), descriptors.data(), sizeof(push), &push);
          cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (args.width + 7u) / 8u, (args.height + 7u) / 8u, 1u);
        }

        endTiming(cmd, queryBase);

        if (args.dump)
          blockingReadback(ctx, cmd, args, debugBuffer);

        PointTransition(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
          false, depthRestore);

        // blessed: the next consumer may be another light's dispatch (compute)
        // or the lit pass reading the mask (fragment)
        PointTransition(cmd, args.outputView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          false, outputRestore);

        cmd->track(args.depthView->image(), DxvkAccess::Read);
        cmd->track(args.outputView->image(), DxvkAccess::Write);

        if (debugBuffer != nullptr)
          cmd->track(debugBuffer, DxvkAccess::Read);

        if (args.tlasRef != nullptr)
          cmd->track(args.tlasRef);

        m_dispatchesInWindow++;
        if (!args.tlasValid)
          m_invalidTlasInWindow++;
      }

    private:

      // blessed: one line per 120 presents into pointshadow-gpu.jsonl,
      // counted in presents that had at least one light (frames with none
      // are skipped over by \p frames and count toward the window)
      void endFrame(uint64_t frames) {
        m_framesInWindow += uint32_t(std::min<uint64_t>(frames, 120u));
        if (m_framesInWindow < 120u)
          return;

        if (!m_fileTried) {
          m_fileTried = true;
          std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
          if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            m_file.open(dir + env::PlatformDirSlash + "pointshadow-gpu.jsonl", std::ios::out | std::ios::app);
          }
          m_start = dxvk::high_resolution_clock::now();
        }

        if (m_file.is_open()) {
          double t = std::chrono::duration<double>(dxvk::high_resolution_clock::now() - m_start).count();
          double perLight = m_gpuSamples ? m_gpuMsSum / double(m_gpuSamples) : 0.0;
          double perFrame = perLight * double(m_dispatchesInWindow) / double(m_framesInWindow);

          m_file << str::format("{\"t\":", t,
            ",\"frames\":", m_framesInWindow,
            ",\"lights_per_frame\":", double(m_dispatchesInWindow) / double(m_framesInWindow),
            ",\"gpu_ms_per_light\":", perLight,
            ",\"gpu_ms_per_frame\":", perFrame,
            ",\"invalid_tlas\":", m_invalidTlasInWindow, "}\n");
          m_file.flush();
        }

        m_framesInWindow      = 0u;
        m_dispatchesInWindow  = 0u;
        m_invalidTlasInWindow = 0u;
        m_gpuMsSum            = 0.0;
        m_gpuSamples          = 0u;
      }

      DxvkDevice*               m_device;
      Rc<DxvkSampler>           m_sampler;
      const DxvkPipelineLayout* m_layout   = nullptr;
      VkPipeline                m_pipeline = VK_NULL_HANDLE;

      // blessed: BLESSED_POINT_SOFT=1, built on the first soft dispatch
      std::unique_ptr<BlessedPointSoftState> m_soft;

      VkQueryPool m_queryPool = VK_NULL_HANDLE;
      uint32_t    m_queryNext = 0u;
      bool        m_queryUsed[PointTimestampPairs] = { };

      uint64_t m_lastFrameId         = 0u;
      uint32_t m_framesInWindow      = 0u;
      uint32_t m_dispatchesInWindow  = 0u;
      uint32_t m_invalidTlasInWindow = 0u;
      double   m_gpuMsSum            = 0.0;
      uint32_t m_gpuSamples          = 0u;

      std::ofstream m_file;
      bool          m_fileTried = false;
      dxvk::high_resolution_clock::time_point m_start;

      uint32_t beginTiming(const Rc<DxvkCommandList>& cmd) {
        if (m_queryPool == VK_NULL_HANDLE)
          return UINT32_MAX;

        uint32_t pair = m_queryNext;
        m_queryNext = (m_queryNext + 1u) % PointTimestampPairs;

        if (m_queryUsed[pair])
          collectTiming(pair * 2u);

        cmd->cmdResetQueryPool(DxvkCmdBuffer::ExecBuffer, m_queryPool, pair * 2u, 2u);
        cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer,
          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, pair * 2u);

        m_queryUsed[pair] = true;
        return pair * 2u;
      }

      void endTiming(const Rc<DxvkCommandList>& cmd, uint32_t base) {
        if (base == UINT32_MAX)
          return;

        cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer,
          VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_queryPool, base + 1u);
      }

      void collectTiming(uint32_t base) {
        struct { uint64_t ts; uint64_t avail; } results[2] = { };

        VkResult vr = m_device->vkd()->vkGetQueryPoolResults(
          m_device->handle(), m_queryPool, base, 2u,
          sizeof(results), results, sizeof(results[0]),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

        if (vr != VK_SUCCESS || !results[0].avail || !results[1].avail)
          return;

        double period = double(m_device->properties().core.properties.limits.timestampPeriod);
        m_gpuMsSum += double(results[1].ts - results[0].ts) * period / 1.0e6;
        m_gpuSamples++;
      }

      // blessed: BLESSED_POINT_DUMP -- debug only, blocking, one dispatch
      void blockingReadback(
              DxvkContext*                      ctx,
        const Rc<DxvkCommandList>&              cmd,
        const BlessedPointShadowDispatchArgs&   args,
        const Rc<DxvkBuffer>&                   debugBuffer) {
        VkMemoryBarrier2 hostBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        hostBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        hostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        hostBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &hostBarrier;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

        Rc<DxvkCommandList> finished = ctx->endRecording(nullptr);

        DxvkSubmitStatus status = { };
        m_device->submitCommandList(finished, nullptr, 0u, &status);
        m_device->waitForSubmission(&status);
        m_device->waitForIdle();

        ctx->beginRecording(m_device->createCommandList());

        const uint8_t*  base     = reinterpret_cast<const uint8_t*>(debugBuffer->getSliceInfo().mapPtr);
        const uint32_t* counters = reinterpret_cast<const uint32_t*>(base);
        const uint32_t* pixels   = reinterpret_cast<const uint32_t*>(base + 16);

        Logger::info(str::format("blessed: pointshadow dump: ", args.width, "x", args.height,
          " channels=", args.channelMask, " light=(", args.lightPos[0], ",", args.lightPos[1], ",", args.lightPos[2],
          ") radius=", args.radius, " tlas_valid=", args.tlasValid ? 1 : 0,
          " culled=", counters[0], " lit=", counters[1], " shadowed=", counters[2]));

        std::error_code ec;
        std::filesystem::create_directories(args.dumpPath.substr(0, args.dumpPath.find_last_of("/\\")), ec);

        std::ofstream pgm(args.dumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!pgm.is_open()) {
          Logger::warn(str::format("blessed: pointshadow dump: could not open '", args.dumpPath, "'"));
          return;
        }

        pgm << "P5\n" << args.width << " " << args.height << "\n255\n";

        std::vector<uint8_t> row(args.width);
        for (uint32_t y = 0; y < args.height; y++) {
          for (uint32_t x = 0; x < args.width; x++)
            row[x] = uint8_t(std::min<uint32_t>(pixels[y * args.width + x], 255u));
          pgm.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size()));
        }
      }
    };

    BlessedPointShadowObjects* PointInstance(DxvkDevice* device) {
      static BlessedPointShadowObjects s_instance(device);
      return &s_instance;
    }

  }


  void DxvkContext::blessedRunPointShadowPass(
    const BlessedPointShadowDispatchArgs& argsIn) {
    if (argsIn.outputView == nullptr || argsIn.depthView == nullptr || !m_device->supportsRayQuery())
      return;

    // blessed: tlas and camera delta resolved here, at dispatch time, for
    // the reason DxvkContext::blessedRunShadowPass gives
    BlessedPointShadowDispatchArgs args = argsIn;

    BlessedSceneFrame scene = { };
    if (BlessedScene* blessedScene = m_device->blessedScene())
      scene = blessedScene->currentFrame();

    args.tlasAddress = scene.tlasAddress;
    args.tlasValid   = scene.valid;
    args.tlasRef     = scene.tlas;

    static const bool s_noCamShift = env::getEnvVar("BLESSED_SHADOW_CAMPOS") == "off";
    for (uint32_t i = 0; i < 3; i++)
      args.camDelta[i] = s_noCamShift ? 0.0f : args.camPosNow[i] - scene.camPos[i];

    // blessed: indoors the sun pass may never run, so the point pass tags
    // the tlas with this frame's camera too (idempotent within a frame)
    if (BlessedScene* taggedScene = m_device->blessedScene())
      taggedScene->noteCamera(args.camPosNow);

    this->endCurrentPass(true);

    // blessed: state-audit -- the game may have cleared these images just
    // before (the sun mask gets a clear_rtv right before its draw, frame
    // dump i=8109). dxvk defers clears into the next render pass; with that
    // draw skipped, none begins, so the clear would land after our raw
    // writes and wipe them. Run it now, and flush dxvk's batched barriers
    // (e.g. a copy's release) ahead of ours, as the volumetrics pass does.
    this->flushDeferredClear(*args.outputView->image(), args.outputView->image()->getAvailableSubresources());
    this->flushDeferredClear(*args.depthView->image(), args.depthView->image()->getAvailableSubresources());
    this->flushBarriers();

    PointInstance(m_device.ptr())->dispatch(this, m_cmd, args);

    // blessed: state-audit -- the pass bound its own compute pipeline,
    // descriptors and push data raw; dxvk still thinks its last pipeline is
    // bound and would skip the rebind for a same-shader dispatch. Same
    // contract as upstream's meta ops.
    this->invalidateState();
  }

}
