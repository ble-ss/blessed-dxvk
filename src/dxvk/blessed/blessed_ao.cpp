// blessed: ray-traced ambient occlusion -- dxvk-side pipelines, history images and the dispatch. See blessed_ao.h.
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>

#include "blessed_ao.h"
#include "blessed_scene.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"
#include "../dxvk_device.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/util_time.h"
#include "../../util/log/log.h"

#include <blessed_ao_trace.h>
#include <blessed_ao_blur.h>
#include <blessed_ao_resolve.h>

namespace dxvk {

  namespace {

    // blessed: matches BlessedAoConstants in blessed_ao_common.glsl byte
    // for byte (scalar layout)
    struct BlessedAoConstants {
      float    invViewProj[16];
      float    prevViewProj[16];
      float    camDelta[3];
      float    reprojDelta[3];
      float    farDepthValue;
      float    radius;
      float    strength;
      uint32_t rays;
      uint32_t frameIndex;
      uint32_t historyValid;
      uint32_t fullWidth;
      uint32_t fullHeight;
      uint32_t traceWidth;
      uint32_t traceHeight;
      uint32_t traceScale;
      uint32_t debugMode;
    };

    // blessed: matches BlessedAoPush in blessed_ao_common.glsl
    struct BlessedAoPush {
      uint64_t constantsAddress;
      uint64_t tlasAddress;
      uint32_t samplerIndex;
      uint32_t tlasValid;
      uint32_t passArg;
    };

    static_assert(sizeof(BlessedAoPush) <= 128,
      "must fit the Vulkan-guaranteed minimum push constant budget");

    constexpr uint32_t TimestampQueryCount = 4; // 2 slots, double-buffered

    // blessed: general 4x4 inverse (cofactors, in double), same as the
    // shadow-slide fix's. The inverse of a transpose is the transpose of
    // the inverse, so row-major bytes come back row-major. False if
    // singular (e.g. a zeroed matrix from a failed cbuffer read).
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

    // blessed: raw barrier on an image dxvk owns (the game's depth or sao
    // texture), same model as the shadow pass's TransitionForCompute: on
    // unified-layout hardware a pure sync barrier, otherwise GENERAL and
    // back to the image's resting layout
    VkImageLayout Transition(
      const Rc<DxvkCommandList>&   cmd,
            DxvkImage*             image,
            VkPipelineStageFlags2  srcStage,
            VkAccessFlags2         srcAccess,
            VkPipelineStageFlags2  dstStage,
            VkAccessFlags2         dstAccess,
            bool                   toCompute,
            VkImageLayout          restoreLayout) {
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

      // blessed: raw
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

      if (!unified)
        image->trackLayout(image->getAvailableSubresources(), newLayout);

      return oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? image->info().layout : oldLayout;
    }

    void ComputeBarrier(const Rc<DxvkCommandList>& cmd) {
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


    // blessed: one of our own trace-resolution images, SAMPLED | STORAGE,
    // resident in GENERAL for its whole life
    struct BlessedAoImage {
      Rc<DxvkImage>     image;
      Rc<DxvkImageView> sampled;
      Rc<DxvkImageView> storage;
    };


    /**
     * \brief Lazily built pipelines, history images and timing for the ao pass
     *
     * One process-lifetime instance, same model as the shadow pass's
     * BlessedShadowObjects.
     */
    class BlessedAoObjects {
    public:

      explicit BlessedAoObjects(DxvkDevice* device)
      : m_device(device) {
        DxvkSamplerKey samplerInfo = { };
        samplerInfo.setFilter(VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST);
        samplerInfo.setAddressModes(
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        samplerInfo.setUsePixelCoordinates(false);
        m_sampler = device->createSampler(samplerInfo);

        m_traceLayout = makeLayout({ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_MAX_ENUM });
        m_blurLayout = makeLayout({ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_MAX_ENUM });
        m_resolveLayout = makeLayout({ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE });

        util::DxvkBuiltInShaderStage traceShader(blessed_ao_trace, nullptr);
        util::DxvkBuiltInShaderStage blurShader(blessed_ao_blur, nullptr);
        util::DxvkBuiltInShaderStage resolveShader(blessed_ao_resolve, nullptr);
        m_tracePipeline   = device->createBuiltInComputePipeline(m_traceLayout, traceShader);
        m_blurPipeline    = device->createBuiltInComputePipeline(m_blurLayout, blurShader);
        m_resolvePipeline = device->createBuiltInComputePipeline(m_resolveLayout, resolveShader);

        if (device->properties().core.properties.limits.timestampComputeAndGraphics) {
          VkQueryPoolCreateInfo queryInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
          queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
          queryInfo.queryCount = TimestampQueryCount;

          if (device->vkd()->vkCreateQueryPool(device->handle(), &queryInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
            Logger::warn("blessed: ao: failed to create timestamp query pool, timing disabled");
            m_queryPool = VK_NULL_HANDLE;
          }
        }

        m_processStart = dxvk::high_resolution_clock::now();
      }

      void dispatch(
              DxvkContext*            ctx,
        const Rc<DxvkCommandList>&    cmd,
        const BlessedAoDispatchArgs&  args) {
        uint32_t scale = args.halfRes ? 2u : 1u;
        uint32_t traceW = (args.width  + scale - 1u) / scale;
        uint32_t traceH = (args.height + scale - 1u) / scale;

        ensureSized(cmd, traceW, traceH);

        uint32_t readIdx  = m_curr;
        uint32_t writeIdx = 1u - m_curr;

        // blessed: the history was written by the previous dispatch, from its
        // CameraViewProjInverse and CameraPosAdjust; reproject with exactly
        // those (the shadow-slide fix), never the game's "previous" matrix
        float prevViewProj[16] = { };
        bool prevValid = m_havePrev && InvertMatrix(m_prevInvViewProj, prevViewProj);

        DxvkBufferCreateInfo constantsInfo = { };
        constantsInfo.size      = sizeof(BlessedAoConstants);
        constantsInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        constantsInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constantsInfo.access    = VK_ACCESS_2_SHADER_READ_BIT;
        constantsInfo.debugName = "blessed ao constants";

        Rc<DxvkBuffer> constantsBuffer = m_device->createBuffer(constantsInfo,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ctx->ensureBufferAddress(constantsBuffer);

        auto* k = reinterpret_cast<BlessedAoConstants*>(constantsBuffer->getSliceInfo().mapPtr);
        std::memcpy(k->invViewProj, args.invViewProj, sizeof(k->invViewProj));
        std::memcpy(k->prevViewProj, prevViewProj, sizeof(k->prevViewProj));
        std::memcpy(k->camDelta, args.camDelta, sizeof(k->camDelta));
        for (uint32_t i = 0; i < 3; i++)
          k->reprojDelta[i] = args.camPosNow[i] - m_prevCamPos[i];
        k->farDepthValue = args.farDepthValue;
        k->radius        = args.radius;
        k->strength      = args.strength;
        k->rays          = args.rays;
        k->frameIndex    = m_frameIndex;
        k->historyValid  = (m_historyValid && prevValid) ? 1u : 0u;
        k->fullWidth     = args.width;
        k->fullHeight    = args.height;
        k->traceWidth    = traceW;
        k->traceHeight   = traceH;
        k->traceScale    = scale;
        k->debugMode     = uint32_t(args.debug);

        BlessedAoPush push = { };
        push.constantsAddress = constantsBuffer->getSliceInfo().gpuAddress;
        push.tlasAddress      = args.tlasAddress;
        push.samplerIndex     = m_sampler->getDescriptor().samplerIndex;
        push.tlasValid        = args.tlasValid ? 1u : 0u;

        bool forceOnly = args.debug == BlessedAoDebug::Black || args.debug == BlessedAoDebug::White;
        bool blur      = args.debug == BlessedAoDebug::None;

        uint32_t queryBase = beginTiming(cmd);

        if (!forceOnly) {
          // blessed: 1/3 trace + temporal
          std::array<DxvkDescriptorWrite, 4> d = { };
          setSampled(d[0], args.depthView);
          setSampled(d[1], m_accum[readIdx].sampled);
          setStorage(d[2], m_accum[writeIdx].storage);
          setStorage(d[3], m_geo.storage);
          run(cmd, m_tracePipeline, m_traceLayout, d.data(), uint32_t(d.size()), push, traceW, traceH);
          ComputeBarrier(cmd);

          if (blur) {
            // blessed: 2/3 blur, horizontal (accum -> blurA) then vertical (blurA -> blurB)
            std::array<DxvkDescriptorWrite, 4> h = { };
            setSampled(h[0], m_accum[writeIdx].sampled);
            setSampled(h[1], m_geo.sampled);
            setSampled(h[2], m_accum[writeIdx].sampled);
            setStorage(h[3], m_blurA.storage);
            push.passArg = 0u;
            run(cmd, m_blurPipeline, m_blurLayout, h.data(), uint32_t(h.size()), push, traceW, traceH);
            ComputeBarrier(cmd);

            std::array<DxvkDescriptorWrite, 4> v = { };
            setSampled(v[0], m_accum[writeIdx].sampled);
            setSampled(v[1], m_geo.sampled);
            setSampled(v[2], m_blurA.sampled);
            setStorage(v[3], m_blurB.storage);
            push.passArg = 1u;
            run(cmd, m_blurPipeline, m_blurLayout, v.data(), uint32_t(v.size()), push, traceW, traceH);
            ComputeBarrier(cmd);
          }
        }

        // blessed: 3/3 resolve into the game's texture
        std::array<DxvkDescriptorWrite, 5> r = { };
        setSampled(r[0], args.depthView);
        setSampled(r[1], m_accum[writeIdx].sampled);
        setSampled(r[2], m_geo.sampled);
        setSampled(r[3], m_blurB.sampled);
        setStorage(r[4], args.outputView);
        push.passArg = 0u;
        run(cmd, m_resolvePipeline, m_resolveLayout, r.data(), uint32_t(r.size()), push, args.width, args.height);

        endTiming(cmd, queryBase);

        for (uint32_t i = 0; i < 2; i++)
          cmd->track(m_accum[i].image, DxvkAccess::Write);
        cmd->track(m_geo.image, DxvkAccess::Write);
        cmd->track(m_blurA.image, DxvkAccess::Write);
        cmd->track(m_blurB.image, DxvkAccess::Write);
        cmd->track(constantsBuffer, DxvkAccess::Read);

        if (args.tlasRef != nullptr)
          cmd->track(args.tlasRef);

        if (!forceOnly) {
          m_curr = writeIdx;
          m_historyValid = true;
          std::memcpy(m_prevInvViewProj, args.invViewProj, sizeof(m_prevInvViewProj));
          std::memcpy(m_prevCamPos, args.camPosNow, sizeof(m_prevCamPos));
          m_havePrev = true;
        }

        m_frameIndex++;
        m_framesInWindow++;
        if (!args.tlasValid)
          m_invalidTlasInWindow++;

        maybeFlushTimingWindow(traceW, traceH, args.rays);
      }

    private:

      DxvkDevice*               m_device;
      Rc<DxvkSampler>           m_sampler;

      const DxvkPipelineLayout* m_traceLayout   = nullptr;
      const DxvkPipelineLayout* m_blurLayout    = nullptr;
      const DxvkPipelineLayout* m_resolveLayout = nullptr;
      VkPipeline                m_tracePipeline   = VK_NULL_HANDLE;
      VkPipeline                m_blurPipeline    = VK_NULL_HANDLE;
      VkPipeline                m_resolvePipeline = VK_NULL_HANDLE;

      // blessed: trace resolution. accum: rgba32f ping-pong (x = history
      // length * 2 + ao, yzw = camera-relative position, zero = sky).
      // geo: rgba16f (normal, debug value). blurA/blurB: rgba16f, x = ao.
      BlessedAoImage m_accum[2];
      BlessedAoImage m_geo;
      BlessedAoImage m_blurA;
      BlessedAoImage m_blurB;
      uint32_t m_width  = 0u;
      uint32_t m_height = 0u;
      uint32_t m_curr   = 0u;
      bool     m_historyValid = false;

      bool     m_havePrev = false;
      float    m_prevInvViewProj[16] = { };
      float    m_prevCamPos[3] = { 0.0f, 0.0f, 0.0f };
      uint32_t m_frameIndex = 0u;

      VkQueryPool m_queryPool     = VK_NULL_HANDLE;
      uint32_t    m_queryRingNext = 0u;
      bool        m_queryPairUsed[TimestampQueryCount / 2u] = { };

      uint32_t m_framesInWindow       = 0u;
      uint32_t m_invalidTlasInWindow  = 0u;
      double   m_gpuMsSumInWindow     = 0.0;
      uint32_t m_gpuMsSamplesInWindow = 0u;

      dxvk::high_resolution_clock::time_point m_processStart;
      std::ofstream m_timingFile;
      bool          m_timingFileTried = false;

      // blessed: five binding slots at most; VK_DESCRIPTOR_TYPE_MAX_ENUM ends the list
      const DxvkPipelineLayout* makeLayout(std::array<VkDescriptorType, 5> types) {
        std::array<DxvkDescriptorSetLayoutBinding, 5> bindings = { };
        uint32_t count = 0u;

        for (VkDescriptorType t : types) {
          if (t == VK_DESCRIPTOR_TYPE_MAX_ENUM)
            break;
          bindings[count++] = { t, 1u, VK_SHADER_STAGE_COMPUTE_BIT };
        }

        return m_device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedAoPush), count, bindings.data());
      }

      static void setSampled(DxvkDescriptorWrite& w, const Rc<DxvkImageView>& view) {
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w.descriptor     = view->getDescriptor();
      }

      static void setStorage(DxvkDescriptorWrite& w, const Rc<DxvkImageView>& view) {
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.descriptor     = view->getDescriptor();
      }

      static void run(
        const Rc<DxvkCommandList>&  cmd,
              VkPipeline            pipeline,
        const DxvkPipelineLayout*   layout,
        const DxvkDescriptorWrite*  descriptors,
              uint32_t              descriptorCount,
        const BlessedAoPush&        push,
              uint32_t              w,
              uint32_t              h) {
        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, layout,
          descriptorCount, descriptors, sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (w + 7u) / 8u, (h + 7u) / 8u, 1u);
      }

      BlessedAoImage createImage(const Rc<DxvkCommandList>& cmd, VkFormat format, uint32_t w, uint32_t h) {
        DxvkImageCreateInfo info = { };
        info.type        = VK_IMAGE_TYPE_2D;
        info.format      = format;
        info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
        info.extent      = { w, h, 1u };
        info.numLayers   = 1u;
        info.mipLevels   = 1u;
        info.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        info.stages      = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        info.access      = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        info.tiling      = VK_IMAGE_TILING_OPTIMAL;
        info.layout      = VK_IMAGE_LAYOUT_GENERAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        DxvkImageViewKey viewInfo = { };
        viewInfo.viewType   = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
        viewInfo.format     = format;
        viewInfo.layout     = VK_IMAGE_LAYOUT_GENERAL;
        viewInfo.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.mipIndex   = 0u;
        viewInfo.mipCount   = 1u;
        viewInfo.layerIndex = 0u;
        viewInfo.layerCount = 1u;

        BlessedAoImage out;
        out.image   = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        out.sampled = out.image->createView(viewInfo);
        viewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        out.storage = out.image->createView(viewInfo);

        // blessed: one-time UNDEFINED -> GENERAL; ours alone from here on
        VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = out.image->handle();
        barrier.subresourceRange    = out.image->getAvailableSubresources();

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1u;
        dep.pImageMemoryBarriers    = &barrier;

        // blessed: raw
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
        out.image->trackLayout(out.image->getAvailableSubresources(), VK_IMAGE_LAYOUT_GENERAL);
        cmd->track(out.image, DxvkAccess::Write);
        return out;
      }

      void ensureSized(const Rc<DxvkCommandList>& cmd, uint32_t w, uint32_t h) {
        if (w == m_width && h == m_height)
          return;

        for (uint32_t i = 0; i < 2; i++)
          m_accum[i] = createImage(cmd, VK_FORMAT_R32G32B32A32_SFLOAT, w, h);
        m_geo   = createImage(cmd, VK_FORMAT_R16G16B16A16_SFLOAT, w, h);
        m_blurA = createImage(cmd, VK_FORMAT_R16G16B16A16_SFLOAT, w, h);
        m_blurB = createImage(cmd, VK_FORMAT_R16G16B16A16_SFLOAT, w, h);

        m_width  = w;
        m_height = h;
        m_curr   = 0u;
        m_historyValid = false;
        m_havePrev = false;

        Logger::info(str::format("blessed: ao: (re)allocated trace images at ", w, "x", h));
      }

      uint32_t beginTiming(const Rc<DxvkCommandList>& cmd) {
        if (m_queryPool == VK_NULL_HANDLE)
          return UINT32_MAX;

        uint32_t pairIndex = m_queryRingNext;
        uint32_t base      = pairIndex * 2u;
        m_queryRingNext    = (m_queryRingNext + 1u) % (TimestampQueryCount / 2u);

        if (m_queryPairUsed[pairIndex])
          collectTiming(base);

        cmd->cmdResetQueryPool(DxvkCmdBuffer::ExecBuffer, m_queryPool, base, 2u);
        cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer,
          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, base);

        m_queryPairUsed[pairIndex] = true;
        return base;
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
        m_gpuMsSumInWindow += double(results[1].ts - results[0].ts) * period / 1.0e6;
        m_gpuMsSamplesInWindow++;
      }

      // blessed: one line per 120 dispatches to <BLESSED_PROBE_DIR>/rtao.jsonl
      void maybeFlushTimingWindow(uint32_t traceW, uint32_t traceH, uint32_t rays) {
        if (m_framesInWindow < 120u)
          return;

        if (!m_timingFileTried) {
          m_timingFileTried = true;
          std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
          if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            m_timingFile.open(dir + env::PlatformDirSlash + "rtao.jsonl", std::ios::out | std::ios::app);
          }
        }

        if (m_timingFile.is_open()) {
          double t = std::chrono::duration<double>(
            dxvk::high_resolution_clock::now() - m_processStart).count();
          double avgMs = m_gpuMsSamplesInWindow > 0u
            ? m_gpuMsSumInWindow / double(m_gpuMsSamplesInWindow) : 0.0;

          m_timingFile << str::format("{\"t\":", t,
            ",\"frames\":", m_framesInWindow,
            ",\"gpu_ms\":", avgMs,
            ",\"trace\":\"", traceW, "x", traceH, "\"",
            ",\"rays\":", rays,
            ",\"invalid_tlas\":", m_invalidTlasInWindow, "}\n");
          m_timingFile.flush();
        }

        m_framesInWindow       = 0u;
        m_invalidTlasInWindow  = 0u;
        m_gpuMsSumInWindow     = 0.0;
        m_gpuMsSamplesInWindow = 0u;
      }
    };

    BlessedAoObjects* Instance(DxvkDevice* device) {
      static BlessedAoObjects s_instance(device);
      return &s_instance;
    }

  }


  void DxvkContext::blessedRunAoPass(
    const BlessedAoDispatchArgs& argsIn) {
    if (argsIn.outputView == nullptr || argsIn.depthView == nullptr || !argsIn.width || !argsIn.height)
      return;

    bool forceOnly = argsIn.debug == BlessedAoDebug::Black || argsIn.debug == BlessedAoDebug::White;
    if (!forceOnly && !m_device->supportsRayQuery())
      return;

    // blessed: the tlas is resolved here, on the cs thread, at dispatch
    // time -- same reasoning as blessedRunShadowPass
    BlessedAoDispatchArgs args = argsIn;

    BlessedSceneFrame scene = { };
    if (BlessedScene* blessedScene = m_device->blessedScene())
      scene = blessedScene->currentFrame();

    args.tlasAddress = scene.tlasAddress;
    args.tlasValid   = scene.valid;
    args.tlasRef     = scene.tlas;
    for (uint32_t i = 0; i < 3; i++)
      args.camDelta[i] = args.camPosNow[i] - scene.camPos[i];

    // blessed: tag the tlas this frame builds with this frame's camera.
    // The shadow pass does the same with the same b12 value; whichever runs
    // last in a frame wins, and they agree. Needed when rtshadow is off.
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

    DxvkImage* depthImage  = args.depthView->image();
    DxvkImage* outputImage = args.outputView->image();

    VkImageLayout depthRestore = Transition(m_cmd, depthImage,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
        | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        | VK_ACCESS_2_SHADER_READ_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
      true, VK_IMAGE_LAYOUT_UNDEFINED);

    // blessed: vanilla's blur wrote the sao texture as a colour target (or,
    // it being a uav-capable texture, from compute) just before
    VkImageLayout outputRestore = Transition(m_cmd, outputImage,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
        | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      true, VK_IMAGE_LAYOUT_UNDEFINED);

    Instance(m_device.ptr())->dispatch(this, m_cmd, args);

    Transition(m_cmd, depthImage,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
        | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
        | VK_ACCESS_2_SHADER_READ_BIT,
      false, depthRestore);

    // blessed: the composite samples it next (fragment shader)
    Transition(m_cmd, outputImage,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
        | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        | VK_ACCESS_2_SHADER_WRITE_BIT,
      false, outputRestore);

    m_cmd->track(depthImage, DxvkAccess::Read);
    m_cmd->track(outputImage, DxvkAccess::Write);

    // blessed: state-audit -- the pass bound its own compute pipeline,
    // descriptors and push data raw; dxvk still thinks its last pipeline is
    // bound and would skip the rebind for a same-shader dispatch. Same
    // contract as upstream's meta ops.
    this->invalidateState();
  }

}
