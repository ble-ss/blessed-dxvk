// blessed: ray-traced volumetric sun light (BLESSED_VOLUMETRICS) -- dxvk-side passes, history, timing and dump. See blessed_volumetrics.h.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "blessed_async.h" // blessed: async-compute
#include "blessed_volumetrics.h"
#include "blessed_scene.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"
#include "../dxvk_device.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/util_time.h"
#include "../../util/log/log.h"

#include <blessed_vol_trace.h>
#include <blessed_vol_upsample.h>
#include <blessed_vol_dump.h>

namespace dxvk {

  namespace {

    // blessed: matches BlessedVolConstants in blessed_vol_trace.comp byte for
    // byte (scalar layout: tight packing, no vec3 padding). The upsample
    // shader reads only the leading fields.
    struct BlessedVolConstants {
      float    invViewProj[16];
      float    prevViewProj[16];
      float    camDelta[3];
      float    sunDir[3];
      float    reprojDelta[3];
      float    farDepthValue;
      float    phaseG;
      float    density;
      float    heightFalloff;
      float    heightBase;
      float    heightFloor;
      float    range;
      float    sunRayLength;
      float    historyWeight;
      float    rejectRel;
      uint32_t steps;
      uint32_t frameIndex;
      uint32_t historyValid;
      uint64_t lightsAddress;
      uint32_t lightCount;
    };

    // blessed: matches BlessedVolPush in both trace and upsample shaders
    struct BlessedVolPush {
      uint64_t constantsAddress;
      uint64_t tlasAddress;
      uint32_t fullWidth;
      uint32_t fullHeight;
      uint32_t lowWidth;
      uint32_t lowHeight;
      uint32_t divisor;
      uint32_t samplerIndex;
      uint32_t tlasValid;
      uint32_t debugMode;
      float    outScale;
      float    maxOut;        // blessed: vol-2, the upsample's soft shoulder
      uint64_t statsAddress;  // blessed: vol-2, per-group sums or 0
    };

    static_assert(sizeof(BlessedVolConstants) <= 256, "one constants ring slot");

    struct BlessedVolDumpPush {
      uint64_t pixelsAddress;
      uint32_t width;
      uint32_t height;
      uint32_t samplerIndex;
    };

    static_assert(sizeof(BlessedVolPush) <= 128, "push constant budget");
    static_assert(sizeof(BlessedVolDumpPush) <= 128, "push constant budget");

    constexpr uint32_t TimestampQueryCount = 4; // 2 slots, double-buffered

    // blessed: same as blessed_shadow.cpp's (anonymous there, so copied):
    // transitions a game image for compute access, returns the layout to
    // restore. With unified layouts this is a pure synchronization barrier.
    VkImageLayout TransitionForCompute(
      const Rc<DxvkCommandList>&   cmd,
      const Rc<DxvkImageView>&     view,
            VkPipelineStageFlags2  srcStage,
            VkAccessFlags2         srcAccess,
            VkPipelineStageFlags2  dstStage,
            VkAccessFlags2         dstAccess) {
      DxvkImage* image = view->image();
      bool unified = image->hasUnifiedLayout();
      VkImageLayout layout = image->pickLayout(VK_IMAGE_LAYOUT_GENERAL);
      // blessed: state-audit -- the tracked layout, not the default one: a
      // pass suspended by endCurrentPass leaves its attachments in their
      // attachment layout. Unified-layout images never get here.
      VkImageLayout oldLayout = unified ? layout : image->queryLayout(image->getAvailableSubresources());

      if (oldLayout == VK_IMAGE_LAYOUT_MAX_ENUM)
        oldLayout = image->info().layout;

      VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      barrier.srcStageMask        = srcStage;
      barrier.srcAccessMask       = srcAccess;
      barrier.dstStageMask        = dstStage;
      barrier.dstAccessMask       = dstAccess;
      barrier.oldLayout           = oldLayout;
      barrier.newLayout           = layout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image               = image->handle();
      barrier.subresourceRange    = image->getAvailableSubresources();

      VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      dep.imageMemoryBarrierCount = 1u;
      dep.pImageMemoryBarriers    = &barrier;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

      if (!unified)
        image->trackLayout(image->getAvailableSubresources(), layout);

      return oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? image->info().layout : oldLayout;
    }

    void TransitionBack(
      const Rc<DxvkCommandList>&   cmd,
      const Rc<DxvkImageView>&     view,
            VkPipelineStageFlags2  srcStage,
            VkAccessFlags2         srcAccess,
            VkPipelineStageFlags2  dstStage,
            VkAccessFlags2         dstAccess,
            VkImageLayout          restoreLayout) {
      DxvkImage* image = view->image();
      bool unified = image->hasUnifiedLayout();
      VkImageLayout current = image->pickLayout(VK_IMAGE_LAYOUT_GENERAL);

      VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      barrier.srcStageMask        = srcStage;
      barrier.srcAccessMask       = srcAccess;
      barrier.dstStageMask        = dstStage;
      barrier.dstAccessMask       = dstAccess;
      barrier.oldLayout           = current;
      barrier.newLayout           = restoreLayout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image               = image->handle();
      barrier.subresourceRange    = image->getAvailableSubresources();

      VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      dep.imageMemoryBarrierCount = 1u;
      dep.pImageMemoryBarriers    = &barrier;
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

      if (!unified)
        image->trackLayout(image->getAvailableSubresources(), restoreLayout);
    }

    // blessed: general 4x4 inverse in double (same as blessed_soft_shadow.cpp)
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


    /**
     * \brief Pipelines, low-res history and timing for the volumetrics pass
     *
     * One process-lifetime instance, built on the first dispatch (same
     * lifetime model as the shadow pass's objects).
     */
    class BlessedVolObjects {
    public:

      explicit BlessedVolObjects(DxvkDevice* device)
      : m_device(device) {
        DxvkSamplerKey samplerInfo = { };
        samplerInfo.setFilter(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST);
        samplerInfo.setAddressModes(
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        samplerInfo.setUsePixelCoordinates(false);
        m_sampler = device->createSampler(samplerInfo);

        static const std::array<DxvkDescriptorSetLayoutBinding, 3> passBindings = {{
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
          { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        }};

        m_passLayout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedVolPush),
          uint32_t(passBindings.size()), passBindings.data());

        util::DxvkBuiltInShaderStage traceShader(blessed_vol_trace, nullptr);
        m_tracePipeline = device->createBuiltInComputePipeline(m_passLayout, traceShader);

        util::DxvkBuiltInShaderStage upsampleShader(blessed_vol_upsample, nullptr);
        m_upsamplePipeline = device->createBuiltInComputePipeline(m_passLayout, upsampleShader);

        static const std::array<DxvkDescriptorSetLayoutBinding, 1> dumpBindings = {{
          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        }};

        m_dumpLayout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedVolDumpPush),
          uint32_t(dumpBindings.size()), dumpBindings.data());

        util::DxvkBuiltInShaderStage dumpShader(blessed_vol_dump, nullptr);
        m_dumpPipeline = device->createBuiltInComputePipeline(m_dumpLayout, dumpShader);

        if (device->properties().core.properties.limits.timestampComputeAndGraphics) {
          VkQueryPoolCreateInfo queryInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
          queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
          queryInfo.queryCount = TimestampQueryCount;

          if (device->vkd()->vkCreateQueryPool(device->handle(), &queryInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
            Logger::warn("blessed: volumetrics: no timestamp query pool, timing disabled");
            m_queryPool = VK_NULL_HANDLE;
          }
        }

        m_processStart = dxvk::high_resolution_clock::now();
      }

      void run(
              DxvkContext*                 ctx,
        const Rc<DxvkCommandList>&         cmd,
        const BlessedVolDispatchArgs&      args) {
        if (args.width == 0u || args.height == 0u)
          return;

        // blessed: async-compute -- the kick records no dump; the upsample
        // falls back to the whole pass on graphics when the kick is missing,
        // stale, or was made for another size
        if (args.op == BlessedVolDispatchArgs::Op::AsyncTrace) {
          asyncTrace(ctx, cmd, args);
          return;
        }

        if (args.op == BlessedVolDispatchArgs::Op::AsyncUpsample && !asyncUpsample(ctx, cmd, args))
          trace(ctx, cmd, args);

        if (args.op == BlessedVolDispatchArgs::Op::Trace)
          trace(ctx, cmd, args);

        if (args.op == BlessedVolDispatchArgs::Op::Clear) // blessed: vol-2
          clearTarget(ctx, cmd, args);

        if (args.dump)
          dumpTarget(ctx, cmd, args);
      }

    private:

      DxvkDevice*               m_device;
      Rc<DxvkSampler>           m_sampler;
      const DxvkPipelineLayout* m_passLayout       = nullptr;
      VkPipeline                m_tracePipeline    = VK_NULL_HANDLE;
      VkPipeline                m_upsamplePipeline = VK_NULL_HANDLE;
      const DxvkPipelineLayout* m_dumpLayout       = nullptr;
      VkPipeline                m_dumpPipeline     = VK_NULL_HANDLE;

      // low-res ping-pong accumulation (rgba16f: x in-scatter, yzw end
      // point / 64), resident in GENERAL for its whole life
      std::array<Rc<DxvkImage>, 2>     m_accumImage;
      std::array<Rc<DxvkImageView>, 2> m_accumSampledView;
      std::array<Rc<DxvkImageView>, 2> m_accumStorageView;
      uint32_t m_lowWidth  = 0u;
      uint32_t m_lowHeight = 0u;
      uint32_t m_curr      = 0u;
      bool     m_historyValid = false;
      uint32_t m_frameIndex   = 0u;

      // the camera that wrote the current history (shadow-slide's approach)
      bool  m_havePrev = false;
      float m_prevInvViewProj[16] = { };
      float m_prevCamPos[3] = { 0.0f, 0.0f, 0.0f };

      // blessed: async-compute -- the depth copy the async trace samples
      // (graphics copies the game's depth into it at the kick), and what the
      // matching upsample at pass 138 needs from that kick
      Rc<DxvkImage>     m_depthCopy;
      Rc<DxvkImageView> m_depthCopyView;
      Rc<DxvkBuffer>    m_kickConstants;

      // blessed: vol-2 -- per-dispatch constants from one persistent
      // host-visible ring (was a new buffer each dispatch). 32 slots of 256
      // bytes: a slot is reused 32 dispatches later, far past any frame
      // still in flight.
      static constexpr uint32_t ConstSlots  = 32u;
      static constexpr uint32_t ConstStride = 256u;
      Rc<DxvkBuffer> m_constRing;
      uint32_t       m_constNext = 0u;

      // blessed: vol-2 -- the upsample's per-group sums (ours, vanilla's)
      // for the means in volumetrics.jsonl and BLESSED_VOL_GAIN=auto. Read
      // back without waiting: a slot is collected once the gpu is done.
      struct StatsSlot {
        Rc<DxvkBuffer> buffer;
        uint32_t       groups       = 0u;
        uint32_t       pixels       = 0u;
        bool           pending      = false;
        bool           vanillaValid = false;
        float          gainUsed     = 0.0f;
      };
      std::array<StatsSlot, 4> m_stats;
      uint32_t m_statsNext    = 0u;
      uint32_t m_statsCounter = 0u;
      float    m_gain         = 0.0f;
      bool     m_gainInit     = false;
      double   m_oursMean     = -1.0;
      double   m_vanillaMean  = -1.0;

      // blessed: vol-2 -- cpu cost per dispatch, cs thread and app thread
      double   m_csUsSum    = 0.0;
      double   m_appUsSum   = 0.0;
      uint32_t m_cpuSamples = 0u;
      BlessedVolPush    m_kickPush  = { };
      bool              m_kickValid = false;

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

      void ensureSized(const Rc<DxvkCommandList>& cmd, uint32_t lowWidth, uint32_t lowHeight) {
        if (lowWidth == m_lowWidth && lowHeight == m_lowHeight)
          return;

        DxvkImageCreateInfo info = { };
        info.type        = VK_IMAGE_TYPE_2D;
        info.format      = VK_FORMAT_R16G16B16A16_SFLOAT;
        info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
        info.extent      = { lowWidth, lowHeight, 1u };
        info.numLayers   = 1u;
        info.mipLevels   = 1u;
        info.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        info.stages      = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        info.access      = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        info.tiling      = VK_IMAGE_TILING_OPTIMAL;
        info.layout      = VK_IMAGE_LAYOUT_GENERAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        info.debugName   = "blessed volumetrics accum";
        // blessed: async-compute -- the trace may write these on the async
        // queue while the upsample reads them on graphics
        info.blessedConcurrent = BlessedAsync::IsRequested();

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

        m_lowWidth     = lowWidth;
        m_lowHeight    = lowHeight;
        m_curr         = 0u;
        m_historyValid = false;
        m_havePrev     = false;

        Logger::info(str::format("blessed: volumetrics: history at ", lowWidth, "x", lowHeight));
      }

      void trace(
              DxvkContext*                 ctx,
        const Rc<DxvkCommandList>&         cmd,
        const BlessedVolDispatchArgs&      args) {
        if (args.outputStorageView == nullptr || args.depthView == nullptr)
          return;

        uint32_t div = std::max(args.resDivisor, 1u);
        uint32_t lowW = (args.width  + div - 1u) / div;
        uint32_t lowH = (args.height + div - 1u) / div;
        ensureSized(cmd, lowW, lowH);

        uint32_t readIdx  = m_curr;
        uint32_t writeIdx = 1u - m_curr;

        VkImageLayout depthRestore = TransitionForCompute(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // blessed: vol-2 -- with pass 138's draw skipped, the last access was
        // last frame's blur (compute read) or our own write, not a raster write
        VkImageLayout outputRestore = TransitionForCompute(cmd, args.outputStorageView,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

        DxvkBufferCreateInfo constantsInfo = { };
        constantsInfo.size      = sizeof(BlessedVolConstants);
        constantsInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        constantsInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constantsInfo.access    = VK_ACCESS_2_SHADER_READ_BIT;
        constantsInfo.debugName = "blessed volumetrics constants";

        (void) constantsInfo;
        VkDeviceAddress constantsAddress = 0ull; // blessed: vol-2, from the ring
        BlessedVolConstants* kSlot = allocConstants(ctx, constantsAddress);
        Rc<DxvkBuffer> constantsBuffer = m_constRing;

        // blessed: reproject with the camera that wrote the history -- the
        // inverse of last dispatch's CameraViewProjInverse and its
        // CameraPosAdjust (the shadow-slide fix, a61f9921)
        float prevViewProj[16] = { };
        bool prevValid = m_havePrev && InvertMatrix(m_prevInvViewProj, prevViewProj);

        BlessedVolConstants* k = kSlot;
        std::memset(k, 0, sizeof(*k));
        std::memcpy(k->invViewProj, args.invViewProj, sizeof(k->invViewProj));
        std::memcpy(k->prevViewProj, prevViewProj, sizeof(k->prevViewProj));
        std::memcpy(k->camDelta, args.camDelta, sizeof(k->camDelta));
        std::memcpy(k->sunDir, args.sunDir, sizeof(k->sunDir));
        for (uint32_t i = 0; i < 3; i++)
          k->reprojDelta[i] = args.camPosNow[i] - m_prevCamPos[i];
        k->farDepthValue = args.farDepthValue;
        k->phaseG        = args.phaseG;
        k->density       = args.density;
        k->heightFalloff = args.heightFalloff;
        k->heightBase    = args.heightBase;
        k->heightFloor   = args.heightFloor;
        k->range         = args.range;
        k->sunRayLength  = args.sunRayLength;
        k->historyWeight = args.historyWeight;
        k->rejectRel     = args.rejectRel;
        k->steps         = args.steps;
        k->frameIndex    = m_frameIndex;
        k->historyValid  = (m_historyValid && prevValid) ? 1u : 0u;
        k->lightsAddress = 0ull;
        k->lightCount    = 0u;

        BlessedVolPush push = { };
        push.constantsAddress = constantsAddress;
        push.tlasAddress  = args.tlasAddress;
        push.fullWidth    = args.width;
        push.fullHeight   = args.height;
        push.lowWidth     = lowW;
        push.lowHeight    = lowH;
        push.divisor      = div;
        push.samplerIndex = m_sampler->getDescriptor().samplerIndex;
        push.tlasValid    = args.tlasValid ? 1u : 0u;
        push.debugMode    = args.debugMode;
        // blessed: debug views write raw values (dist, vis, ...) so a dump
        // reads them directly; only the real image is scaled
        setOutput(push, args); // blessed: vol-2

        uint32_t queryBase = beginTiming(cmd);

        std::array<DxvkDescriptorWrite, 3> traceDescriptors = { };
        traceDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        traceDescriptors[0].descriptor     = args.depthView->getDescriptor();
        traceDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        traceDescriptors[1].descriptor     = m_accumSampledView[readIdx]->getDescriptor();
        traceDescriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        traceDescriptors[2].descriptor     = m_accumStorageView[writeIdx]->getDescriptor();

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_passLayout,
          uint32_t(traceDescriptors.size()), traceDescriptors.data(), sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (lowW + 7u) / 8u, (lowH + 7u) / 8u, 1u);

        VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &barrier;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

        std::array<DxvkDescriptorWrite, 3> upDescriptors = { };
        upDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        upDescriptors[0].descriptor     = args.depthView->getDescriptor();
        upDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        upDescriptors[1].descriptor     = m_accumSampledView[writeIdx]->getDescriptor();
        upDescriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        upDescriptors[2].descriptor     = args.outputStorageView->getDescriptor();

        StatsSlot* statsSlot = nullptr; // blessed: vol-2
        push.statsAddress = beginStats(ctx, args, statsSlot);

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_upsamplePipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_passLayout,
          uint32_t(upDescriptors.size()), upDescriptors.data(), sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (args.width + 7u) / 8u, (args.height + 7u) / 8u, 1u);

        endStats(cmd, statsSlot);

        endTiming(cmd, queryBase);

        TransitionBack(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
          depthRestore);

        // blessed: next readers are vanilla's blur (compute) and, next
        // frame, pass 138 itself (colour attachment)
        TransitionBack(cmd, args.outputStorageView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          outputRestore);

        cmd->track(args.depthView->image(), DxvkAccess::Read);
        cmd->track(args.outputStorageView->image(), DxvkAccess::Write);
        cmd->track(m_accumImage[readIdx], DxvkAccess::Read);
        cmd->track(m_accumImage[writeIdx], DxvkAccess::Write);
        cmd->track(constantsBuffer, DxvkAccess::Read);

        if (args.tlasRef != nullptr)
          cmd->track(args.tlasRef);

        m_curr = writeIdx;
        m_historyValid = true;
        m_frameIndex++;

        std::memcpy(m_prevInvViewProj, args.invViewProj, sizeof(m_prevInvViewProj));
        std::memcpy(m_prevCamPos, args.camPosNow, sizeof(m_prevCamPos));
        m_havePrev = true;

        m_framesInWindow++;
        if (!args.tlasValid)
          m_invalidTlasInWindow++;

        maybeFlushTimingWindow(lowW, lowH, args);
      }

      // blessed: async-compute -- the depth copy, created concurrent (it is
      // written on graphics and sampled on the async queue) and resident in
      // GENERAL for its whole life, like the accum images
      void ensureDepthCopy(const Rc<DxvkCommandList>& cmd, const DxvkImage& depth) {
        const DxvkImageCreateInfo& src = depth.info();

        if (m_depthCopy != nullptr
         && m_depthCopy->info().format == src.format
         && m_depthCopy->info().extent.width == src.extent.width
         && m_depthCopy->info().extent.height == src.extent.height)
          return;

        DxvkImageCreateInfo info = { };
        info.type        = VK_IMAGE_TYPE_2D;
        info.format      = src.format;
        info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
        info.extent      = { src.extent.width, src.extent.height, 1u };
        info.numLayers   = 1u;
        info.mipLevels   = 1u;
        info.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.stages      = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        info.access      = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        info.tiling      = VK_IMAGE_TILING_OPTIMAL;
        info.layout      = VK_IMAGE_LAYOUT_GENERAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        info.debugName   = "blessed volumetrics depth copy";
        info.blessedConcurrent = true;

        m_depthCopy = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        DxvkImageViewKey viewInfo = { };
        viewInfo.viewType   = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
        viewInfo.format     = src.format;
        viewInfo.layout     = VK_IMAGE_LAYOUT_GENERAL;
        viewInfo.aspects    = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.mipIndex   = 0u;
        viewInfo.mipCount   = 1u;
        viewInfo.layerIndex = 0u;
        viewInfo.layerCount = 1u;
        m_depthCopyView = m_depthCopy->createView(viewInfo);

        VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask       = 0;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = m_depthCopy->handle();
        barrier.subresourceRange    = m_depthCopy->getAvailableSubresources();

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1u;
        dep.pImageMemoryBarriers    = &barrier;

        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep); // blessed: raw
        m_depthCopy->trackLayout(m_depthCopy->getAvailableSubresources(), VK_IMAGE_LAYOUT_GENERAL);
        cmd->track(m_depthCopy, DxvkAccess::Write);

        Logger::info(str::format("blessed: volumetrics: async depth copy at ",
          src.extent.width, "x", src.extent.height));
      }

      // blessed: async-compute -- this frame's constants (camera, reprojection,
      // tuning) and push block, the same values trace() writes
      Rc<DxvkBuffer> makeConstants(
              DxvkContext*                 ctx,
        const BlessedVolDispatchArgs&      args,
              uint32_t                     lowW,
              uint32_t                     lowH,
              uint32_t                     div,
              BlessedVolPush&              push) {
        DxvkBufferCreateInfo constantsInfo = { };
        constantsInfo.size      = sizeof(BlessedVolConstants);
        constantsInfo.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        constantsInfo.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constantsInfo.access    = VK_ACCESS_2_SHADER_READ_BIT;
        constantsInfo.debugName = "blessed volumetrics constants";

        (void) constantsInfo;
        VkDeviceAddress constantsAddress = 0ull; // blessed: vol-2, from the ring
        BlessedVolConstants* kSlot = allocConstants(ctx, constantsAddress);
        Rc<DxvkBuffer> constantsBuffer = m_constRing;

        float prevViewProj[16] = { };
        bool prevValid = m_havePrev && InvertMatrix(m_prevInvViewProj, prevViewProj);

        BlessedVolConstants* k = kSlot;
        std::memset(k, 0, sizeof(*k));
        std::memcpy(k->invViewProj, args.invViewProj, sizeof(k->invViewProj));
        std::memcpy(k->prevViewProj, prevViewProj, sizeof(k->prevViewProj));
        std::memcpy(k->camDelta, args.camDelta, sizeof(k->camDelta));
        std::memcpy(k->sunDir, args.sunDir, sizeof(k->sunDir));
        for (uint32_t i = 0; i < 3; i++)
          k->reprojDelta[i] = args.camPosNow[i] - m_prevCamPos[i];
        k->farDepthValue = args.farDepthValue;
        k->phaseG        = args.phaseG;
        k->density       = args.density;
        k->heightFalloff = args.heightFalloff;
        k->heightBase    = args.heightBase;
        k->heightFloor   = args.heightFloor;
        k->range         = args.range;
        k->sunRayLength  = args.sunRayLength;
        k->historyWeight = args.historyWeight;
        k->rejectRel     = args.rejectRel;
        k->steps         = args.steps;
        k->frameIndex    = m_frameIndex;
        k->historyValid  = (m_historyValid && prevValid) ? 1u : 0u;
        k->lightsAddress = 0ull;
        k->lightCount    = 0u;

        push = BlessedVolPush();
        push.constantsAddress = constantsAddress;
        push.tlasAddress  = args.tlasAddress;
        push.fullWidth    = args.width;
        push.fullHeight   = args.height;
        push.lowWidth     = lowW;
        push.lowHeight    = lowH;
        push.divisor      = div;
        push.samplerIndex = m_sampler->getDescriptor().samplerIndex;
        push.tlasValid    = args.tlasValid ? 1u : 0u;
        push.debugMode    = args.debugMode;
        push.outScale     = 1.0f; // set by the upsample, from pass 138's own intensity
        return constantsBuffer;
      }

      // blessed: async-compute -- runs at the kick draw (pass 127). Graphics:
      // size the history, copy the game's depth. Async queue: trace +
      // temporal into the low-res accum. The upsample at pass 138 waits.
      void asyncTrace(
              DxvkContext*                 ctx,
        const Rc<DxvkCommandList>&         cmd,
        const BlessedVolDispatchArgs&      args) {
        m_kickValid = false;

        if (!ctx->blessedAsyncAvailable() || args.depthView == nullptr)
          return;

        if (!(args.depthView->image()->formatInfo()->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT))
          return;

        uint32_t div = std::max(args.resDivisor, 1u);
        uint32_t lowW = (args.width  + div - 1u) / div;
        uint32_t lowH = (args.height + div - 1u) / div;

        // graphics: everything that creates or initializes an image
        ensureSized(cmd, lowW, lowH);
        ensureDepthCopy(cmd, *args.depthView->image());

        VkImageLayout depthRestore = TransitionForCompute(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        DxvkImage* depthImage = args.depthView->image();

        VkImageCopy2 region = { VK_STRUCTURE_TYPE_IMAGE_COPY_2 };
        region.srcSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 0u, 1u };
        region.dstSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 0u, 1u };
        region.extent         = { depthImage->info().extent.width, depthImage->info().extent.height, 1u };

        VkCopyImageInfo2 copyInfo = { VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2 };
        copyInfo.srcImage       = depthImage->handle();
        copyInfo.srcImageLayout = depthImage->pickLayout(VK_IMAGE_LAYOUT_GENERAL);
        copyInfo.dstImage       = m_depthCopy->handle();
        copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        copyInfo.regionCount    = 1u;
        copyInfo.pRegions       = &region;
        cmd->cmdCopyImage(DxvkCmdBuffer::ExecBuffer, &copyInfo);

        TransitionBack(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          depthRestore);

        cmd->track(depthImage, DxvkAccess::Read);
        cmd->track(m_depthCopy, DxvkAccess::Write);

        BlessedVolPush push = { };
        Rc<DxvkBuffer> constantsBuffer = makeConstants(ctx, args, lowW, lowH, div, push);

        uint32_t readIdx  = m_curr;
        uint32_t writeIdx = 1u - m_curr;

        // async queue from here: compute stages only
        ctx->blessedAsyncBegin();

        // the previous kick read and wrote these accum images on this queue
        // (a barrier's first scope spans earlier submissions on the queue);
        // graphics-side accesses reach us through the kick's semaphore wait
        VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &barrier;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep); // blessed: raw

        uint32_t queryBase = beginTiming(cmd);

        std::array<DxvkDescriptorWrite, 3> traceDescriptors = { };
        traceDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        traceDescriptors[0].descriptor     = m_depthCopyView->getDescriptor();
        traceDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        traceDescriptors[1].descriptor     = m_accumSampledView[readIdx]->getDescriptor();
        traceDescriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        traceDescriptors[2].descriptor     = m_accumStorageView[writeIdx]->getDescriptor();

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_passLayout,
          uint32_t(traceDescriptors.size()), traceDescriptors.data(), sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (lowW + 7u) / 8u, (lowH + 7u) / 8u, 1u);

        endTiming(cmd, queryBase);

        cmd->track(m_depthCopy, DxvkAccess::Read);
        cmd->track(m_accumImage[readIdx], DxvkAccess::Read);
        cmd->track(m_accumImage[writeIdx], DxvkAccess::Write);
        cmd->track(constantsBuffer, DxvkAccess::Read);

        if (args.tlasRef != nullptr)
          cmd->track(args.tlasRef);

        ctx->blessedAsyncEnd();

        m_kickConstants = constantsBuffer;
        m_kickPush      = push;
        m_kickValid     = true;

        m_curr = writeIdx;
        m_historyValid = true;
        m_frameIndex++;

        std::memcpy(m_prevInvViewProj, args.invViewProj, sizeof(m_prevInvViewProj));
        std::memcpy(m_prevCamPos, args.camPosNow, sizeof(m_prevCamPos));
        m_havePrev = true;

        m_framesInWindow++;
        if (!args.tlasValid)
          m_invalidTlasInWindow++;

        maybeFlushTimingWindow(lowW, lowH, args);
      }

      // blessed: async-compute -- pass 138, graphics, after blessedAsyncSync:
      // upsample the kick's accum into the target. False (nothing recorded)
      // when there is no usable kick; the caller then runs trace().
      bool asyncUpsample(
              DxvkContext*                 ctx,
        const Rc<DxvkCommandList>&         cmd,
        const BlessedVolDispatchArgs&      args) {
        bool valid = m_kickValid;
        m_kickValid = false;

        if (!valid || args.outputStorageView == nullptr || args.depthView == nullptr)
          return false;

        uint32_t div = std::max(args.resDivisor, 1u);

        if (m_kickPush.fullWidth != args.width || m_kickPush.fullHeight != args.height
         || m_kickPush.divisor != div || m_kickPush.lowWidth != m_lowWidth || m_kickPush.lowHeight != m_lowHeight)
          return false;

        VkImageLayout depthRestore = TransitionForCompute(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // blessed: vol-2 -- with pass 138's draw skipped, the last access was
        // last frame's blur (compute read) or our own write, not a raster write
        VkImageLayout outputRestore = TransitionForCompute(cmd, args.outputStorageView,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

        BlessedVolPush push = m_kickPush;
        setOutput(push, args); // blessed: vol-2

        std::array<DxvkDescriptorWrite, 3> upDescriptors = { };
        upDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        upDescriptors[0].descriptor     = args.depthView->getDescriptor();
        upDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        upDescriptors[1].descriptor     = m_accumSampledView[m_curr]->getDescriptor();
        upDescriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        upDescriptors[2].descriptor     = args.outputStorageView->getDescriptor();

        StatsSlot* statsSlot = nullptr; // blessed: vol-2
        push.statsAddress = beginStats(ctx, args, statsSlot);

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_upsamplePipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_passLayout,
          uint32_t(upDescriptors.size()), upDescriptors.data(), sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (args.width + 7u) / 8u, (args.height + 7u) / 8u, 1u);

        endStats(cmd, statsSlot);

        TransitionBack(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
          depthRestore);

        TransitionBack(cmd, args.outputStorageView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          outputRestore);

        cmd->track(args.depthView->image(), DxvkAccess::Read);
        cmd->track(args.outputStorageView->image(), DxvkAccess::Write);
        cmd->track(m_accumImage[m_curr], DxvkAccess::Read);
        cmd->track(m_kickConstants, DxvkAccess::Read);
        m_kickConstants = nullptr;
        return true;
      }

      // blessed: vol-2 -- one slot of the constants ring
      BlessedVolConstants* allocConstants(DxvkContext* ctx, VkDeviceAddress& address) {
        if (m_constRing == nullptr) {
          DxvkBufferCreateInfo info = { };
          info.size      = VkDeviceSize(ConstSlots) * ConstStride;
          info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
          info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          info.access    = VK_ACCESS_2_SHADER_READ_BIT;
          info.debugName = "blessed volumetrics constants ring";
          m_constRing = m_device->createBuffer(info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
          ctx->ensureBufferAddress(m_constRing);
        }

        uint32_t slot = m_constNext++ % ConstSlots;
        auto slice = m_constRing->getSliceInfo();
        address = slice.gpuAddress + VkDeviceAddress(slot) * ConstStride;
        return reinterpret_cast<BlessedVolConstants*>(
          reinterpret_cast<uint8_t*>(slice.mapPtr) + size_t(slot) * ConstStride);
      }

      float currentGain(const BlessedVolDispatchArgs& args) {
        if (!args.autoGain)
          return args.gain;
        if (!m_gainInit) {
          m_gain = args.gain;
          m_gainInit = true;
        }
        return m_gain;
      }

      // blessed: vol-2 -- the upsample's scale and shoulder (debug views raw)
      void setOutput(BlessedVolPush& push, const BlessedVolDispatchArgs& args) {
        bool rawDebug = args.debugMode == 1u || args.debugMode == 2u || args.debugMode == 4u || args.debugMode == 5u;
        push.outScale = rawDebug ? 1.0f : args.gameIntensity * currentGain(args);
        push.maxOut   = rawDebug ? 0.0f : args.maxOut;
      }

      // blessed: vol-2 -- collects every finished stats slot (never waits)
      void collectStats(const BlessedVolDispatchArgs& args) {
        for (auto& slot : m_stats) {
          if (!slot.pending || slot.buffer->isInUse(DxvkAccess::Write))
            continue;

          slot.pending = false;
          const float* sums = reinterpret_cast<const float*>(slot.buffer->getSliceInfo().mapPtr);
          double ours = 0.0, vanilla = 0.0;
          for (uint32_t i = 0; i < slot.groups; i++) {
            ours    += double(sums[2u * i + 0u]);
            vanilla += double(sums[2u * i + 1u]);
          }

          m_oursMean = ours / double(std::max(slot.pixels, 1u));
          m_vanillaMean = slot.vanillaValid ? vanilla / double(std::max(slot.pixels, 1u)) : -1.0;

          // blessed: steer halfway (in log space) toward the gain that
          // would have matched vanilla's mean on that frame; the shoulder
          // makes the relation sub-linear, which the iteration absorbs
          if (args.autoGain && slot.vanillaValid && m_oursMean > 1.0e-6 && m_vanillaMean > 1.0e-6) {
            double desired = double(slot.gainUsed) * (m_vanillaMean / m_oursMean);
            double next = std::exp(0.5 * (std::log(double(m_gain)) + std::log(desired)));
            m_gain = float(std::clamp(next, 0.25, 400.0));
          }
        }
      }

      // blessed: vol-2 -- a stats slot for this upsample, or 0
      uint64_t beginStats(DxvkContext* ctx, const BlessedVolDispatchArgs& args, StatsSlot*& used) {
        used = nullptr;
        collectStats(args);

        if (!args.statsEvery || args.debugMode != 0u)
          return 0ull;
        if ((m_statsCounter++ % args.statsEvery) != 0u)
          return 0ull;

        StatsSlot& slot = m_stats[m_statsNext];
        if (slot.pending)
          return 0ull;
        m_statsNext = (m_statsNext + 1u) % uint32_t(m_stats.size());

        uint32_t groups = ((args.width + 7u) / 8u) * ((args.height + 7u) / 8u);
        if (slot.buffer == nullptr || slot.groups != groups) {
          DxvkBufferCreateInfo info = { };
          info.size      = VkDeviceSize(groups) * 2u * sizeof(float);
          info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
          info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          info.access    = VK_ACCESS_2_SHADER_WRITE_BIT;
          info.debugName = "blessed volumetrics stats";
          slot.buffer = m_device->createBuffer(info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
          ctx->ensureBufferAddress(slot.buffer);
          slot.groups = groups;
        }

        slot.pixels       = args.width * args.height;
        slot.pending      = true;
        slot.vanillaValid = !args.vanillaSkipped;
        slot.gainUsed     = currentGain(args);
        used = &slot;
        return slot.buffer->getSliceInfo().gpuAddress;
      }

      void endStats(const Rc<DxvkCommandList>& cmd, StatsSlot* used) {
        if (!used)
          return;

        VkMemoryBarrier2 hostBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        hostBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        hostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        hostBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &hostBarrier;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep); // blessed: raw
        cmd->track(used->buffer, DxvkAccess::Write);
      }

    public:

      void noteCpu(double csUs, double appUs) {
        m_csUsSum  += csUs;
        m_appUsSum += appUs;
        m_cpuSamples++;
      }

      // blessed: vol-2 -- the draw was skipped and we cannot trace (no tlas):
      // write zeros so the blur and pass 167 do not see last frame's image
      void clearTarget(
              DxvkContext*                 ctx,
        const Rc<DxvkCommandList>&         cmd,
        const BlessedVolDispatchArgs&      args) {
        if (args.outputStorageView == nullptr || args.depthView == nullptr)
          return;

        uint32_t div = std::max(args.resDivisor, 1u);
        ensureSized(cmd, (args.width + div - 1u) / div, (args.height + div - 1u) / div);

        VkImageLayout outputRestore = TransitionForCompute(cmd, args.outputStorageView,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

        BlessedVolPush push = { };
        push.fullWidth    = args.width;
        push.fullHeight   = args.height;
        push.lowWidth     = m_lowWidth;
        push.lowHeight    = m_lowHeight;
        push.divisor      = div;
        push.samplerIndex = m_sampler->getDescriptor().samplerIndex;
        push.debugMode    = 255u; // the upsample's clear mode

        std::array<DxvkDescriptorWrite, 3> descriptors = { };
        descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        descriptors[0].descriptor     = args.depthView->getDescriptor();
        descriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        descriptors[1].descriptor     = m_accumSampledView[m_curr]->getDescriptor();
        descriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptors[2].descriptor     = args.outputStorageView->getDescriptor();

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_upsamplePipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_passLayout,
          uint32_t(descriptors.size()), descriptors.data(), sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (args.width + 7u) / 8u, (args.height + 7u) / 8u, 1u);

        TransitionBack(cmd, args.outputStorageView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          outputRestore);

        cmd->track(args.outputStorageView->image(), DxvkAccess::Write);
        cmd->track(m_accumImage[m_curr], DxvkAccess::Read);
      }

    private:

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

      // blessed: <BLESSED_PROBE_DIR>/volumetrics.jsonl, one line per 120 dispatches
      void maybeFlushTimingWindow(uint32_t lowW, uint32_t lowH, const BlessedVolDispatchArgs& args) {
        if (m_framesInWindow < 120u)
          return;

        if (!m_timingFileTried) {
          m_timingFileTried = true;
          std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
          if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            m_timingFile.open(dir + env::PlatformDirSlash + "volumetrics.jsonl", std::ios::out | std::ios::app);
          }
        }

        if (m_timingFile.is_open()) {
          double t = std::chrono::duration<double>(dxvk::high_resolution_clock::now() - m_processStart).count();
          double avgMs = m_gpuMsSamplesInWindow > 0 ? m_gpuMsSumInWindow / double(m_gpuMsSamplesInWindow) : 0.0;

          m_timingFile << str::format("{\"t\":", t,
            ",\"frames\":", m_framesInWindow,
            ",\"gpu_ms\":", avgMs,
            ",\"low\":[", lowW, ",", lowH, "]",
            ",\"steps\":", args.steps,
            ",\"invalid_tlas\":", m_invalidTlasInWindow,
            ",\"game_intensity\":", args.gameIntensity,
            ",\"gain\":", currentGain(args),
            ",\"auto_gain\":", args.autoGain ? 1 : 0,
            ",\"ours_mean\":", m_oursMean,
            ",\"vanilla_mean\":", m_vanillaMean,
            ",\"vanilla_skipped\":", args.vanillaSkipped ? 1 : 0,
            ",\"cs_us\":", m_cpuSamples ? m_csUsSum / double(m_cpuSamples) : 0.0,
            ",\"app_us\":", m_cpuSamples ? m_appUsSum / double(m_cpuSamples) : 0.0,
            ",\"sun\":[", args.sunDir[0], ",", args.sunDir[1], ",", args.sunDir[2], "]}\n");
          m_timingFile.flush();
        }

        m_framesInWindow       = 0u;
        m_invalidTlasInWindow  = 0u;
        m_gpuMsSumInWindow     = 0.0;
        m_gpuMsSamplesInWindow = 0u;
        m_csUsSum    = 0.0;
        m_appUsSum   = 0.0;
        m_cpuSamples = 0u;
      }

      // blessed: BLESSED_VOL_DUMP -- debug only, deliberately blocking. Copies
      // the target (after our pass, or vanilla's own output) into a host
      // buffer, writes a .pgm (byte = value * dumpScale) and logs stats.
      void dumpTarget(
              DxvkContext*                 ctx,
        const Rc<DxvkCommandList>&         cmd,
        const BlessedVolDispatchArgs&      args) {
        if (args.outputSampledView == nullptr)
          return;

        VkDeviceSize size = VkDeviceSize(args.width) * args.height * sizeof(float);

        DxvkBufferCreateInfo info = { };
        info.size      = size;
        info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        info.access    = VK_ACCESS_2_SHADER_WRITE_BIT;
        info.debugName = "blessed volumetrics dump";

        Rc<DxvkBuffer> buffer = m_device->createBuffer(info,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ctx->ensureBufferAddress(buffer);

        VkImageLayout restore = TransitionForCompute(cmd, args.outputSampledView,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
            | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        BlessedVolDumpPush push = { };
        push.pixelsAddress = buffer->getSliceInfo().gpuAddress;
        push.width         = args.width;
        push.height        = args.height;
        push.samplerIndex  = m_sampler->getDescriptor().samplerIndex;

        DxvkDescriptorWrite descriptor = { };
        descriptor.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        descriptor.descriptor     = args.outputSampledView->getDescriptor();

        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpPipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_dumpLayout, 1u, &descriptor, sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (args.width + 7u) / 8u, (args.height + 7u) / 8u, 1u);

        TransitionBack(cmd, args.outputSampledView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          restore);

        VkMemoryBarrier2 hostBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        hostBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        hostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        hostBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &hostBarrier;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

        cmd->track(args.outputSampledView->image(), DxvkAccess::Read);
        cmd->track(buffer, DxvkAccess::Write);

        // blessed: same blocking submit as the shadow pass's dump
        Rc<DxvkCommandList> finished = ctx->endRecording(nullptr);
        DxvkSubmitStatus status = { };
        m_device->submitCommandList(finished, nullptr, 0u, &status);
        m_device->waitForSubmission(&status);
        m_device->waitForIdle();
        ctx->beginRecording(m_device->createCommandList());

        const float* px = reinterpret_cast<const float*>(buffer->getSliceInfo().mapPtr);
        size_t count = size_t(args.width) * args.height;

        std::vector<float> sorted(px, px + count);
        double sum = 0.0;
        size_t nonzero = 0u;
        for (float v : sorted) {
          sum += double(v);
          if (v > 1.0f / 255.0f)
            nonzero++;
        }
        std::sort(sorted.begin(), sorted.end());

        Logger::info(str::format("blessed: volumetrics dump [", args.dumpTag, "] ", args.width, "x", args.height,
          " mean=", sum / double(count),
          " p50=", sorted[count / 2u],
          " p95=", sorted[size_t(double(count - 1u) * 0.95)],
          " max=", sorted[count - 1u],
          " min=", sorted[0],
          " nonzero=", double(nonzero) / double(count),
          " tlas_valid=", args.tlasValid ? 1 : 0,
          " game_intensity=", args.gameIntensity,
          " -> ", args.dumpPath));

        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
        if (!dir.empty()) {
          std::ofstream stats(dir + env::PlatformDirSlash + "volumetrics-dump.jsonl", std::ios::out | std::ios::app);
          stats << str::format("{\"tag\":\"", args.dumpTag, "\",\"mean\":", sum / double(count),
            ",\"p50\":", sorted[count / 2u], ",\"p95\":", sorted[size_t(double(count - 1u) * 0.95)],
            ",\"max\":", sorted[count - 1u], ",\"min\":", sorted[0],
            ",\"nonzero\":", double(nonzero) / double(count),
            ",\"tlas_valid\":", args.tlasValid ? 1 : 0,
            ",\"game_intensity\":", args.gameIntensity, "}\n");
        }

        std::error_code ec;
        std::filesystem::create_directories(args.dumpPath.substr(0, args.dumpPath.find_last_of("/\\")), ec);

        std::ofstream pgm(args.dumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!pgm.is_open()) {
          Logger::warn(str::format("blessed: volumetrics dump: could not open '", args.dumpPath, "'"));
          return;
        }

        pgm << "P5\n" << args.width << " " << args.height << "\n255\n";
        std::vector<uint8_t> row(args.width);
        for (uint32_t y = 0; y < args.height; y++) {
          for (uint32_t x = 0; x < args.width; x++) {
            float v = px[size_t(y) * args.width + x] * args.dumpScale;
            row[x] = uint8_t(std::clamp(v, 0.0f, 255.0f) + 0.5f);
          }
          pgm.write(reinterpret_cast<const char*>(row.data()), row.size());
        }
      }
    };

    BlessedVolObjects* Instance(DxvkDevice* device) {
      static BlessedVolObjects s_instance(device);
      return &s_instance;
    }

  }


  void DxvkContext::blessedRunVolumetricsPass(
    const BlessedVolDispatchArgs& argsIn) {
    // blessed: async-compute -- the kick and the upsample are the trace too
    bool traceOp = argsIn.op == BlessedVolDispatchArgs::Op::Trace
                || argsIn.op == BlessedVolDispatchArgs::Op::AsyncTrace
                || argsIn.op == BlessedVolDispatchArgs::Op::AsyncUpsample;

    if (traceOp && !m_device->supportsRayQuery())
      return;

    // blessed: tlas resolved here, on the cs thread, like the shadow pass
    // (see DxvkContext::blessedRunShadowPass for the race this avoids).
    // The shadow pass tags the tlas with its camera; this pass only reads.
    BlessedVolDispatchArgs args = argsIn;

    BlessedSceneFrame scene = { };
    if (BlessedScene* blessedScene = m_device->blessedScene())
      scene = blessedScene->currentFrame();

    args.tlasAddress = scene.tlasAddress;
    args.tlasValid   = scene.valid;
    args.tlasRef     = scene.tlas;
    for (uint32_t i = 0; i < 3; i++)
      args.camDelta[i] = args.camPosNow[i] - scene.camPos[i];

    // blessed: no tlas, no rays -- leave vanilla's output alone rather than
    // write an unshadowed fog (the dump still runs, tagged tlas_valid=0)
    if (traceOp && !args.tlasValid && args.debugMode != 1u && args.debugMode != 5u) {
      // blessed: async-compute -- no kick then (the upsample falls back and
      // lands here as DumpOnly too)
      if (args.op == BlessedVolDispatchArgs::Op::AsyncTrace)
        return;

      // blessed: vol-2 -- vanilla's draw was skipped, so the target holds
      // last frame's image: clear it rather than leave it
      args.op = args.vanillaSkipped ? BlessedVolDispatchArgs::Op::Clear : BlessedVolDispatchArgs::Op::DumpOnly;
    }

    static uint64_t s_count = 0;
    if (++s_count == 120) {
      Logger::info(str::format("blessed: volumetrics tlas cam=(", scene.camPos[0], ",", scene.camPos[1], ",", scene.camPos[2],
        ") now=(", args.camPosNow[0], ",", args.camPosNow[1], ",", args.camPosNow[2],
        ") sun=(", args.sunDir[0], ",", args.sunDir[1], ",", args.sunDir[2],
        ") intensity=", args.gameIntensity, " valid=", scene.valid ? 1 : 0));
    }

    // blessed: async-compute -- every op below writes images an outstanding
    // kick may still use (the kick's own accum and depth copy); the upsample
    // is the consumer this wait exists for. One bool test when none is.
    this->blessedAsyncSync();

    this->endCurrentPass(true);

    // blessed: BLESSED_VOLUMETRICS=fill clears the target through dxvk, which
    // may defer the clear; our raw barriers and dispatches would not see it.
    // Resolve it now, and flush dxvk's batched barriers ahead of ours.
    const Rc<DxvkImageView>& target = args.outputStorageView != nullptr ? args.outputStorageView : args.outputSampledView;
    if (target != nullptr) {
      DxvkImage* image = target->image();
      this->flushDeferredClear(*image, image->getAvailableSubresources());
    }
    this->flushBarriers();

    auto csT0 = dxvk::high_resolution_clock::now(); // blessed: vol-2, cs_us
    BlessedVolObjects* objects = Instance(m_device.ptr());
    objects->run(this, m_cmd, args);

    if (args.op != BlessedVolDispatchArgs::Op::AsyncTrace) {
      double csUs = std::chrono::duration<double, std::micro>(dxvk::high_resolution_clock::now() - csT0).count();
      objects->noteCpu(csUs, double(args.appNs) / 1000.0);
    }

    // blessed: state-audit -- the pass bound its own compute pipeline,
    // descriptors and push data raw; dxvk still thinks its last pipeline is
    // bound and would skip the rebind for a same-shader dispatch. Same
    // contract as upstream's meta ops.
    this->invalidateState();
  }

}
