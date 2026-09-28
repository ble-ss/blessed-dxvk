// blessed: ray-traced sun shadow pass -- dxvk-side compute dispatch. See blessed_shadow.h.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "blessed_shadow.h"
#include "blessed_soft_shadow.h"
#include "blessed_scene.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"
#include "../dxvk_device.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/util_time.h"
#include "../../util/log/log.h"

#include <blessed_sun_shadow.h>

namespace dxvk {

  namespace {

    // blessed: push data layout must match BlessedShadowPushData in
    // shaders/blessed_sun_shadow.comp exactly (scalar layout, 128 bytes --
    // the Vulkan-guaranteed minimum maxPushConstantsSize)
    struct BlessedShadowPushData {
      float    invViewProj[16];
      uint64_t tlasAddress;
      uint64_t debugAddress;
      float    camDelta[3];
      float    sunDir[3];
      float    farDepthValue;
      uint32_t width;
      uint32_t height;
      uint32_t tlasValid;
      uint32_t samplerIndex;
      uint32_t debugMode;  // blessed: BLESSED_SHADOW_DEBUG (0 off, 1 dist rings, 2 hit t; 7 hist and 8 reproj are soft-only)
    };

    static_assert(sizeof(BlessedShadowPushData) == 128,
      "must match the 128-byte layout blessed_sun_shadow.comp expects");

    constexpr uint32_t TimestampQueryCount = 4; // 2 slots, double-buffered

    // blessed: transitions one image for compute access and reports the
    // layout to restore afterward. On hardware exposing unified image
    // layouts (VK_KHR_unified_image_layouts; expected on this project's rtx
    // 3060 ti) the image never actually left VK_IMAGE_LAYOUT_GENERAL, so
    // this degenerates to a pure synchronization barrier. On hardware
    // without it, this assumes the image's resting layout is its default
    // DxvkImageCreateInfo::layout -- unverified against a real validation
    // run; see the shadow-pass report for this caveat.
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

      // blessed: raw -- manual VkImageMemoryBarrier2 on an image dxvk itself
      // still owns and tracks (the game's rtv0/dsv). See the caveat above.
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

      // blessed: raw -- see TransitionForCompute
      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

      if (!unified)
        image->trackLayout(image->getAvailableSubresources(), restoreLayout);
    }


    /**
     * \brief Lazily-built pipeline + per-run state for the shadow pass
     *
     * A single process-lifetime instance (see Instance()), same lifetime
     * model as the other blessed meta objects -- there is only ever one
     * DxvkDevice in this fork's process.
     */
    class BlessedShadowObjects {
    public:

      explicit BlessedShadowObjects(DxvkDevice* device)
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
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedShadowPushData),
          uint32_t(bindings.size()), bindings.data());

        util::DxvkBuiltInShaderStage shader(blessed_sun_shadow, nullptr);
        m_pipeline = device->createBuiltInComputePipeline(m_layout, shader);

        if (device->properties().core.properties.limits.timestampComputeAndGraphics) {
          VkQueryPoolCreateInfo queryInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
          queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
          queryInfo.queryCount = TimestampQueryCount;

          VkResult vr = device->vkd()->vkCreateQueryPool(
            device->handle(), &queryInfo, nullptr, &m_queryPool);

          if (vr != VK_SUCCESS) {
            Logger::warn("blessed: shadow pass: failed to create timestamp query pool, timing disabled");
            m_queryPool = VK_NULL_HANDLE;
          }
        }

        m_processStart = dxvk::high_resolution_clock::now();
      }

      void dispatch(
              DxvkContext*                ctx,
        const Rc<DxvkCommandList>&        cmd,
        const BlessedShadowDispatchArgs&  args) {
        if (args.width == 0u || args.height == 0u)
          return;

        // blessed: debug buffer (counters + optional pixel dump), only ever
        // allocated on the frame it's actually needed
        Rc<DxvkBuffer> debugBuffer;
        VkDeviceSize debugSize = 0u;

        if (args.dump) {
          debugSize = 16ull + VkDeviceSize(args.width) * args.height * sizeof(uint32_t);

          DxvkBufferCreateInfo info = { };
          info.size      = debugSize;
          info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
          info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          info.access    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
          info.debugName = "blessed shadow debug";

          debugBuffer = m_device->createBuffer(info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
          std::memset(debugBuffer->getSliceInfo().mapPtr, 0, size_t(debugSize));
          ctx->ensureBufferAddress(debugBuffer);
        }

        // blessed: the render pass is already ended by DxvkContext::blessedRunShadowPass
        // (endCurrentPass is private to DxvkContext; this object is not a member)

        VkImageLayout depthRestore = TransitionForCompute(cmd, args.depthView,
          // blessed: the mask pass read depth as a texture (fragment shader);
          // earlier passes wrote it as the depth attachment
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
            | VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // blessed: skip-more -- with BLESSED_SHADOW_SKIP_MASK_DRAW=1 the raster
        // mask draw never runs, so the last real access to this image can be
        // our OWN previous frame's compute write (or read), not a colour
        // attachment write. A src scope that only names the colour-attachment
        // stage misses that writer entirely and under-synchronizes against it
        // (the same hazard vol-2 fixed for pass 138's output; see
        // blessed_volumetrics.cpp's matching comment). Naming both stages is
        // always a safe superset, so this costs nothing when the raster draw
        // does run.
        VkImageLayout outputRestore = TransitionForCompute(cmd, args.outputView,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

        BlessedShadowPushData push = { };
        std::memcpy(push.invViewProj, args.invViewProj, sizeof(push.invViewProj));
        push.tlasAddress   = args.tlasAddress;
        push.debugAddress  = debugBuffer != nullptr ? debugBuffer->getSliceInfo().gpuAddress : 0ull;
        std::memcpy(push.camDelta, args.camDelta, sizeof(push.camDelta));
        std::memcpy(push.sunDir, args.sunDir, sizeof(push.sunDir));
        push.farDepthValue = args.farDepthValue;
        push.width         = args.width;
        push.height        = args.height;
        push.tlasValid     = args.tlasValid ? 1u : 0u;
        push.samplerIndex  = m_sampler->getDescriptor().samplerIndex;
        static const uint32_t s_debugMode = [] {
          std::string m = env::getEnvVar("BLESSED_SHADOW_DEBUG");
          return m == "dist" ? 1u : m == "t" ? 2u : m == "inst" ? 3u : m == "align" ? 4u : m == "ratio" ? 5u : m == "yzw1" ? 6u : m == "hist" ? 7u : m == "reproj" ? 8u : 0u;
        }();
        push.debugMode     = s_debugMode;

        uint32_t queryBase = beginTiming(cmd);

        if (args.softEnabled) {
          // blessed: BLESSED_SHADOW_SOFT=1 -- built lazily, only the first
          // time a soft dispatch actually happens, so the hard-shadow path
          // above never pays for it
          if (m_soft == nullptr)
            m_soft = std::make_unique<BlessedSoftShadowState>(m_device);

          m_soft->dispatch(ctx, cmd, args, push.debugMode,
            debugBuffer != nullptr ? debugBuffer->getSliceInfo().gpuAddress : 0ull);
        } else {
          std::array<DxvkDescriptorWrite, 2> descriptors = { };
          descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
          descriptors[0].descriptor     = args.depthView->getDescriptor();
          descriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
          descriptors[1].descriptor     = args.outputView->getDescriptor();

          cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
          cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_layout,
            uint32_t(descriptors.size()), descriptors.data(), sizeof(push), &push);

          uint32_t groupsX = (args.width  + 7u) / 8u;
          uint32_t groupsY = (args.height + 7u) / 8u;
          cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, groupsX, groupsY, 1u);
        }

        endTiming(cmd, queryBase);

        if (args.dump) {
          // blessed: debug-only, deliberately blocking -- see BlockingReadback
          blockingReadback(ctx, cmd, args, debugBuffer, debugSize);
        }

        TransitionBack(cmd, args.depthView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
          depthRestore);

        TransitionBack(cmd, args.outputView,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
          outputRestore);

        cmd->track(args.depthView->image(), DxvkAccess::Read);
        cmd->track(args.outputView->image(), DxvkAccess::Write);

        if (debugBuffer != nullptr)
          cmd->track(debugBuffer, DxvkAccess::Read);

        // blessed: keep the tlas (and, transitively, every blas it
        // references -- see BlessedAccelStruct::setReferencedBlases) alive
        // until this dispatch's ray query finishes reading it. Only ever
        // unset when tlasValid is false, in which case the shader never
        // reads args.tlasAddress at all.
        if (args.tlasRef != nullptr)
          cmd->track(args.tlasRef);

        m_framesInWindow++;
        if (!args.tlasValid)
          m_invalidTlasInWindow++;

        maybeFlushTimingWindow();
      }

    private:

      DxvkDevice*               m_device;
      Rc<DxvkSampler>           m_sampler;
      const DxvkPipelineLayout* m_layout   = nullptr;
      VkPipeline                m_pipeline = VK_NULL_HANDLE;

      // blessed: BLESSED_SHADOW_SOFT=1 -- lazily built, see dispatch() below
      std::unique_ptr<BlessedSoftShadowState> m_soft;

      VkQueryPool m_queryPool     = VK_NULL_HANDLE;
      uint32_t    m_queryRingNext = 0u; // next pair of queries to reset+write
      bool        m_queryPairUsed[TimestampQueryCount / 2u] = { };

      uint32_t m_framesInWindow       = 0u;
      uint32_t m_invalidTlasInWindow  = 0u;
      double   m_gpuMsSumInWindow     = 0.0;
      uint32_t m_gpuMsSamplesInWindow = 0u;

      dxvk::high_resolution_clock::time_point m_processStart;
      std::ofstream m_timingFile;
      bool          m_timingFileTried = false;

      // blessed: writes the "before" timestamp for this dispatch into the
      // next ring slot, first collecting whatever the slot held two
      // dispatches ago (non-blocking: skipped if not yet available, which
      // only ever drops a single sample under normal frame pacing).
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

      // blessed: host-side, non-blocking; only called right before a slot's
      // queries are about to be reused, i.e. at least one full ring cycle
      // (2 dispatches) after they were written -- ready by then in practice.
      void collectTiming(uint32_t base) {
        struct { uint64_t ts; uint64_t avail; } results[2] = { };

        VkResult vr = m_device->vkd()->vkGetQueryPoolResults(
          m_device->handle(), m_queryPool, base, 2u,
          sizeof(results), results, sizeof(results[0]),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

        if (vr != VK_SUCCESS || !results[0].avail || !results[1].avail)
          return;

        double period = double(m_device->properties().core.properties.limits.timestampPeriod);
        double ns = double(results[1].ts - results[0].ts) * period;

        m_gpuMsSumInWindow += ns / 1.0e6;
        m_gpuMsSamplesInWindow++;
      }

      void maybeFlushTimingWindow() {
        if (m_framesInWindow < 120u)
          return;

        if (!m_timingFileTried) {
          m_timingFileTried = true;
          std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
          if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            m_timingFile.open(dir + env::PlatformDirSlash + "rtshadow.jsonl",
              std::ios::out | std::ios::app);
          }
        }

        if (m_timingFile.is_open()) {
          double t = std::chrono::duration<double>(
            dxvk::high_resolution_clock::now() - m_processStart).count();
          double avgMs = m_gpuMsSamplesInWindow > 0
            ? m_gpuMsSumInWindow / double(m_gpuMsSamplesInWindow) : 0.0;

          m_timingFile << str::format("{\"t\":", t,
            ",\"frames\":", m_framesInWindow,
            ",\"gpu_ms\":", avgMs,
            ",\"invalid_tlas\":", m_invalidTlasInWindow, "}\n");
          m_timingFile.flush();
        }

        m_framesInWindow      = 0u;
        m_invalidTlasInWindow = 0u;
        m_gpuMsSumInWindow     = 0.0;
        m_gpuMsSamplesInWindow = 0u;
      }

      // blessed: BLESSED_SHADOW_DUMP -- deliberately blocking (waitForIdle),
      // fires on exactly one frame, writes a .pgm and logs the counters.
      // Never called on the hot path.
      void blockingReadback(
              DxvkContext*                ctx,
        const Rc<DxvkCommandList>&        cmd,
        const BlessedShadowDispatchArgs&  args,
        const Rc<DxvkBuffer>&             debugBuffer,
              VkDeviceSize                debugSize) {
        // Ensure every write this dispatch made (including the atomics and
        // the pixel dump) is visible to the host before we read it back.
        VkMemoryBarrier2 hostBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        hostBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        hostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        hostBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &hostBarrier;

        // blessed: raw
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

        Rc<DxvkCommandList> finished = ctx->endRecording(nullptr);

        DxvkSubmitStatus status = { };
        m_device->submitCommandList(finished, nullptr, 0u, &status);
        m_device->waitForSubmission(&status);
        m_device->waitForIdle();

        ctx->beginRecording(m_device->createCommandList());

        const uint8_t* base = reinterpret_cast<const uint8_t*>(debugBuffer->getSliceInfo().mapPtr);
        const uint32_t* counters = reinterpret_cast<const uint32_t*>(base);
        const uint32_t* pixels   = reinterpret_cast<const uint32_t*>(base + 16);

        Logger::info(str::format("blessed: rtshadow dump: ", args.width, "x", args.height,
          " tlas_valid=", args.tlasValid ? 1 : 0,
          " sky=", counters[0], " lit=", counters[1], " shadowed=", counters[2]));

        std::error_code ec;
        std::filesystem::create_directories(args.dumpPath.substr(0,
          args.dumpPath.find_last_of("/\\")), ec);

        std::ofstream pgm(args.dumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!pgm.is_open()) {
          Logger::warn(str::format("blessed: rtshadow dump: could not open '", args.dumpPath, "'"));
          return;
        }

        pgm << "P5\n" << args.width << " " << args.height << "\n255\n";

        std::vector<uint8_t> row(args.width);
        for (uint32_t y = 0; y < args.height; y++) {
          for (uint32_t x = 0; x < args.width; x++)
            row[x] = uint8_t(std::min<uint32_t>(pixels[y * args.width + x], 255u));
          pgm.write(reinterpret_cast<const char*>(row.data()), row.size());
        }

        (void) debugSize;
      }
    };

    BlessedShadowObjects* Instance(DxvkDevice* device) {
      static BlessedShadowObjects s_instance(device);
      return &s_instance;
    }

  }


  void DxvkContext::blessedRunShadowPass(
    const BlessedShadowDispatchArgs& argsIn) {
    if (!argsIn.outputStorageCapable || argsIn.outputView == nullptr || argsIn.depthView == nullptr)
      return;

    if (!m_device->supportsRayQuery())
      return;

    // blessed: resolve the tlas HERE, on the cs thread, at the instant this
    // dispatch actually runs -- not on the app thread when the draw was
    // recorded (that used to be BlessedShadow::OnDraw calling
    // BlessedScene::currentFrame() directly; see the struct comment on
    // BlessedShadowDispatchArgs). currentFrame() ping-pongs between two
    // tlas slots that only ever change inside BlessedScene::endFrame(),
    // itself only ever run on this same cs thread from D3D11SwapChain::
    // Present's EmitCs -- so reading it from here, at dispatch time, is
    // reading it in true cs-stream order relative to the endFrame() that
    // most recently ran, instead of a stale app-thread guess.
    BlessedShadowDispatchArgs args = argsIn;

    BlessedSceneFrame scene = { };
    if (BlessedScene* blessedScene = m_device->blessedScene())
      scene = blessedScene->currentFrame();

    args.tlasAddress = scene.tlasAddress;
    args.tlasValid   = scene.valid;
    // blessed: carries the tlas object itself into dispatch() below so it
    // can track it (BlessedShadowObjects::dispatch's cmd->track(args.tlasRef))
    // and keep it -- and every blas it references -- alive until this
    // dispatch's GPU work is done, independent of the scene cache's own
    // eviction or the ping-pong slot getting rebuilt out from under it.
    args.tlasRef     = scene.tlas;

    // blessed: BLESSED_SHADOW_CAMPOS=off -- no camera shift between the
    // tlas frame and this one (right for a static camera; also the
    // fallback when no camPos slot is configured on either side).
    static const bool s_noCamShift = env::getEnvVar("BLESSED_SHADOW_CAMPOS") == "off";
    if (s_noCamShift) {
      args.camDelta[0] = args.camDelta[1] = args.camDelta[2] = 0.0f;
    } else {
      args.camDelta[0] = args.camPosNow[0] - scene.camPos[0];
      args.camDelta[1] = args.camPosNow[1] - scene.camPos[1];
      args.camDelta[2] = args.camPosNow[2] - scene.camPos[2];
    }

    // blessed: tag the tlas this frame will build with this frame's camera
    // (see BlessedScene::noteCamera)
    if (BlessedScene* taggedScene = m_device->blessedScene())
      taggedScene->noteCamera(args.camPosNow);

    // blessed: one-shot diagnostic, mirrors the app-side dump-frame log
    // that used to print this -- moved here because scene is now resolved
    // on this thread, not that one.
    static uint64_t s_frameCounter = 0;
    if (++s_frameCounter == 120) {
      Logger::info(str::format("blessed: rtshadow tlas cam=(", scene.camPos[0], ",", scene.camPos[1], ",",
        scene.camPos[2], ") now=(", args.camPosNow[0], ",", args.camPosNow[1], ",", args.camPosNow[2],
        ") delta=(", args.camDelta[0], ",", args.camDelta[1], ",", args.camDelta[2],
        ") instances=", scene.instanceCount, " valid=", scene.valid ? 1 : 0));
    }

    // blessed: BlessedShadowObjects::dispatch is not a DxvkContext member and
    // endCurrentPass is private, so this has to happen here.
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

    BlessedShadowObjects* objects = Instance(m_device.ptr());
    objects->dispatch(this, m_cmd, args);

    // blessed: state-audit -- the pass bound its own compute pipeline,
    // descriptors and push data raw; dxvk still thinks its last pipeline is
    // bound and would skip the rebind for a same-shader dispatch. Same
    // contract as upstream's meta ops.
    this->invalidateState();
  }

}
