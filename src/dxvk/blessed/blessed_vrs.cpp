// blessed: vrs -- injected variable rate shading on skyrim's lit pass (pass 115). See blessed_vrs.h and docs/research/vrs-injection.md.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "blessed_vrs.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"
#include "../dxvk_device.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"

#include <blessed_vrs_rate.h>
#include <blessed_vrs_tint.h>

namespace dxvk {

  namespace {

    // blessed: matches BlessedVrsRatePush in blessed_vrs_rate.comp
    struct BlessedVrsRatePush {
      uint32_t width;
      uint32_t height;
      uint32_t samplerIndex;
      float    cutoff;
      float    motionFactor;
      uint32_t flags;
    };

    // blessed: matches BlessedVrsTintPush in blessed_vrs_tint.comp
    struct BlessedVrsTintPush {
      uint32_t width;
      uint32_t height;
      uint32_t texelWidth;
      uint32_t texelHeight;
    };

    // blessed: the rate texel for a constant mode, (log2 w << 2) | log2 h
    uint32_t ConstRate(BlessedVrsMode mode) {
      return mode == BlessedVrsMode::Const2x2 ? 0x5u : 0x0u;
    }

    float EnvFloat(const char* name, float fallback, bool allowNegative) {
      std::string s = env::getEnvVar(name);

      if (s.empty())
        return fallback;

      char* end = nullptr;
      float v = std::strtof(s.c_str(), &end);
      return (end != s.c_str() && (allowNegative || v >= 0.0f)) ? v : fallback;
    }

    void GlobalBarrier(
      const Rc<DxvkCommandList>&   cmd,
            VkPipelineStageFlags2  srcStages,
            VkAccessFlags2         srcAccess,
            VkPipelineStageFlags2  dstStages,
            VkAccessFlags2         dstAccess) {
      VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      barrier.srcStageMask  = srcStages;
      barrier.srcAccessMask = srcAccess;
      barrier.dstStageMask  = dstStages;
      barrier.dstAccessMask = dstAccess;

      VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      dep.memoryBarrierCount = 1u;
      dep.pMemoryBarriers    = &barrier;

      cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
    }

  }


  BlessedVrsMode BlessedVrs::mode() {
    static const BlessedVrsMode s_mode = [] {
      std::string s = env::getEnvVar("BLESSED_VRS");

      if (s == "adaptive")
        return BlessedVrsMode::Adaptive;
      if (s == "1x1")
        return BlessedVrsMode::Const1x1;
      if (s == "2x2")
        return BlessedVrsMode::Const2x2;

      return BlessedVrsMode::Off;
    } ();

    return s_mode;
  }


  bool BlessedVrs::debugRate() {
    static const bool s_value = env::getEnvVar("BLESSED_VRS_DEBUG") == "rate";
    return s_value;
  }


  bool BlessedVrs::allDraws() {
    static const bool s_value = env::getEnvVar("BLESSED_VRS_ALLDRAWS") == "1";
    return s_value;
  }


  float BlessedVrs::cutoff() {
    // blessed: negative is allowed and keeps every tile at 1x1 while the
    // analysis still runs: adaptive at a negative cutoff vs 1x1 is the
    // analysis's own cost
    static const float s_value = EnvFloat("BLESSED_VRS_CUTOFF", 0.015f, true);
    return s_value;
  }


  float BlessedVrs::motionFactor() {
    static const float s_value = EnvFloat("BLESSED_VRS_MOTION", 0.01f, false);
    return s_value;
  }


  bool BlessedVrs::hysteresis() {
    static const bool s_value = env::getEnvVar("BLESSED_VRS_HYSTERESIS") != "0";
    return s_value;
  }


  const char* BlessedVrs::modeName(BlessedVrsMode mode) {
    switch (mode) {
      case BlessedVrsMode::Const1x1: return "1x1";
      case BlessedVrsMode::Const2x2: return "2x2";
      case BlessedVrsMode::Adaptive: return "adaptive";
      default:                       return "off";
    }
  }


  BlessedVrsState::BlessedVrsState(DxvkDevice* dev)
  : device(dev) {
    mode         = BlessedVrs::mode();
    debugRate    = BlessedVrs::debugRate();
    allDraws     = BlessedVrs::allDraws();
    hysteresis   = BlessedVrs::hysteresis();
    cutoff       = BlessedVrs::cutoff();
    motionFactor = BlessedVrs::motionFactor();

    // blessed: 16x16 when the device allows it (the research's assumption
    // and the analysis shader's tile); otherwise the closest legal size,
    // with the constant modes only
    const auto& props = device->properties().khrFragmentShadingRate;
    const VkExtent2D lo = props.minFragmentShadingRateAttachmentTexelSize;
    const VkExtent2D hi = props.maxFragmentShadingRateAttachmentTexelSize;

    texelSize.width  = std::clamp(16u, std::max(lo.width, 1u),  std::max(hi.width, 1u));
    texelSize.height = std::clamp(16u, std::max(lo.height, 1u), std::max(hi.height, 1u));
    adaptiveOk = texelSize.width == 16u && texelSize.height == 16u;

    static bool s_logged = false;

    if (!std::exchange(s_logged, true)) {
      Logger::info(str::format("blessed: vrs: mode=", BlessedVrs::modeName(mode),
        " texel=", texelSize.width, "x", texelSize.height,
        " maxFragment=", props.maxFragmentSize.width, "x", props.maxFragmentSize.height,
        " debug=", debugRate ? "rate" : "off",
        " alldraws=", allDraws ? 1 : 0,
        " cutoff=", cutoff, " motion=", motionFactor,
        " hysteresis=", hysteresis ? 1 : 0));

      if (mode == BlessedVrsMode::Adaptive && !adaptiveOk)
        Logger::warn("blessed: vrs: rate texel is not 16x16, adaptive falls back to 1x1");
    }
  }


  BlessedVrsState::~BlessedVrsState() {
    auto vk = device->vkd();

    for (uint32_t i = 0; i < 2; i++) {
      if (rateAttachmentView[i])
        vk->vkDestroyImageView(vk->device(), rateAttachmentView[i], nullptr);
    }

    for (VkImageView view : retiredViews)
      vk->vkDestroyImageView(vk->device(), view, nullptr);
  }


  void DxvkContext::blessedVrsCreate() {
    if (!m_device->features().khrFragmentShadingRate.attachmentFragmentShadingRate)
      return;

    // blessed: the state exists whenever the feature is on, even if nothing
    // can be attached: every pipeline has the rate as dynamic state, so
    // every render pass still needs it set (1x1, KEEP)
    m_blessedVrs = new BlessedVrsState(m_device.ptr());

    // blessed: r8_uint must be a rate attachment (the spec requires it) and
    // a storage image (the analysis writes it)
    constexpr VkFormatFeatureFlags2 needed = VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR
                                           | VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;

    if ((m_device->getFormatFeatures(VK_FORMAT_R8_UINT).optimal & needed) != needed) {
      Logger::warn("blessed: vrs: r8_uint is not a storage + rate attachment format here, nothing is attached");
      m_blessedVrs->attachOk = false;
    }
  }


  void DxvkContext::blessedVrsDestroy() {
    // blessed: the context outlives its last submission's tracking, and
    // the device waits for idle before contexts go away
    delete std::exchange(m_blessedVrs, nullptr);
  }


  void DxvkContext::blessedVrsUpdateDraw() {
    BlessedVrsState* st = m_blessedVrs;

    VkFragmentShadingRateCombinerOpKHR op = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;

    if (st->passMatched) {
      const auto& fs = m_state.gp.shaders.fs;
      bool fullRate = !st->allDraws && fs != nullptr && fs->blessedVrsFullRate();

      op = fullRate
        ? VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR
        : VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;

      if (fullRate)
        st->fullDraws++;
      else
        st->coarseDraws++;
    }

    // blessed: every pipeline has the rate as dynamic state, so it must be
    // set once per render pass (VUID-vkCmdDraw-pipelineFragmentShadingRate-09238)
    // and again when the op changes
    if (st->drawDirty || op != st->drawOp) {
      m_cmd->cmdSetFragmentShadingRate(op);
      st->drawOp = op;
      st->drawDirty = false;
    }
  }


  void DxvkContext::blessedVrsPrePass() {
    BlessedVrsState* st = m_blessedVrs;

    // blessed: a new render pass instance -- the combiner must be set
    // again before its first draw
    st->drawDirty = true;

    const DxvkFramebufferInfo& fb = m_state.om.framebufferInfo;
    const DxvkFramebufferSize fbSize = fb.size();

    // blessed: the lit pass, by its attachments, not its index: rgba16f
    // colour, rg16f motion, rgba8 ssr normal, rg8 ssr mask, plus depth,
    // single-sampled, one layer. Tilers (secondary command buffers) are
    // never matched.
    bool match = st->attachOk
              && !m_device->perfHints().preferRenderPassOps
              && fbSize.layers == 1u
              && fb.getDepthTarget().view != nullptr
              && fb.getDepthTarget().shadow == nullptr
              && (fb.getDepthTarget().view->info().aspects & VK_IMAGE_ASPECT_DEPTH_BIT)
              && fb.getDepthTarget().view->image()->info().sampleCount == VK_SAMPLE_COUNT_1_BIT;

    for (uint32_t i = 0; i < MaxNumRenderTargets && match; i++) {
      const auto& rt = fb.getColorTarget(i);

      if (i >= 4u) {
        match = rt.view == nullptr;
        continue;
      }

      if (rt.view == nullptr || rt.shadow != nullptr
       || rt.view->image()->info().sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        match = false;
        continue;
      }

      VkFormat format = rt.view->info().format;

      switch (i) {
        case 0u: match = format == VK_FORMAT_R16G16B16A16_SFLOAT; break;
        case 1u: match = format == VK_FORMAT_R16G16_SFLOAT; break;
        case 2u: match = format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB; break;
        case 3u: match = format == VK_FORMAT_R8G8_UNORM; break;
      }
    }

    // blessed: one lit size at a time. The largest seen wins; a smaller
    // one takes over only after 300 matches in a row without the current
    // size (a resolution change), so a smaller pass with the same formats
    // cannot make the images flip every frame.
    if (match) {
      VkExtent2D size = { fbSize.width, fbSize.height };

      if (size.width != st->litSize.width || size.height != st->litSize.height) {
        uint64_t area    = uint64_t(size.width) * uint64_t(size.height);
        uint64_t curArea = uint64_t(st->litSize.width) * uint64_t(st->litSize.height);

        if (area > curArea || ++st->litMissCount > 300u) {
          st->litSize = size;
          st->litMissCount = 0u;
        } else {
          match = false;
        }
      } else {
        st->litMissCount = 0u;
      }
    }

    // blessed: the analysis itself runs from bindRenderTargets (see
    // blessedVrsTargetsChanged): this is the middle of a draw, where
    // invalidating dxvk's pipeline state is not allowed
    st->passMatched = match;

    if (!match)
      return;

    VkExtent2D tiles = {
      (st->litSize.width  + st->texelSize.width  - 1u) / st->texelSize.width,
      (st->litSize.height + st->texelSize.height - 1u) / st->texelSize.height };

    if (tiles.width != st->tileExtent.width || tiles.height != st->tileExtent.height)
      blessedVrsAllocate(tiles);

    if (!st->matchedPasses++) {
      Logger::info(str::format("blessed: vrs: lit pass matched at ", st->litSize.width, "x", st->litSize.height,
        ", rate image ", tiles.width, "x", tiles.height));
    }

    bool adaptive = st->mode == BlessedVrsMode::Adaptive && st->adaptiveOk;

    if (adaptive || st->debugRate) {
      st->pendingAnalyze  = true;
      st->litColor        = fb.getColorTarget(0).view->image();
      st->litMotion       = fb.getColorTarget(1).view->image();
      st->litColorFormat  = fb.getColorTarget(0).view->info().format;
      st->litMotionFormat = fb.getColorTarget(1).view->info().format;
    }
  }


  void DxvkContext::blessedVrsAllocate(VkExtent2D tiles) {
    BlessedVrsState* st = m_blessedVrs;
    auto vk = m_device->vkd();

    // blessed: dxvk's batched barriers go ahead of our raw ones
    flushBarriers();

    for (uint32_t i = 0; i < 2; i++) {
      if (st->rateAttachmentView[i])
        st->retiredViews.push_back(std::exchange(st->rateAttachmentView[i], VK_NULL_HANDLE));
      if (st->rateImage[i] != nullptr)
        st->retiredImages.push_back(std::move(st->rateImage[i]));
    }

    DxvkImageCreateInfo info = { };
    info.type        = VK_IMAGE_TYPE_2D;
    info.format      = VK_FORMAT_R8_UINT;
    info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    info.extent      = { tiles.width, tiles.height, 1u };
    info.numLayers   = 1u;
    info.mipLevels   = 1u;
    info.usage       = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR
                     | VK_IMAGE_USAGE_STORAGE_BIT
                     | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                     | VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR
                     | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access      = VK_ACCESS_SHADER_READ_BIT
                     | VK_ACCESS_SHADER_WRITE_BIT
                     | VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR
                     | VK_ACCESS_TRANSFER_WRITE_BIT;
    info.tiling      = VK_IMAGE_TILING_OPTIMAL;
    info.layout      = VK_IMAGE_LAYOUT_GENERAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.debugName   = "blessed vrs rate";

    DxvkImageCreateInfo historyInfo = info;
    historyInfo.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    historyInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    historyInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    historyInfo.debugName = "blessed vrs history";

    DxvkImageViewKey storageKey = { };
    storageKey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    storageKey.usage      = VK_IMAGE_USAGE_STORAGE_BIT;
    storageKey.format     = VK_FORMAT_R8_UINT;
    storageKey.layout     = VK_IMAGE_LAYOUT_GENERAL;
    storageKey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    storageKey.mipIndex   = 0u;
    storageKey.mipCount   = 1u;
    storageKey.layerIndex = 0u;
    storageKey.layerCount = 1u;

    if (st->historyImage != nullptr)
      st->retiredImages.push_back(std::move(st->historyImage));

    st->historyImage = m_device->createImage(historyInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    st->historyStorageView = st->historyImage->createView(storageKey);

    std::array<DxvkImage*, 3> images = { };

    for (uint32_t i = 0; i < 2; i++) {
      st->rateImage[i] = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      st->rateStorageView[i] = st->rateImage[i]->createView(storageKey);

      // blessed: raw view -- no VkImageViewUsageCreateInfo, so it inherits
      // the image's fragment shading rate usage
      VkImageViewCreateInfo viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
      viewInfo.image    = st->rateImage[i]->handle();
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format   = VK_FORMAT_R8_UINT;
      viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u };

      if (vk->vkCreateImageView(vk->device(), &viewInfo, nullptr, &st->rateAttachmentView[i]))
        throw DxvkError("blessed: vrs: failed to create rate attachment view");

      images[i] = st->rateImage[i].ptr();
    }

    images[2] = st->historyImage.ptr();

    // blessed: one-time UNDEFINED -> GENERAL; these images stay in GENERAL
    // for life (legal for storage and for the rate attachment alike)
    std::array<VkImageMemoryBarrier2, 3> barriers = { };

    for (uint32_t i = 0; i < 3; i++) {
      auto& b = barriers[i];
      b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
      b.srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
      b.srcAccessMask       = VK_ACCESS_2_NONE;
      b.dstStageMask        = VK_PIPELINE_STAGE_2_CLEAR_BIT;
      b.dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
      b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
      b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.image               = images[i]->handle();
      b.subresourceRange    = images[i]->getAvailableSubresources();
    }

    VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = uint32_t(barriers.size());
    dep.pImageMemoryBarriers    = barriers.data();
    m_cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

    // blessed: constant modes hold their rate for life; adaptive (and the
    // 1x1 fallback) starts at 1x1 until the first analysis
    VkClearColorValue rateValue = { };
    rateValue.uint32[0] = ConstRate(st->mode);

    VkClearColorValue zero = { };
    VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u };

    for (uint32_t i = 0; i < 3; i++) {
      m_cmd->cmdClearColorImage(DxvkCmdBuffer::ExecBuffer, images[i]->handle(),
        VK_IMAGE_LAYOUT_GENERAL, i < 2u ? &rateValue : &zero, 1u, &range);

      images[i]->trackLayout(images[i]->getAvailableSubresources(), VK_IMAGE_LAYOUT_GENERAL);
    }

    GlobalBarrier(m_cmd,
      VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    m_cmd->track(st->rateImage[0], DxvkAccess::Write);
    m_cmd->track(st->rateImage[1], DxvkAccess::Write);
    m_cmd->track(st->historyImage, DxvkAccess::Write);

    st->tileExtent   = tiles;
    st->rateCur      = 0u;
    st->historyValid = false;

    Logger::info(str::format("blessed: vrs: (re)allocated rate images at ", tiles.width, "x", tiles.height,
      " (", BlessedVrs::modeName(st->mode), ")"));
  }


  void DxvkContext::blessedVrsTargetsChanged(const DxvkRenderTargets& targets) {
    BlessedVrsState* st = m_blessedVrs;

    if (!st->pendingAnalyze)
      return;

    // blessed: the same colour and motion images in slots 0 and 1 means the
    // lit pass goes on (a redundant bind, or its depth changed)
    if (targets.color[0].view != nullptr && targets.color[0].view->image() == st->litColor.ptr()
     && targets.color[1].view != nullptr && targets.color[1].view->image() == st->litMotion.ptr())
      return;

    st->pendingAnalyze = false;

    // blessed: top level (a cs command), outside any draw: ending the pass
    // and invalidating dxvk's pipeline state are both allowed here
    endCurrentPass(true);
    blessedVrsEndOfLitPass();
  }


  void DxvkContext::blessedVrsEndOfLitPass() {
    BlessedVrsState* st = m_blessedVrs;

    Rc<DxvkImage> color  = std::move(st->litColor);
    Rc<DxvkImage> motion = std::move(st->litMotion);

    if (color == nullptr || motion == nullptr || st->rateImage[0] == nullptr)
      return;

    VkExtent3D colorExtent = color->info().extent;

    if (colorExtent.width != st->litSize.width || colorExtent.height != st->litSize.height
     || motion->info().extent.width != st->litSize.width || motion->info().extent.height != st->litSize.height)
      return;

    // blessed: our raw compute binds follow; dxvk rebinds its own state
    // afterwards (invalidateState at the end)
    endComputePass();

    if (st->pointSampler == nullptr) {
      DxvkSamplerKey samplerKey = { };
      samplerKey.setFilter(VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST);
      samplerKey.setAddressModes(
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
      samplerKey.setUsePixelCoordinates(false);
      st->pointSampler = m_device->createSampler(samplerKey);

      static const std::array<DxvkDescriptorSetLayoutBinding, 4> rateBindings = {{
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
      }};

      st->rateLayout = m_device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlag::UsesSamplerHeap,
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedVrsRatePush),
        uint32_t(rateBindings.size()), rateBindings.data());

      util::DxvkBuiltInShaderStage rateShader(blessed_vrs_rate, nullptr);
      st->ratePipeline = m_device->createBuiltInComputePipeline(st->rateLayout, rateShader);

      static const std::array<DxvkDescriptorSetLayoutBinding, 2> tintBindings = {{
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT },
      }};

      st->tintLayout = m_device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlags(),
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BlessedVrsTintPush),
        uint32_t(tintBindings.size()), tintBindings.data());

      util::DxvkBuiltInShaderStage tintShader(blessed_vrs_tint, nullptr);
      st->tintPipeline = m_device->createBuiltInComputePipeline(st->tintLayout, tintShader);
    }

    bool rawBinds = false;

    // blessed: the rate image pass 115 read this frame
    const uint32_t usedIdx = st->rateCur;

    // blessed: the analysis -- next frame's rates into rateImage[1 - rateCur]
    if (st->mode == BlessedVrsMode::Adaptive && st->adaptiveOk && st->ratePipeline) {
      bool sampled = (color->info().usage & VK_IMAGE_USAGE_SAMPLED_BIT)
                  && (motion->info().usage & VK_IMAGE_USAGE_SAMPLED_BIT);

      if (!sampled) {
        if (!std::exchange(st->warnedAnalyze, true))
          Logger::warn("blessed: vrs: lit pass colour or motion is not sampled-capable; adaptive rates stay as they are");
      } else {
        DxvkImageViewKey viewKey = { };
        viewKey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
        viewKey.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
        viewKey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
        viewKey.mipIndex   = 0u;
        viewKey.mipCount   = 1u;
        viewKey.layerIndex = 0u;
        viewKey.layerCount = 1u;

        viewKey.format = st->litColorFormat;
        Rc<DxvkImageView> colorView = color->createView(viewKey);

        viewKey.format = st->litMotionFormat;
        Rc<DxvkImageView> motionView = motion->createView(viewKey);

        std::array<DxvkResourceAccess, 2> access = {{
          { *colorView,  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, false },
          { *motionView, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, false },
        }};

        syncResources(DxvkCmdBuffer::ExecBuffer, access.size(), access.data());
        flushBarriers();

        uint32_t next = 1u - st->rateCur;

        // blessed: rateImage[next] was last read as the attachment one frame
        // ago; the history was written by the last analysis
        GlobalBarrier(m_cmd,
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

        std::array<DxvkDescriptorWrite, 4> descriptors = { };
        descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        descriptors[0].descriptor     = colorView->getDescriptor();
        descriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        descriptors[1].descriptor     = motionView->getDescriptor();
        descriptors[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptors[2].descriptor     = st->rateStorageView[next]->getDescriptor();
        descriptors[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptors[3].descriptor     = st->historyStorageView->getDescriptor();

        BlessedVrsRatePush push = { };
        push.width        = colorExtent.width;
        push.height       = colorExtent.height;
        push.samplerIndex = st->pointSampler->getDescriptor().samplerIndex;
        push.cutoff       = st->cutoff;
        push.motionFactor = st->motionFactor;
        push.flags        = (st->hysteresis ? 1u : 0u) | (st->historyValid ? 2u : 0u);

        m_cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, st->ratePipeline);
        m_cmd->bindResources(DxvkCmdBuffer::ExecBuffer, st->rateLayout,
          uint32_t(descriptors.size()), descriptors.data(), sizeof(push), &push);
        m_cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, st->tileExtent.width, st->tileExtent.height, 1u);

        // blessed: the next lit pass reads it as its attachment, the next
        // debug tint as a storage image
        GlobalBarrier(m_cmd,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);

        m_cmd->track(st->rateImage[next], DxvkAccess::Write);
        m_cmd->track(st->historyImage, DxvkAccess::Write);

        st->rateCur = next;
        st->historyValid = true;
        rawBinds = true;

        if (!(st->analyses++ % 1200u)) {
          Logger::info(str::format("blessed: vrs: analyses=", st->analyses,
            " lit passes=", st->matchedPasses,
            " coarse draws=", st->coarseDraws, " full-rate draws=", st->fullDraws));
        }
      }
    }

    // blessed: the debug tint, after the analysis read the clean colour --
    // it shows the rates pass 115 used this frame (rateImage[usedIdx]).
    // The colour target is not storage-capable, so: copy it out, tint the
    // copy, copy it back. Debug only.
    if (st->debugRate) {
      constexpr VkImageUsageFlags copyUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

      bool tintOk = st->litColorFormat == VK_FORMAT_R16G16B16A16_SFLOAT
                 && color->info().format == VK_FORMAT_R16G16B16A16_SFLOAT
                 && (color->info().usage & copyUsage) == copyUsage
                 && st->tintPipeline;

      if (!tintOk) {
        if (!std::exchange(st->warnedTint, true))
          Logger::warn("blessed: vrs: debug tint needs an rgba16f colour target with transfer usage; tint off");
      } else {
        if (st->tintImage == nullptr
         || st->tintImage->info().extent.width  != colorExtent.width
         || st->tintImage->info().extent.height != colorExtent.height) {
          DxvkImageCreateInfo tintInfo = { };
          tintInfo.type        = VK_IMAGE_TYPE_2D;
          tintInfo.format      = VK_FORMAT_R16G16B16A16_SFLOAT;
          tintInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
          tintInfo.extent      = { colorExtent.width, colorExtent.height, 1u };
          tintInfo.numLayers   = 1u;
          tintInfo.mipLevels   = 1u;
          tintInfo.usage       = VK_IMAGE_USAGE_STORAGE_BIT | copyUsage;
          tintInfo.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
          tintInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                               | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
          tintInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
          tintInfo.layout      = VK_IMAGE_LAYOUT_GENERAL;
          tintInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          tintInfo.debugName   = "blessed vrs tint";

          st->tintImage = m_device->createImage(tintInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

          DxvkImageViewKey tintKey = { };
          tintKey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
          tintKey.usage      = VK_IMAGE_USAGE_STORAGE_BIT;
          tintKey.format     = VK_FORMAT_R16G16B16A16_SFLOAT;
          tintKey.layout     = VK_IMAGE_LAYOUT_GENERAL;
          tintKey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
          tintKey.mipIndex   = 0u;
          tintKey.mipCount   = 1u;
          tintKey.layerIndex = 0u;
          tintKey.layerCount = 1u;

          st->tintStorageView = st->tintImage->createView(tintKey);
        }

        VkImageSubresourceLayers layers = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u };
        VkExtent3D extent = { colorExtent.width, colorExtent.height, 1u };

        copyImage(st->tintImage, layers, VkOffset3D { 0, 0, 0 }, color, layers, VkOffset3D { 0, 0, 0 }, extent);

        endCurrentPass(true);

        DxvkResourceAccess tintAccess(*st->tintStorageView, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, false);
        syncResources(DxvkCmdBuffer::ExecBuffer, 1u, &tintAccess);
        flushBarriers();

        std::array<DxvkDescriptorWrite, 2> descriptors = { };
        descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptors[0].descriptor     = st->tintStorageView->getDescriptor();
        descriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptors[1].descriptor     = st->rateStorageView[usedIdx]->getDescriptor();

        BlessedVrsTintPush push = { };
        push.width       = colorExtent.width;
        push.height      = colorExtent.height;
        push.texelWidth  = st->texelSize.width;
        push.texelHeight = st->texelSize.height;

        m_cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, st->tintPipeline);
        m_cmd->bindResources(DxvkCmdBuffer::ExecBuffer, st->tintLayout,
          uint32_t(descriptors.size()), descriptors.data(), sizeof(push), &push);
        m_cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer,
          (colorExtent.width + 7u) / 8u, (colorExtent.height + 7u) / 8u, 1u);

        m_cmd->track(st->rateImage[usedIdx], DxvkAccess::Read);
        rawBinds = true;

        // blessed: dxvk saw the tint image's compute write in syncResources
        // above, so the copy back waits for it
        invalidateState();
        copyImage(color, layers, VkOffset3D { 0, 0, 0 }, st->tintImage, layers, VkOffset3D { 0, 0, 0 }, extent);
        endCurrentPass(true);
      }
    }

    if (rawBinds)
      invalidateState();
  }


  void DxvkContext::blessedVrsBeginRendering(VkRenderingInfo& renderingInfo) {
    BlessedVrsState* st = m_blessedVrs;

    if (!st->passMatched)
      return;

    auto& info = st->attachmentInfo;
    info = { VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR };
    info.imageView   = st->rateAttachmentView[st->rateCur];
    info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    info.shadingRateAttachmentTexelSize = st->texelSize;
    info.pNext = std::exchange(renderingInfo.pNext, &info);

    m_cmd->track(st->rateImage[st->rateCur], DxvkAccess::Read);
  }

}
