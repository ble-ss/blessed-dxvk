// blessed: BLESSED_LOOK -- the bless colour chain as a dxvk-side post pass (images, shaders, dispatches)
#include "blessed_look.h"

#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../dxvk_shader_spirv.h"

#include "../../util/log/log.h"
#include "../../util/util_string.h"

#include <blessed_look_prep.h>
#include <blessed_look_composite.h>

namespace dxvk {

  namespace {

    // blessed: resource slots for the look's own bindings. Everything the
    // pass binds is unbound again before it returns, and the d3d11 side
    // calls RestoreCommandListState() right after, so the game's own
    // bindings in these slots come back untouched.
    constexpr uint32_t SlotSrc   = 0u;
    constexpr uint32_t SlotBloom = 1u;
    constexpr uint32_t SlotSpill = 2u;
    constexpr uint32_t SlotDst   = 3u;

    // blessed: rgba8/bgra8 targets only. The look reads them as unorm of
    // the same channel order, so the shaders see the stored (display-
    // encoded) values, which is the range bless was tuned in; returns
    // VK_FORMAT_UNDEFINED for anything else
    VkFormat LookFormatFor(VkFormat format) {
      switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        default:                      return VK_FORMAT_UNDEFINED;
      }
    }

    uint32_t DivUp(uint32_t a, uint32_t b) {
      return (a + b - 1u) / b;
    }

  }


  BlessedLookPass::BlessedLookPass(DxvkDevice* device)
  : m_device(device) {
    createShaders();

    const auto& limits = m_device->properties().core.properties.limits;
    m_timestampPeriodNs = limits.timestampPeriod;
  }


  BlessedLookPass::~BlessedLookPass() {

  }


  void BlessedLookPass::createShaders() {
    const std::array<DxvkBindingInfo, 2> prepBindings = {{
      { 0u, 0u, SlotSrc, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 1u, SlotDst, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT },
    }};

    const std::array<DxvkBindingInfo, 4> compositeBindings = {{
      { 0u, 0u, SlotSrc,   VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 1u, SlotBloom, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 2u, SlotSpill, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 3u, SlotDst,   VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT },
    }};

    DxvkSpirvShaderCreateInfo prepInfo = { };
    prepInfo.bindingCount  = uint32_t(prepBindings.size());
    prepInfo.bindings      = prepBindings.data();
    prepInfo.localPushData = DxvkPushDataBlock(0u, sizeof(BlessedLookPrepPush), sizeof(uint32_t), 0u);
    prepInfo.debugName     = "blessed look prep";
    m_prep = new DxvkSpirvShader(prepInfo, blessed_look_prep);

    DxvkSpirvShaderCreateInfo compositeInfo = { };
    compositeInfo.bindingCount  = uint32_t(compositeBindings.size());
    compositeInfo.bindings      = compositeBindings.data();
    compositeInfo.localPushData = DxvkPushDataBlock(0u, sizeof(BlessedLookCompositePush), sizeof(uint32_t), 0u);
    compositeInfo.debugName     = "blessed look composite";
    m_composite = new DxvkSpirvShader(compositeInfo, blessed_look_composite);
  }


  BlessedLookPass::Target BlessedLookPass::createTarget(
          VkFormat        format,
          uint32_t        width,
          uint32_t        height,
    const char*           name,
          bool            storage) {
    DxvkImageCreateInfo info = { };
    info.type        = VK_IMAGE_TYPE_2D;
    info.format      = format;
    info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    info.extent      = { width, height, 1u };
    info.numLayers   = 1u;
    info.mipLevels   = 1u;
    info.usage       = VK_IMAGE_USAGE_SAMPLED_BIT
                     | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (storage)
      info.usage    |= VK_IMAGE_USAGE_STORAGE_BIT;
    info.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                     | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    info.tiling      = VK_IMAGE_TILING_OPTIMAL;
    info.layout      = VK_IMAGE_LAYOUT_GENERAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.debugName   = name;

    Target t;
    t.image = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    DxvkImageViewKey key = { };
    key.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    key.format     = format;
    key.layout     = VK_IMAGE_LAYOUT_GENERAL;
    key.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    key.mipIndex   = 0u;
    key.mipCount   = 1u;
    key.layerIndex = 0u;
    key.layerCount = 1u;

    key.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    t.sampled = t.image->createView(key);

    if (storage) {
      key.usage = VK_IMAGE_USAGE_STORAGE_BIT;
      t.storage = t.image->createView(key);
    }
    return t;
  }


  bool BlessedLookPass::ensureTargets(VkFormat format, uint32_t width, uint32_t height) {
    if (m_format == format && m_width == width && m_height == height)
      return true;

    try {
      uint32_t hw = (width  + 1u) / 2u;   // bless: (width + 1) / 2
      uint32_t hh = (height + 1u) / 2u;
      uint32_t qw = (width  + 3u) / 4u;   // bless: (width + 3) / 4
      uint32_t qh = (height + 3u) / 4u;

      // m_src: the copy-in fallback, rtv 0's channel order, sampled only.
      // m_out: always rgba8 unorm (the composite's declared storage format),
      // copied back byte for byte; the composite swaps r/b for a bgra8 rtv 0
      m_src    = createTarget(format, width, height, "blessed look src", false);
      m_out    = createTarget(VK_FORMAT_R8G8B8A8_UNORM, width, height, "blessed look out", true);
      m_bloomA = createTarget(VK_FORMAT_R16G16B16A16_SFLOAT, hw, hh, "blessed look bloom a", true);
      m_bloomB = createTarget(VK_FORMAT_R16G16B16A16_SFLOAT, hw, hh, "blessed look bloom b", true);
      m_spillA = createTarget(VK_FORMAT_R16G16B16A16_SFLOAT, qw, qh, "blessed look spill a", true);
      m_spillB = createTarget(VK_FORMAT_R16G16B16A16_SFLOAT, qw, qh, "blessed look spill b", true);
    } catch (const DxvkError& e) {
      Logger::err(str::format("BlessedLook: could not create targets (format ", uint32_t(format),
        ", ", width, "x", height, "): ", e.message(), "; look disabled"));
      m_usable = false;
      return false;
    }

    m_format = format;
    m_width  = width;
    m_height = height;

    Logger::info(str::format("BlessedLook: targets ", width, "x", height, ", format ", uint32_t(format)));
    return true;
  }


  void BlessedLookPass::prep(
          DxvkContext*          ctx,
    const Rc<DxvkImageView>&    src,
    const Target&               dst,
    const BlessedLookPrepPush&  push) {
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSrc, Rc<DxvkImageView>(src));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotDst, Rc<DxvkImageView>(dst.storage));
    ctx->pushData(VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(push), &push);
    ctx->dispatch(DivUp(uint32_t(push.dstSize[0]), 8u), DivUp(uint32_t(push.dstSize[1]), 8u), 1u);
  }


  void BlessedLookPass::readTimers() {
    uint32_t slot = (m_timerFrame % TimerRing) * 2u;

    if (m_timers[slot] == nullptr)
      return;

    DxvkQueryData begin = { };
    DxvkQueryData end   = { };

    if (m_timers[slot + 0u]->getData(begin) != DxvkGpuQueryStatus::Available
     || m_timers[slot + 1u]->getData(end)   != DxvkGpuQueryStatus::Available)
      return;

    uint64_t ticks = end.timestamp.time - begin.timestamp.time;
    m_lastGpuMs.store(float(double(ticks) * double(m_timestampPeriodNs) / 1.0e6), std::memory_order_relaxed);
  }


  void BlessedLookPass::run(DxvkContext* ctx, const BlessedLookArgs& args) {
    if (!m_usable || args.target == nullptr || !args.width || !args.height)
      return;

    VkFormat format = LookFormatFor(args.target->info().format);

    if (format == VK_FORMAT_UNDEFINED) {
      Logger::err(str::format("BlessedLook: final copy target format ", uint32_t(args.target->info().format),
        " is not rgba8/bgra8; look disabled"));
      m_usable = false;
      return;
    }

    if (!ensureTargets(format, args.width, args.height))
      return;

    uint32_t timerSlot = (m_timerFrame % TimerRing) * 2u;

    if (args.timing) {
      readTimers();

      for (uint32_t i = 0u; i < 2u; i++) {
        if (m_timers[timerSlot + i] == nullptr)
          m_timers[timerSlot + i] = m_device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0u, 0u);
      }

      ctx->writeTimestamp(m_timers[timerSlot]);
    }

    VkImageSubresourceLayers ownLayers = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u };
    VkExtent3D extent = { args.width, args.height, 1u };

    // read rtv 0 in place when it is sampleable in its own (unorm) format;
    // otherwise copy it into m_src first. Either way the composite writes
    // m_out, which is copied back, so no pass reads and writes one image.
    Rc<DxvkImageView> srcView;

    if ((args.target->info().usage & VK_IMAGE_USAGE_SAMPLED_BIT)
     && args.target->info().format == format) {
      DxvkImageViewKey key = { };
      key.viewType   = VK_IMAGE_VIEW_TYPE_2D;
      key.format     = format;
      key.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
      key.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
      key.mipIndex   = args.targetLayers.mipLevel;
      key.mipCount   = 1u;
      key.layerIndex = args.targetLayers.baseArrayLayer;
      key.layerCount = 1u;
      srcView = args.target->createView(key);
    } else {
      ctx->copyImage(m_src.image, ownLayers, VkOffset3D(),
        args.target, args.targetLayers, VkOffset3D(), extent);
      srcView = m_src.sampled;
    }

    int32_t w  = int32_t(args.width);
    int32_t h  = int32_t(args.height);
    int32_t hw = int32_t(m_bloomA.image->info().extent.width);
    int32_t hh = int32_t(m_bloomA.image->info().extent.height);
    int32_t qw = int32_t(m_spillA.image->info().extent.width);
    int32_t qh = int32_t(m_spillA.image->info().extent.height);

    ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(Rc<DxvkShader>(m_prep));

    // bloom_extract.fsh: bright pass + horizontal 9-tap, full res -> half res,
    // bloomRadius full-res pixels per tap
    BlessedLookPrepPush push = { };
    push.srcSize[0] = w;  push.srcSize[1] = h;
    push.dstSize[0] = hw; push.dstSize[1] = hh;
    push.offsetUv[0] = args.bloomRadius / float(w);
    push.offsetUv[1] = 0.0f;
    push.mode      = 0;
    push.threshold = args.bloomThreshold;
    push.knee      = args.bloomKnee;
    push.surface   = args.bloomSurface;
    push.haze      = args.bloomHaze;
    prep(ctx, srcView, m_bloomA, push);

    // bloom_vertical.fsh: the same radius in full-res pixels, on the half-res target
    push = BlessedLookPrepPush();
    push.srcSize[0] = hw; push.srcSize[1] = hh;
    push.dstSize[0] = hw; push.dstSize[1] = hh;
    push.offsetUv[0] = 0.0f;
    push.offsetUv[1] = args.bloomRadius / float(h);
    push.mode = 1;
    prep(ctx, m_bloomA.sampled, m_bloomB, push);

    // spill_extract.fsh: saturation-keyed pass, 4-tap box, full res -> quarter res
    push = BlessedLookPrepPush();
    push.srcSize[0] = w;  push.srcSize[1] = h;
    push.dstSize[0] = qw; push.dstSize[1] = qh;
    push.mode      = 2;
    push.threshold = args.spillThreshold;
    push.knee      = args.spillKnee;
    prep(ctx, srcView, m_spillA, push);

    // spill_blur.fsh h then v: spillRadius quarter-res pixels per tap
    push = BlessedLookPrepPush();
    push.srcSize[0] = qw; push.srcSize[1] = qh;
    push.dstSize[0] = qw; push.dstSize[1] = qh;
    push.offsetUv[0] = args.spillRadius / float(qw);
    push.mode = 1;
    prep(ctx, m_spillA.sampled, m_spillB, push);

    push.offsetUv[0] = 0.0f;
    push.offsetUv[1] = args.spillRadius / float(qh);
    prep(ctx, m_spillB.sampled, m_spillA, push);

    // bloom_grade.fsh: screen bloom, add spill, exposure, grain, grade
    BlessedLookCompositePush composite = args.composite;
    composite.swapRB = format == VK_FORMAT_B8G8R8A8_UNORM ? 1 : 0;
    composite.size[0]      = w;  composite.size[1]      = h;
    composite.bloomSize[0] = hw; composite.bloomSize[1] = hh;
    composite.spillSize[0] = qw; composite.spillSize[1] = qh;

    ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(Rc<DxvkShader>(m_composite));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSrc,   Rc<DxvkImageView>(srcView));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotBloom, Rc<DxvkImageView>(m_bloomB.sampled));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSpill, Rc<DxvkImageView>(m_spillA.sampled));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotDst,   Rc<DxvkImageView>(m_out.storage));
    ctx->pushData(VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(composite), &composite);
    ctx->dispatch(DivUp(args.width, 8u), DivUp(args.height, 8u), 1u);

    // leave nothing of ours bound; the d3d11 side restores its own state next
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSrc,   nullptr);
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotBloom, nullptr);
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSpill, nullptr);
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotDst,   nullptr);
    ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(nullptr);

    ctx->copyImage(args.target, args.targetLayers, VkOffset3D(),
      m_out.image, ownLayers, VkOffset3D(), extent);

    if (args.timing) {
      ctx->writeTimestamp(m_timers[timerSlot + 1u]);
      m_timerFrame++;
    }
  }

}
