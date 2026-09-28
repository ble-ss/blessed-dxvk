// blessed: half-rate far field -- the dxvk-side far layer, its capture and the prefill draws
#include "blessed_halfrate.h"

#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../dxvk_shader_spirv.h"

#include "../../util/log/log.h"
#include "../../util/util_string.h"

#include <cmath>
#include <cstring>

#include <blessed_halfrate_capture.h>
#include <blessed_halfrate_fill_vert.h>
#include <blessed_halfrate_fill_frag.h>

namespace dxvk {

  namespace {

    // resource slots, shared by the capture cs and the fill fs
    constexpr uint32_t SlotParams  = 0u;
    constexpr uint32_t SlotDepth   = 1u;
    constexpr uint32_t SlotSrcCol  = 2u;
    constexpr uint32_t SlotSrcRt2  = 3u;
    constexpr uint32_t SlotSrcRt3  = 4u;
    constexpr uint32_t SlotOutCol  = 5u;
    constexpr uint32_t SlotOutRt2  = 6u;
    constexpr uint32_t SlotOutRt3  = 7u;
    constexpr uint32_t SlotOutDep  = 8u;
    constexpr uint32_t SlotLayDep  = 5u;   // fill only: the layer depth, read

    uint32_t DivUp(uint32_t a, uint32_t b) {
      return (a + b - 1u) / b;
    }

    // blessed: general 4x4 inverse (cofactors, in double), layout-agnostic,
    // the same as blessed_soft_shadow.cpp's. False if singular.
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

  }


  BlessedHalfRatePass::BlessedHalfRatePass(DxvkDevice* device)
  : m_device(device) {
    for (auto& ms : m_lastMs)
      ms.store(-1.0f, std::memory_order_relaxed);

    m_timestampPeriodNs = m_device->properties().core.properties.limits.timestampPeriod;

    try {
      createShaders();

      DxvkBufferCreateInfo uboInfo = { };
      uboInfo.size      = sizeof(BlessedHalfRateParams);
      uboInfo.usage     = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      uboInfo.stages    = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      uboInfo.access    = VK_ACCESS_UNIFORM_READ_BIT;
      uboInfo.debugName = "blessed halfrate params";
      m_ubo = m_device->createBuffer(uboInfo,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    } catch (const DxvkError& e) {
      Logger::err(str::format("BlessedHalfRate: could not create shaders or params buffer: ", e.message()));
      m_usable.store(false, std::memory_order_relaxed);
    }
  }


  BlessedHalfRatePass::~BlessedHalfRatePass() {

  }


  void BlessedHalfRatePass::createShaders() {
    static_assert(sizeof(BlessedHalfRateParams) == 4u * 64u + 3u * 16u,
      "BlessedHalfRateParams must match the std140 block in the half-rate shaders");

    const std::array<DxvkBindingInfo, 9> captureBindings = {{
      { 0u, 0u, SlotParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_UNIFORM_READ_BIT, DxvkDescriptorFlag::UniformBuffer },
      { 0u, 1u, SlotDepth,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 2u, SlotSrcCol, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 3u, SlotSrcRt2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 4u, SlotSrcRt3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 5u, SlotOutCol, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT },
      { 0u, 6u, SlotOutRt2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT },
      { 0u, 7u, SlotOutRt3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT },
      { 0u, 8u, SlotOutDep, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT },
    }};

    const std::array<DxvkBindingInfo, 6> fillBindings = {{
      { 0u, 0u, SlotParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_UNIFORM_READ_BIT, DxvkDescriptorFlag::UniformBuffer },
      { 0u, 1u, SlotDepth,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 2u, SlotSrcCol, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 3u, SlotSrcRt2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 4u, SlotSrcRt3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 5u, SlotLayDep, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
    }};

    DxvkSpirvShaderCreateInfo csInfo = { };
    csInfo.bindingCount = uint32_t(captureBindings.size());
    csInfo.bindings     = captureBindings.data();
    csInfo.debugName    = "blessed halfrate capture";
    m_captureCs = new DxvkSpirvShader(csInfo, blessed_halfrate_capture);

    DxvkSpirvShaderCreateInfo vsInfo = { };
    vsInfo.debugName = "blessed halfrate fill vs";
    m_fillVs = new DxvkSpirvShader(vsInfo, blessed_halfrate_fill_vert);

    DxvkSpirvShaderCreateInfo fsInfo = { };
    fsInfo.bindingCount = uint32_t(fillBindings.size());
    fsInfo.bindings     = fillBindings.data();
    fsInfo.debugName    = "blessed halfrate fill fs";
    m_fillFs = new DxvkSpirvShader(fsInfo, blessed_halfrate_fill_frag);
  }


  BlessedHalfRatePass::Layer BlessedHalfRatePass::createLayer(VkFormat format, const char* name) {
    DxvkImageCreateInfo info = { };
    info.type        = VK_IMAGE_TYPE_2D;
    info.format      = format;
    info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    info.extent      = { m_width, m_height, 1u };
    info.numLayers   = 1u;
    info.mipLevels   = 1u;
    info.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    info.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    info.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    info.tiling      = VK_IMAGE_TILING_OPTIMAL;
    info.layout      = VK_IMAGE_LAYOUT_GENERAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.debugName   = name;

    Layer l;
    l.image = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

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
    l.sampled = l.image->createView(key);

    key.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    l.storage = l.image->createView(key);
    return l;
  }


  bool BlessedHalfRatePass::ensureResources(VkFormat depthFormat, uint32_t width, uint32_t height) {
    if (m_depthFormat == depthFormat && m_width == width && m_height == height)
      return true;

    m_layerValid = false;

    try {
      m_width  = width;
      m_height = height;

      DxvkImageCreateInfo info = { };
      info.type        = VK_IMAGE_TYPE_2D;
      info.format      = depthFormat;
      info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      info.extent      = { width, height, 1u };
      info.numLayers   = 1u;
      info.mipLevels   = 1u;
      info.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      info.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      info.tiling      = VK_IMAGE_TILING_OPTIMAL;
      info.layout      = VK_IMAGE_LAYOUT_GENERAL;
      info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      info.debugName   = "blessed halfrate depth copy";
      m_depthCopy = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      DxvkImageViewKey key = { };
      key.viewType   = VK_IMAGE_VIEW_TYPE_2D;
      key.format     = depthFormat;
      key.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
      key.layout     = VK_IMAGE_LAYOUT_GENERAL;
      key.aspects    = VK_IMAGE_ASPECT_DEPTH_BIT;
      key.mipIndex   = 0u;
      key.mipCount   = 1u;
      key.layerIndex = 0u;
      key.layerCount = 1u;
      m_depthCopyView = m_depthCopy->createView(key);

      m_col   = createLayer(VK_FORMAT_R16G16B16A16_SFLOAT, "blessed halfrate layer colour");
      m_rt2   = createLayer(VK_FORMAT_R8G8B8A8_UNORM,      "blessed halfrate layer rt2");
      m_rt3   = createLayer(VK_FORMAT_R8G8B8A8_UNORM,      "blessed halfrate layer rt3");
      m_depth = createLayer(VK_FORMAT_R32_SFLOAT,          "blessed halfrate layer depth");
    } catch (const DxvkError& e) {
      Logger::err(str::format("BlessedHalfRate: could not create the far layer (", width, "x", height,
        ", depth format ", uint32_t(depthFormat), "): ", e.message(), "; half-rate far field off"));
      m_usable.store(false, std::memory_order_relaxed);
      m_depthFormat = VK_FORMAT_UNDEFINED;
      return false;
    }

    m_depthFormat = depthFormat;
    Logger::info(str::format("BlessedHalfRate: far layer ", width, "x", height,
      ", depth format ", uint32_t(depthFormat)));
    return true;
  }


  bool BlessedHalfRatePass::checkTargets(const BlessedHalfRateTargets& t) const {
    static const std::array<VkFormat, 4> expected = {{
      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT,
      VK_FORMAT_R8G8B8A8_UNORM,      VK_FORMAT_R8G8_UNORM }};

    if (t.dsv == nullptr || !t.width || !t.height)
      return false;

    for (uint32_t i = 0; i < 4u; i++) {
      if (t.rtv[i] == nullptr || t.rtv[i]->info().format != expected[i])
        return false;

      const auto& info = t.rtv[i]->image()->info();

      if (info.sampleCount != VK_SAMPLE_COUNT_1_BIT
       || (i != 1u && !(info.usage & VK_IMAGE_USAGE_SAMPLED_BIT)))
        return false;
    }

    const auto& depthInfo = t.dsv->image()->info();
    return depthInfo.sampleCount == VK_SAMPLE_COUNT_1_BIT
        && (depthInfo.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  }


  Rc<DxvkImageView> BlessedHalfRatePass::sampledView(const Rc<DxvkImageView>& rtv) const {
    DxvkImageViewKey key = rtv->info();
    key.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    key.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
    key.layout     = VK_IMAGE_LAYOUT_UNDEFINED;
    key.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    key.mipCount   = 1u;
    key.layerCount = 1u;
    return rtv->image()->createView(key);
  }


  void BlessedHalfRatePass::copyDepth(DxvkContext* ctx, const BlessedHalfRateTargets& t) {
    const DxvkImageViewKey& dsvInfo = t.dsv->info();

    VkImageSubresourceLayers dst = { VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 0u, 1u };
    VkImageSubresourceLayers src = { VK_IMAGE_ASPECT_DEPTH_BIT, dsvInfo.mipIndex, dsvInfo.layerIndex, 1u };

    ctx->copyImage(m_depthCopy, dst, VkOffset3D(),
      t.dsv->image(), src, VkOffset3D(), VkExtent3D { t.width, t.height, 1u });
  }


  void BlessedHalfRatePass::uploadParams(DxvkContext* ctx, const BlessedHalfRateParams& params) {
    Rc<DxvkResourceAllocation> slice = m_ubo->allocateStorage();
    std::memcpy(slice->mapPtr(), &params, sizeof(params));
    ctx->invalidateBuffer(m_ubo, std::move(slice));
  }


  void BlessedHalfRatePass::beginTimer(DxvkContext* ctx, uint32_t which) {
    uint32_t slot = ((m_timerFrame % TimerRing) * 2u + which) * 2u;

    for (uint32_t i = 0u; i < 2u; i++) {
      if (m_timers[slot + i] == nullptr)
        m_timers[slot + i] = m_device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0u, 0u);
    }

    ctx->writeTimestamp(m_timers[slot]);
  }


  void BlessedHalfRatePass::endTimer(DxvkContext* ctx, uint32_t which) {
    uint32_t slot = ((m_timerFrame % TimerRing) * 2u + which) * 2u;
    ctx->writeTimestamp(m_timers[slot + 1u]);
    m_timerFrame++;
  }


  void BlessedHalfRatePass::readTimers() {
    // the oldest ring entry of each kind, written TimerRing runs ago
    for (uint32_t which = 0u; which < 2u; which++) {
      for (uint32_t f = 0u; f < TimerRing; f++) {
        uint32_t slot = (f * 2u + which) * 2u;

        if (m_timers[slot] == nullptr || m_timers[slot + 1u] == nullptr)
          continue;

        DxvkQueryData begin = { };
        DxvkQueryData end   = { };

        if (m_timers[slot + 0u]->getData(begin) != DxvkGpuQueryStatus::Available
         || m_timers[slot + 1u]->getData(end)   != DxvkGpuQueryStatus::Available)
          continue;

        uint64_t ticks = end.timestamp.time - begin.timestamp.time;
        m_lastMs[which].store(float(double(ticks) * double(m_timestampPeriodNs) / 1.0e6), std::memory_order_relaxed);
        break;
      }
    }
  }


  void BlessedHalfRatePass::capture(DxvkContext* ctx, const BlessedHalfRateArgs& args) {
    m_layerValid = false;

    if (!usable() || !checkTargets(args.targets))
      return;

    const BlessedHalfRateTargets& t = args.targets;

    if (!ensureResources(t.dsv->image()->info().format, t.width, t.height))
      return;

    if (args.timing) {
      readTimers();
      beginTimer(ctx, 0u);
    }

    copyDepth(ctx, t);

    BlessedHalfRateParams params;
    std::memcpy(params.invViewProj, args.invViewProj, sizeof(params.invViewProj));
    params.captureDist = args.captureDist;
    params.clearDepth  = args.clearDepth;
    params.size[0] = int32_t(t.width);
    params.size[1] = int32_t(t.height);
    uploadParams(ctx, params);

    ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(Rc<DxvkShader>(m_captureCs));
    ctx->bindUniformBuffer(VK_SHADER_STAGE_COMPUTE_BIT, SlotParams, DxvkBufferSlice(m_ubo));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotDepth,  Rc<DxvkImageView>(m_depthCopyView));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSrcCol, sampledView(t.rtv[0]));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSrcRt2, sampledView(t.rtv[2]));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotSrcRt3, sampledView(t.rtv[3]));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotOutCol, Rc<DxvkImageView>(m_col.storage));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotOutRt2, Rc<DxvkImageView>(m_rt2.storage));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotOutRt3, Rc<DxvkImageView>(m_rt3.storage));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotOutDep, Rc<DxvkImageView>(m_depth.storage));
    ctx->dispatch(DivUp(t.width, 8u), DivUp(t.height, 8u), 1u);

    // leave nothing of ours bound; the d3d11 side restores its own state next
    for (uint32_t slot = SlotDepth; slot <= SlotOutDep; slot++)
      ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, slot, nullptr);
    ctx->bindUniformBuffer(VK_SHADER_STAGE_COMPUTE_BIT, SlotParams, DxvkBufferSlice());
    ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(nullptr);

    if (args.timing)
      endTimer(ctx, 0u);

    std::memcpy(m_layerInvViewProj, args.invViewProj, sizeof(m_layerInvViewProj));
    std::memcpy(m_layerCamPos, args.camPos, sizeof(m_layerCamPos));
    m_layerValid = true;
  }


  void BlessedHalfRatePass::prefill(DxvkContext* ctx, const BlessedHalfRateArgs& args) {
    // the d3d11 side only asks for a prefill within BLESSED_HALFRATE_PERIOD
    // frames of a capture it issued; this is the cs-thread half of that check
    if (!usable() || !m_layerValid || !checkTargets(args.targets))
      return;

    const BlessedHalfRateTargets& t = args.targets;

    if (t.width != m_width || t.height != m_height
     || t.dsv->image()->info().format != m_depthFormat)
      return;

    BlessedHalfRateParams params;
    std::memcpy(params.invViewProj, args.invViewProj, sizeof(params.invViewProj));
    std::memcpy(params.layerInvViewProj, m_layerInvViewProj, sizeof(params.layerInvViewProj));

    if (!InvertMatrix(args.invViewProj, params.viewProj)
     || !InvertMatrix(m_layerInvViewProj, params.layerViewProj))
      return;

    params.camDelta[0] = args.camPos[0] - m_layerCamPos[0];
    params.camDelta[1] = args.camPos[1] - m_layerCamPos[1];
    params.camDelta[2] = args.camPos[2] - m_layerCamPos[2];
    params.captureDist    = args.captureDist;
    params.sameSurfaceTol = args.sameSurfaceTol;
    params.clearDepth     = args.clearDepth;
    params.size[0] = int32_t(t.width);
    params.size[1] = int32_t(t.height);

    if (args.timing) {
      readTimers();
      beginTimer(ctx, 1u);
    }

    copyDepth(ctx, t);

    DxvkRenderTargets rt;
    for (uint32_t i = 0; i < 4u; i++)
      rt.color[i].view = t.rtv[i];
    rt.depth.view = t.dsv;
    ctx->bindRenderTargets(std::move(rt), 0u);

    VkViewport viewport = { 0.0f, 0.0f, float(t.width), float(t.height), 0.0f, 1.0f };
    VkRect2D scissor = { { 0, 0 }, { t.width, t.height } };
    DxvkViewport vp = { viewport, scissor };
    ctx->setViewports(1u, &vp);

    ctx->setInputLayout(0u, nullptr, 0u, nullptr);
    ctx->setInputAssemblyState(DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false));
    ctx->setRasterizerState(args.rsState);
    ctx->setMultisampleState(args.msState);
    ctx->setLogicOpState(args.loState);

    for (uint32_t i = 0; i < 4u; i++)
      ctx->setBlendMode(i, args.cbState);

    ctx->bindShader<VK_SHADER_STAGE_VERTEX_BIT>(Rc<DxvkShader>(m_fillVs));
    ctx->bindShader<VK_SHADER_STAGE_FRAGMENT_BIT>(Rc<DxvkShader>(m_fillFs));
    ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, SlotDepth,  Rc<DxvkImageView>(m_depthCopyView));
    ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, SlotSrcCol, Rc<DxvkImageView>(m_col.sampled));
    ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, SlotSrcRt2, Rc<DxvkImageView>(m_rt2.sampled));
    ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, SlotSrcRt3, Rc<DxvkImageView>(m_rt3.sampled));
    ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, SlotLayDep, Rc<DxvkImageView>(m_depth.sampled));

    VkDrawIndirectCommand draw = { };
    draw.vertexCount   = 3u;
    draw.instanceCount = 1u;

    // draw A: pixels the prepass covered; no depth test, no depth write
    DxvkDepthStencilState dsA;
    dsA.setDepthTest(false);
    dsA.setDepthWrite(false);
    dsA.setStencilTest(false);
    dsA.setDepthCompareOp(VK_COMPARE_OP_ALWAYS);
    ctx->setDepthStencilState(dsA);

    params.mode = 0.0f;
    uploadParams(ctx, params);
    ctx->bindUniformBuffer(VK_SHADER_STAGE_FRAGMENT_BIT, SlotParams, DxvkBufferSlice(m_ubo));
    ctx->draw(1u, &draw);

    // draw B: pixels the prepass left empty; writes the layer's depth
    // (Vulkan only writes depth with the test enabled, hence ALWAYS)
    DxvkDepthStencilState dsB = dsA;
    dsB.setDepthTest(true);
    dsB.setDepthWrite(true);
    ctx->setDepthStencilState(dsB);

    params.mode = 1.0f;
    uploadParams(ctx, params);
    ctx->draw(1u, &draw);

    // leave nothing of ours bound; the d3d11 side restores its own state next
    for (uint32_t slot = SlotDepth; slot <= SlotLayDep; slot++)
      ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, slot, nullptr);
    ctx->bindUniformBuffer(VK_SHADER_STAGE_FRAGMENT_BIT, SlotParams, DxvkBufferSlice());
    ctx->bindShader<VK_SHADER_STAGE_VERTEX_BIT>(nullptr);
    ctx->bindShader<VK_SHADER_STAGE_FRAGMENT_BIT>(nullptr);
    ctx->bindRenderTargets(DxvkRenderTargets(), 0u);

    if (args.timing)
      endTimer(ctx, 1u);

    // the layer stays valid for the next off frame too (period 3 and 4);
    // how old it may get is the d3d11 side's call
  }

}
