// blessed: cascade-cache -- gpu half of the cached static sun cascades (see blessed_cascade_cache.h)
#include "blessed_cascade_cache.h"

#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../dxvk_shader_spirv.h"

#include <array>

#include <blessed_cascade_min_vert.h>
#include <blessed_cascade_min_frag.h>
#include <blessed_cascade_verify.h>

namespace dxvk {

  namespace {

    // blessed: resource slots. The d3d11 side resets its command list
    // state before either pass and restores it after, so these slots hold
    // the game's bindings again right after we are done.
    constexpr uint32_t SlotCached = 0u;
    constexpr uint32_t SlotFresh  = 1u;

    // must match the push block in blessed_cascade_verify.comp (scalar)
    struct VerifyPush {
      uint64_t counters;
      uint32_t size[2];
    };

  }


  BlessedCascadeCachePass::BlessedCascadeCachePass(DxvkDevice* device)
  : m_device(device) {
    const std::array<DxvkBindingInfo, 1> fsBindings = {{
      { 0u, 0u, SlotCached, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
    }};

    const std::array<DxvkBindingInfo, 2> verifyBindings = {{
      { 0u, 0u, SlotCached, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
      { 0u, 1u, SlotFresh,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
    }};

    DxvkSpirvShaderCreateInfo vsInfo = { };
    vsInfo.debugName = "blessed cascade composite vs";
    m_vs = new DxvkSpirvShader(vsInfo, blessed_cascade_min_vert);

    DxvkSpirvShaderCreateInfo fsInfo = { };
    fsInfo.bindingCount = uint32_t(fsBindings.size());
    fsInfo.bindings     = fsBindings.data();
    fsInfo.debugName    = "blessed cascade composite fs";
    m_fs = new DxvkSpirvShader(fsInfo, blessed_cascade_min_frag);

    DxvkSpirvShaderCreateInfo verifyInfo = { };
    verifyInfo.bindingCount  = uint32_t(verifyBindings.size());
    verifyInfo.bindings      = verifyBindings.data();
    verifyInfo.localPushData = DxvkPushDataBlock(0u, sizeof(VerifyPush), sizeof(uint64_t), 0u);
    verifyInfo.debugName     = "blessed cascade verify";
    m_verify = new DxvkSpirvShader(verifyInfo, blessed_cascade_verify);
  }


  BlessedCascadeCachePass::~BlessedCascadeCachePass() {

  }


  void BlessedCascadeCachePass::composite(
          DxvkContext*          ctx,
    const Rc<DxvkImageView>&    dstDepth,
    const Rc<DxvkImageView>&    srcSampled,
          VkExtent2D            extent,
          VkCompareOp           compareOp) {
    DxvkRenderTargets rt;
    rt.depth.view = dstDepth;
    ctx->bindRenderTargets(std::move(rt), 0u);

    ctx->setInputAssemblyState(DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false));
    ctx->setInputLayout(0u, nullptr, 0u, nullptr);

    DxvkRasterizerState rs = { };
    rs.setPolygonMode(VK_POLYGON_MODE_FILL);
    rs.setCullMode(VK_CULL_MODE_NONE);
    rs.setFrontFace(VK_FRONT_FACE_CLOCKWISE);
    rs.setDepthClip(true);
    rs.setConservativeMode(VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    rs.setSampleCount(0);
    rs.setFlatShading(false);
    rs.setLineMode(VK_LINE_RASTERIZATION_MODE_DEFAULT_EXT);
    ctx->setRasterizerState(rs);
    ctx->setDepthBias(DxvkDepthBias());

    // blessed: the min. Depth testing is order-independent, so this gives
    // the same values as if the cached draws had run into this target.
    DxvkDepthStencilState ds = { };
    ds.setDepthTest(true);
    ds.setDepthWrite(true);
    ds.setDepthCompareOp(compareOp);
    ctx->setDepthStencilState(ds);

    DxvkMultisampleState ms = { };
    ms.setSampleMask(0xffffu);
    ctx->setMultisampleState(ms);

    DxvkLogicOpState lo = { };
    lo.setLogicOp(false, VK_LOGIC_OP_NO_OP);
    ctx->setLogicOpState(lo);

    // blessed: depth range 0..1 -- gl_FragDepth is clamped to it, never
    // remapped, so the d16 value round-trips exactly
    DxvkViewport vp = { };
    vp.viewport = { 0.0f, 0.0f, float(extent.width), float(extent.height), 0.0f, 1.0f };
    vp.scissor  = { { 0, 0 }, extent };
    ctx->setViewports(1u, &vp);

    ctx->bindShader<VK_SHADER_STAGE_VERTEX_BIT>(Rc<DxvkShader>(m_vs));
    ctx->bindShader<VK_SHADER_STAGE_FRAGMENT_BIT>(Rc<DxvkShader>(m_fs));
    ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, SlotCached, Rc<DxvkImageView>(srcSampled));

    VkDrawIndirectCommand draw = { };
    draw.vertexCount   = 3u;
    draw.instanceCount = 1u;
    ctx->draw(1u, &draw);

    ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, SlotCached, nullptr);
    ctx->bindShader<VK_SHADER_STAGE_VERTEX_BIT>(nullptr);
    ctx->bindShader<VK_SHADER_STAGE_FRAGMENT_BIT>(nullptr);
    ctx->bindRenderTargets(DxvkRenderTargets(), 0u);
  }


  void BlessedCascadeCachePass::verify(
          DxvkContext*          ctx,
    const Rc<DxvkImageView>&    cachedSampled,
    const Rc<DxvkImageView>&    freshSampled,
    const Rc<DxvkBuffer>&       counters,
          VkExtent2D            extent) {
    ctx->ensureBufferAddress(counters);

    VerifyPush push = { };
    push.counters = counters->getSliceInfo().gpuAddress;
    push.size[0]  = extent.width;
    push.size[1]  = extent.height;

    ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(Rc<DxvkShader>(m_verify));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotCached, Rc<DxvkImageView>(cachedSampled));
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotFresh,  Rc<DxvkImageView>(freshSampled));
    ctx->pushData(VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(push), &push);
    ctx->dispatch((extent.width + 7u) / 8u, (extent.height + 7u) / 8u, 1u);

    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotCached, nullptr);
    ctx->bindResourceImageView(VK_SHADER_STAGE_COMPUTE_BIT, SlotFresh,  nullptr);
    ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(nullptr);
  }

}
