// blessed: half-rate far field -- the dxvk-side far layer, its capture and the prefill draws
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "../dxvk_buffer.h"
#include "../dxvk_constant_state.h"
#include "../dxvk_framebuffer.h"
#include "../dxvk_gpu_query.h"
#include "../dxvk_image.h"
#include "../dxvk_shader.h"

namespace dxvk {

  class DxvkContext;
  class DxvkDevice;

  /**
   * \brief Uniform data of both half-rate shaders, std140
   *
   * Matrices are raw row-major cbuffer bytes, used as v * M in glsl
   * (the same convention as the shadow passes).
   */
  struct BlessedHalfRateParams {
    float   invViewProj[16]      = { };
    float   viewProj[16]         = { };
    float   layerViewProj[16]    = { };
    float   layerInvViewProj[16] = { };
    float   camDelta[4]          = { };
    float   captureDist          = 4000.0f;
    float   sameSurfaceTol       = 0.03f;
    float   clearDepth           = 1.0f;
    float   mode                 = 0.0f;
    int32_t size[4]              = { };
  };

  /**
   * \brief The main lit pass's four targets and its depth, as images
   *
   * Filled on the d3d11 app thread from the bound views; every field is
   * safe to copy into a cs lambda.
   */
  struct BlessedHalfRateTargets {
    std::array<Rc<DxvkImageView>, 4> rtv;   // colour, motion vectors, rt2, rt3
    Rc<DxvkImageView>                dsv;
    uint32_t                         width  = 0;
    uint32_t                         height = 0;
  };

  /**
   * \brief One capture (on frame) or prefill (off frame)
   */
  struct BlessedHalfRateArgs {
    BlessedHalfRateTargets targets;
    float    invViewProj[16] = { };
    float    camPos[3]       = { };
    float    captureDist     = 4000.0f;
    float    sameSurfaceTol  = 0.03f;
    float    clearDepth      = 1.0f;
    bool     timing          = false;

    // the prefill's render states, built on the d3d11 side from its own
    // defaults so the pass never guesses what "default" means there
    DxvkRasterizerState   rsState;
    DxvkMultisampleState  msState;
    DxvkLogicOpState      loState;
    DxvkBlendMode         cbState;
  };

  /**
   * \brief The far layer and the two things done with it
   *
   * Runs on the cs thread through DxvkContext's public api only, so dxvk
   * tracks every barrier and layout, the game's targets included. Both
   * entry points unbind what they bound; the caller must restore the
   * d3d11 context's own state afterwards (RestoreCommandListState).
   */
  class BlessedHalfRatePass : public RcObject {

  public:

    explicit BlessedHalfRatePass(DxvkDevice* device);
    ~BlessedHalfRatePass();

    /// On frame, main pass finished: copy the far pixels into the layer
    void capture(DxvkContext* ctx, const BlessedHalfRateArgs& args);

    /// Off frame, before the main pass's first draw: prefill its targets
    void prefill(DxvkContext* ctx, const BlessedHalfRateArgs& args);

    /// Latest gpu ms of a capture and of a prefill (BLESSED_HALFRATE_TIMING=1), or < 0
    float lastCaptureMs() const { return m_lastMs[0].load(std::memory_order_relaxed); }
    float lastPrefillMs() const { return m_lastMs[1].load(std::memory_order_relaxed); }

    /// False once resource creation has failed; the pass then stays off
    bool usable() const {
      return m_usable.load(std::memory_order_relaxed);
    }

  private:

    struct Layer {
      Rc<DxvkImage>     image;
      Rc<DxvkImageView> sampled;
      Rc<DxvkImageView> storage;
    };

    DxvkDevice*       m_device;

    Rc<DxvkShader>    m_captureCs;
    Rc<DxvkShader>    m_fillVs;
    Rc<DxvkShader>    m_fillFs;
    Rc<DxvkBuffer>    m_ubo;

    VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;
    uint32_t          m_width  = 0;
    uint32_t          m_height = 0;

    Rc<DxvkImage>     m_depthCopy;
    Rc<DxvkImageView> m_depthCopyView;
    Layer             m_col;
    Layer             m_rt2;
    Layer             m_rt3;
    Layer             m_depth;

    float             m_layerInvViewProj[16] = { };
    float             m_layerCamPos[3]       = { };
    bool              m_layerValid = false;

    static constexpr uint32_t TimerRing = 4u;
    std::array<Rc<DxvkQuery>, TimerRing * 4u> m_timers;
    uint32_t          m_timerFrame = 0;
    float             m_timestampPeriodNs = 1.0f;
    std::array<std::atomic<float>, 2> m_lastMs = { };

    std::atomic<bool> m_usable = { true };

    void createShaders();

    bool ensureResources(VkFormat depthFormat, uint32_t width, uint32_t height);

    Layer createLayer(VkFormat format, const char* name);

    bool checkTargets(const BlessedHalfRateTargets& t) const;

    Rc<DxvkImageView> sampledView(const Rc<DxvkImageView>& rtv) const;

    void copyDepth(DxvkContext* ctx, const BlessedHalfRateTargets& t);

    void uploadParams(DxvkContext* ctx, const BlessedHalfRateParams& params);

    void beginTimer(DxvkContext* ctx, uint32_t which);
    void endTimer(DxvkContext* ctx, uint32_t which);
    void readTimers();

  };

}
