// blessed: BLESSED_LOOK -- the bless colour chain as a dxvk-side post pass (images, shaders, dispatches)
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "../dxvk_image.h"
#include "../dxvk_gpu_query.h"
#include "../dxvk_shader.h"

namespace dxvk {

  class DxvkContext;
  class DxvkDevice;

  /**
   * \brief Push data of blessed_look_prep.comp, std430 layout
   */
  struct BlessedLookPrepPush {
    int32_t srcSize[2]  = { };
    int32_t dstSize[2]  = { };
    float   offsetUv[2] = { };
    int32_t mode        = 0;
    float   threshold   = 0.0f;
    float   knee        = 0.0f;
    float   surface     = 0.0f;
    float   haze        = 0.0f;
    float   pad         = 0.0f;
  };

  /**
   * \brief Push data of blessed_look_composite.comp, std430 layout
   *
   * The d3d11 side fills everything but the three sizes, which the
   * pass sets once it knows the target's extent.
   */
  struct BlessedLookCompositePush {
    int32_t size[2]        = { };
    int32_t bloomSize[2]   = { };
    int32_t spillSize[2]   = { };
    float   bloomStrength  = 0.32f;
    float   spillStrength  = 0.35f;
    float   exposure       = 1.0f;
    float   grainStrength  = 0.03f;
    float   grainScale     = 0.60f;
    float   grainTime      = 0.0f;
    float   shoulderKnee   = 0.78f;
    float   shoulderBend   = 0.80f;
    float   warmth         = 0.35f;
    float   pastel         = 0.22f;
    float   splitTone      = 0.60f;
    float   saturation     = 1.06f;
    float   contrast       = 0.92f;
    float   lift           = 0.035f;
    float   gradeStrength  = 1.0f;
    float   vignette       = 0.0f;
    int32_t debugMode      = 0;
    int32_t splitX         = 0;
    int32_t swapRB         = 0;
  };

  /**
   * \brief One run of the look pass
   *
   * Filled on the d3d11 app thread (see src/d3d11/blessed_look.cpp) and
   * captured by value into the cs lambda; every field is safe to copy.
   */
  struct BlessedLookArgs {
    Rc<DxvkImage>             target;          // rtv 0's image, read and written
    VkImageSubresourceLayers  targetLayers = { };
    uint32_t                  width  = 0;
    uint32_t                  height = 0;

    float bloomThreshold = 0.80f;
    float bloomKnee      = 0.35f;
    float bloomSurface   = 0.10f;
    float bloomHaze      = 0.10f;
    float bloomRadius    = 3.5f;   // full-res px per tap
    float spillThreshold = 0.55f;
    float spillKnee      = 0.25f;
    float spillRadius    = 6.0f;   // quarter-res px per tap

    BlessedLookCompositePush composite;

    bool timing = false;           // BLESSED_LOOK_TIMING=1
  };

  /**
   * \brief The look pass: owns its shaders, scratch images and timers
   *
   * Runs on the cs thread through DxvkContext's public api only
   * (copyImage, bindShader, bindResourceImageView, pushData, dispatch),
   * so dxvk tracks every barrier and layout, the game's rtv 0 included.
   * The caller must restore the d3d11 context's own bindings after the
   * lambda that calls \ref run (RestoreCommandListState); \ref run
   * unbinds everything it bound.
   *
   * Chain, bless's bloom_grain.json order: rtv 0 (read in place, or a
   * copy when it isn't sampleable as unorm) -> bloom extract
   * + h blur (half res) -> bloom v blur -> spill extract (quarter res)
   * -> spill blur h -> spill blur v -> composite -> copy back to rtv 0.
   */
  class BlessedLookPass : public RcObject {

  public:

    explicit BlessedLookPass(DxvkDevice* device);
    ~BlessedLookPass();

    void run(DxvkContext* ctx, const BlessedLookArgs& args);

    /// Latest measured gpu time of the whole pass in ms, or a
    /// negative value when none is available (BLESSED_LOOK_TIMING=1)
    float lastGpuMs() const {
      return m_lastGpuMs.load(std::memory_order_relaxed);
    }

    /// False once resource creation has failed; the pass then stays off
    bool usable() const {
      return m_usable;
    }

  private:

    struct Target {
      Rc<DxvkImage>     image;
      Rc<DxvkImageView> sampled;
      Rc<DxvkImageView> storage;
    };

    DxvkDevice*       m_device;

    Rc<DxvkShader>    m_prep;
    Rc<DxvkShader>    m_composite;

    VkFormat          m_format = VK_FORMAT_UNDEFINED;
    uint32_t          m_width  = 0;
    uint32_t          m_height = 0;

    Target            m_src;
    Target            m_out;
    Target            m_bloomA;
    Target            m_bloomB;
    Target            m_spillA;
    Target            m_spillB;

    static constexpr uint32_t TimerRing = 4u;
    std::array<Rc<DxvkQuery>, TimerRing * 2u> m_timers;
    uint32_t          m_timerFrame = 0;
    float             m_timestampPeriodNs = 1.0f;
    std::atomic<float> m_lastGpuMs = { -1.0f }; // written on the cs thread, read on the app thread

    bool              m_usable = true;

    void createShaders();

    bool ensureTargets(VkFormat format, uint32_t width, uint32_t height);

    Target createTarget(VkFormat format, uint32_t width, uint32_t height, const char* name, bool storage);

    void prep(
            DxvkContext*          ctx,
      const Rc<DxvkImageView>&    src,
      const Target&               dst,
      const BlessedLookPrepPush&  push);

    void readTimers();

  };

}
