// blessed: BLESSED_SHADOW_SOFT=1 -- soft, temporally-accumulated ray-traced
// sun shadows. Two compute dispatches (blessed_soft_shadow_trace.comp then
// blessed_soft_shadow_filter.comp) in place of blessed_sun_shadow.comp's
// single one. Owned and lazily built by BlessedShadowObjects
// (blessed_shadow.cpp) only once a dispatch with args.softEnabled is seen;
// the hard-shadow path never touches this class.
#pragma once

#include <array>
#include <cstdint>

#include "blessed_shadow.h"

#include "../dxvk_sampler.h"

namespace dxvk {

  class DxvkContext;
  class DxvkCommandList;
  class DxvkDevice;
  class DxvkBuffer;

  class BlessedSoftShadowState {
  public:

    explicit BlessedSoftShadowState(DxvkDevice* device);

    /**
     * \brief Runs the sun-disk trace + temporal accumulate pass, then the
     *        spatial filter pass, writing the final visibility into
     *        args.outputView's x channel.
     *
     * Assumes the caller (BlessedShadowObjects::dispatch) has already
     * transitioned args.depthView and args.outputView for compute access
     * and will transition them back, and that the render pass is already
     * ended. This call owns and tracks only its own images and buffers.
     *
     * \param [in] debugMode BLESSED_SHADOW_DEBUG value (0 off, 7 hist,
     *        8 reproj),
     *        read once by BlessedShadowObjects::dispatch and shared with
     *        the hard-shadow path
     * \param [in] debugAddress BLESSED_SHADOW_DUMP pixel buffer device
     *        address, or 0 -- forwarded to the filter pass verbatim
     */
    void dispatch(
            DxvkContext*                ctx,
      const Rc<DxvkCommandList>&        cmd,
      const BlessedShadowDispatchArgs&  args,
            uint32_t                    debugMode,
            uint64_t                    debugAddress);

  private:

    void ensureSized(const Rc<DxvkCommandList>& cmd, uint32_t width, uint32_t height);

    // blessed: BLESSED_SHADOW_DEBUG=reproj only -- logs, every 120th
    // dispatch, how far the game's CameraPreviousViewProjUnjittered and our
    // own previous matrix disagree, in pixels (see the .cpp)
    void logReprojCompare(
      const BlessedShadowDispatchArgs&  args,
      const float                       ownPrevViewProj[16],
      const float                       ownDelta[3]) const;

    DxvkDevice* m_device;

    // blessed: bilinear, clamp-to-edge -- used for the history reprojection
    // sample (real filtering) and the depth texelFetch (filtering ignored)
    Rc<DxvkSampler> m_linearSampler;

    const DxvkPipelineLayout* m_traceLayout    = nullptr;
    VkPipeline                m_tracePipeline  = VK_NULL_HANDLE;
    const DxvkPipelineLayout* m_filterLayout   = nullptr;
    VkPipeline                m_filterPipeline = VK_NULL_HANDLE;

    uint32_t m_width  = 0u;
    uint32_t m_height = 0u;

    // blessed: false right after (re)creation -- forces every pixel in the
    // trace pass to reject history for that one dispatch, rather than read
    // whatever garbage the freshly-allocated images happen to contain
    bool m_historyValid = false;

    // blessed: folded into the trace shader's per-pixel hash so a static
    // scene still gets a fresh dither pattern every dispatch
    uint32_t m_frameIndex = 0u;

    // blessed: BLESSED_SHADOW_REPROJ. "own" (default): reproject with the
    // matrix and camera of the dispatch that wrote the history, i.e. the
    // inverse of last dispatch's CameraViewProjInverse and last dispatch's
    // CameraPosAdjust. "game": the old path, the game's
    // CameraPreviousViewProjUnjittered plus the app-thread camera delta.
    // See fork-shadow-slide.md: the game path let history slide on turns.
    bool m_reprojGame = false;

    // blessed: what the dispatch that wrote the current history saw --
    // m_havePrev is false until one dispatch has run, and after a resize
    bool  m_havePrev = false;
    float m_prevInvViewProj[16] = { };
    float m_prevCamPos[3] = { 0.0f, 0.0f, 0.0f };

    // blessed: ping-pong pair, both usage SAMPLED | STORAGE, format
    // rgba16f (x = visibility, yzw = the surface's camera-relative position
    // when it was written; all-zero yzw = sky / no surface), kept resident
    // in VK_IMAGE_LAYOUT_GENERAL for their whole lifetime -- see
    // ensureSized. m_curr names the slot holding the most recently
    // written (i.e. current-frame) result; the trace pass reads the other
    // slot as history and writes m_curr's *new* target into 1 - m_curr,
    // then the indices swap.
    std::array<Rc<DxvkImage>, 2>     m_accumImage;
    std::array<Rc<DxvkImageView>, 2> m_accumSampledView;
    std::array<Rc<DxvkImageView>, 2> m_accumStorageView;
    uint32_t m_curr = 0u;
  };

}
