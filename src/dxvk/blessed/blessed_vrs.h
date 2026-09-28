// blessed: vrs -- injected variable rate shading on skyrim's lit pass (pass 115), BLESSED_VRS=adaptive|1x1|2x2
#pragma once

#include <cstdint>
#include <vector>

#include "../dxvk_include.h"

namespace dxvk {

  class DxvkDevice;
  class DxvkImage;
  class DxvkImageView;
  class DxvkSampler;
  class DxvkPipelineLayout;

  /**
   * \brief BLESSED_VRS modes
   *
   * Off: no extension, no pipeline flag, no per-draw work.
   * Const1x1: everything wired, the rate image holds 1x1 everywhere.
   *   Measures the cost of the plumbing alone.
   * Const2x2: the rate image holds 2x2 everywhere. The ceiling.
   * Adaptive: the rate image is rebuilt each frame from the lit pass's own
   *   colour and motion (fidelityfx variable shading, ported).
   */
  enum class BlessedVrsMode : uint32_t {
    Off       = 0,
    Const1x1  = 1,
    Const2x2  = 2,
    Adaptive  = 3,
  };


  /**
   * \brief Environment switches, each read once
   *
   * BLESSED_VRS=adaptive|1x1|2x2 (anything else: off)
   * BLESSED_VRS_DEBUG=rate    tints the lit pass's colour by the rate it used
   * BLESSED_VRS_ALLDRAWS=1    discard / depth-writing shaders go coarse too
   * BLESSED_VRS_CUTOFF=f      luminance delta below which a tile goes
   *                           coarse (fidelityfx VarianceCutoff), 0.015;
   *                           negative keeps all tiles 1x1 but still runs
   *                           the analysis (its cost, measured alone)
   * BLESSED_VRS_MOTION=f      fidelityfx MotionFactor, per pixel of
   *                           motion, 0.01
   * BLESSED_VRS_HYSTERESIS=0  a tile goes coarse on the first frame that
   *                           asks for it, not the second
   */
  class BlessedVrs {

  public:

    static BlessedVrsMode mode();

    static bool debugRate();

    static bool allDraws();

    static float cutoff();

    static float motionFactor();

    static bool hysteresis();

    static const char* modeName(BlessedVrsMode mode);

  };


  /**
   * \brief Per-context vrs state
   *
   * Created by DxvkContext only when the device enabled fragment shading
   * rate, which only happens when BLESSED_VRS names a mode. The logic
   * lives in DxvkContext::blessedVrs* (blessed_vrs.cpp) because it needs
   * the context's barrier tracking. This is plain data.
   */
  class BlessedVrsState {

  public:

    explicit BlessedVrsState(DxvkDevice* device);

    ~BlessedVrsState();

    DxvkDevice*             device;

    BlessedVrsMode          mode        = BlessedVrsMode::Off;
    bool                    debugRate   = false;
    bool                    allDraws    = false;
    bool                    hysteresis  = true;
    float                   cutoff      = 0.015f;
    float                   motionFactor = 0.01f;

    /// Rate attachment texel size, from the device limits
    VkExtent2D              texelSize   = { 16u, 16u };
    /// False when the texel size is not 16x16 (the analysis shader's tile)
    bool                    adaptiveOk  = true;
    /// False when r8_uint cannot be a storage rate attachment: no pass is
    /// ever matched, the combiner stays KEEP
    bool                    attachOk    = true;

    // --- per render pass ---

    /// The current render pass is the lit pass and carries the attachment
    bool                    passMatched = false;
    /// The lit pass ran and its colour/motion wait for the analysis
    bool                    pendingAnalyze = false;
    /// The lit pass's colour (slot 0) and motion (slot 1), held until the
    /// analysis runs at the next render pass begin
    Rc<DxvkImage>           litColor;
    Rc<DxvkImage>           litMotion;
    /// The render target view formats of the two
    VkFormat                litColorFormat  = VK_FORMAT_UNDEFINED;
    VkFormat                litMotionFormat = VK_FORMAT_UNDEFINED;

    // --- per draw ---

    /// The combiner must be set before the next draw
    bool                    drawDirty   = true;
    /// The combiner op last set, attachment slot
    VkFragmentShadingRateCombinerOpKHR drawOp = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;

    // --- lit pass size, see blessedVrsPrePass ---

    VkExtent2D              litSize     = { 0u, 0u };
    uint32_t                litMissCount = 0u;

    // --- resources, all ours, all resident in VK_IMAGE_LAYOUT_GENERAL ---

    /// Rate images: pass 115 reads [cur]; the analysis writes [1 - cur]
    Rc<DxvkImage>           rateImage[2];
    Rc<DxvkImageView>       rateStorageView[2];
    /// Raw views for the attachment: dxvk views do not carry the
    /// fragment shading rate usage
    VkImageView             rateAttachmentView[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t                rateCur     = 0u;
    /// Per tile, the rate the analysis asked for last frame (hysteresis)
    Rc<DxvkImage>           historyImage;
    Rc<DxvkImageView>       historyStorageView;
    /// False until one analysis wrote the history after a (re)allocation
    bool                    historyValid = false;
    /// Tile grid the images were made for
    VkExtent2D              tileExtent  = { 0u, 0u };
    /// Debug tint scratch, rgba16f, full size
    Rc<DxvkImage>           tintImage;
    Rc<DxvkImageView>       tintStorageView;

    /// Attachment info chained into the matched pass's VkRenderingInfo
    VkRenderingFragmentShadingRateAttachmentInfoKHR attachmentInfo = { VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR };

    /// Views and images replaced on a resize. Destroyed with this object:
    /// a resize is rare and a view may still be in flight.
    std::vector<VkImageView>   retiredViews;
    std::vector<Rc<DxvkImage>> retiredImages;

    // --- compute pipelines, built on first use ---

    Rc<DxvkSampler>           pointSampler;
    const DxvkPipelineLayout* rateLayout    = nullptr;
    VkPipeline                ratePipeline  = VK_NULL_HANDLE;
    const DxvkPipelineLayout* tintLayout    = nullptr;
    VkPipeline                tintPipeline  = VK_NULL_HANDLE;

    // --- counters for the log ---

    bool                    warnedAnalyze = false;
    bool                    warnedTint  = false;
    uint64_t                matchedPasses = 0u;
    uint64_t                analyses    = 0u;
    uint64_t                coarseDraws = 0u;
    uint64_t                fullDraws   = 0u;

  };

}
