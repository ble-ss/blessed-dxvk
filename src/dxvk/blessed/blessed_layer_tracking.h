// blessed: perlayer -- per-layer use tracking for layered render targets (BLESSED_LAYER_TRACKING=1)
#pragma once

#include <array>
#include <cstdint>

namespace dxvk {

  /**
   * \brief Global switch, read once from BLESSED_LAYER_TRACKING at load
   *
   * When set, every image records which array layers the current command
   * list has touched, and a render pass whose attachments only touch layers
   * this command list has not used yet may put its initial layout
   * transitions on the out-of-order init barrier buffer, the same way dxvk
   * already does for images the command list has not used at all.
   */
  extern const bool g_blessedLayerTracking;

  /**
   * \brief Layers of one image touched in one command list
   *
   * \c layerMask is only meaningful while \c trackingId equals the
   * current tracking ID. Bit i is array layer i, all mips, all aspects.
   * \c ~0 means "unknown, treat every layer as used".
   */
  struct DxvkBlessedLayerState {
    uint64_t trackingId = 0u;
    uint64_t layerMask  = 0u;
  };

  constexpr uint64_t BlessedAllLayers = ~uint64_t(0u);

  /**
   * \brief Layer records of a framebuffer's attachments before they get tracked
   *
   * Indexed like DxvkFramebufferInfo::getAttachment. Nine slots: eight
   * colour targets and one depth target.
   */
  struct DxvkBlessedLayerSnapshot {
    std::array<uint64_t, 9u> layerMask = { };
  };

  /**
   * \brief Counters, logged at powers of two
   */
  struct DxvkBlessedLayerStats {
    uint64_t rescues          = 0u; // attachments allowed out of order by layer disjointness
    uint64_t rescuesCounted   = 0u; // rescues already attributed to a pass
    uint64_t passesOutOfOrder = 0u; // passes whose transitions moved to the init barriers
    uint64_t passesInOrder    = 0u; // passes with a rescue that still went in order
  };

  /**
   * \brief Mask for a layer range
   *
   * Ranges that reach past layer 63 give the full mask.
   */
  inline uint64_t blessedLayerRangeMask(uint32_t baseLayer, uint32_t layerCount) {
    if (baseLayer >= 64u || layerCount >= 64u || baseLayer + layerCount > 64u || !layerCount)
      return BlessedAllLayers;

    return ((uint64_t(1u) << layerCount) - 1u) << baseLayer;
  }

}
