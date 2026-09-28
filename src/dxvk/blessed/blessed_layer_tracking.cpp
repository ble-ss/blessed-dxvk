// blessed: perlayer -- the BLESSED_LAYER_TRACKING switch and the render target side of per-layer tracking
#include "blessed_layer_tracking.h"

#include "../dxvk_context.h"

#include "../../util/util_env.h"

namespace dxvk {

  const bool g_blessedLayerTracking = env::getEnvVar("BLESSED_LAYER_TRACKING") == "1";

  static_assert(MaxNumRenderTargets + 1u <= std::tuple_size_v<decltype(DxvkBlessedLayerSnapshot::layerMask)>);


  bool DxvkContext::blessedLayerTransitionOutOfOrder(
    const DxvkImageView&            view) {
    if (likely(!g_blessedLayerTracking))
      return false;

    // Only regularly synchronized passes: acquireRenderTargets flushes all
    // pending barriers before the pass, so no access to any other layer of
    // this image is left waiting on a barrier we have not recorded yet.
    if (m_flags.test(DxvkContextFlag::GpRenderPassUnsynchronized))
      return false;

    DxvkImage& image = *view.image();

    if (!image.blessedLayerTrackable())
      return false;

    VkImageSubresourceRange subresources = view.imageSubresources();
    uint64_t layers = blessedLayerRangeMask(subresources.baseArrayLayer, subresources.layerCount);

    if (layers == BlessedAllLayers)
      return false;

    // Every use of the image in this command list so far touched only
    // the layers in this mask. If the attachment's layers are not among
    // them, the command list has not used these subresources at all, which
    // is the same condition prepareOutOfOrderTransition checks per image.
    if (image.blessedTouchedLayers(m_trackingId) & layers)
      return false;

    m_blessedLayerStats.rescues += 1u;
    return true;
  }


  DxvkBlessedLayerSnapshot DxvkContext::blessedSnapshotAttachmentLayers(
    const DxvkFramebufferInfo&      framebufferInfo) {
    DxvkBlessedLayerSnapshot snapshot;

    for (uint32_t i = 0; i < framebufferInfo.numAttachments(); i++) {
      const auto& attachment = framebufferInfo.getAttachment(i);
      snapshot.layerMask[i] = attachment.view->image()->blessedTouchedLayers(m_trackingId);
    }

    return snapshot;
  }


  void DxvkContext::blessedRestoreAttachmentLayers(
    const DxvkFramebufferInfo&      framebufferInfo,
    const DxvkBlessedLayerSnapshot& snapshot) {
    // Between the snapshot and here, the only tracked uses of these images
    // were the attachments themselves. Reset every image to its earlier
    // record first, then add each attachment's layers, so that two
    // attachments on one image add up instead of overwriting each other.
    for (uint32_t i = 0; i < framebufferInfo.numAttachments(); i++) {
      const auto& attachment = framebufferInfo.getAttachment(i);
      attachment.view->image()->blessedSetTouchedLayers(m_trackingId, snapshot.layerMask[i]);
    }

    for (uint32_t i = 0; i < framebufferInfo.numAttachments(); i++) {
      const auto& attachment = framebufferInfo.getAttachment(i);
      DxvkImage& image = *attachment.view->image();

      VkImageSubresourceRange subresources = attachment.view->imageSubresources();
      uint64_t layers = blessedLayerRangeMask(subresources.baseArrayLayer, subresources.layerCount);

      image.blessedSetTouchedLayers(m_trackingId, image.blessedTouchedLayers(m_trackingId) | layers);
    }
  }


  void DxvkContext::blessedCountLayerPass(
          DxvkCmdBuffer             cmdBuffer) {
    auto& stats = m_blessedLayerStats;

    if (stats.rescues == stats.rescuesCounted)
      return;

    stats.rescuesCounted = stats.rescues;

    if (cmdBuffer != DxvkCmdBuffer::ExecBuffer)
      stats.passesOutOfOrder += 1u;
    else
      stats.passesInOrder += 1u;

    uint64_t passes = stats.passesOutOfOrder + stats.passesInOrder;

    if (!(passes & (passes - 1u))) {
      Logger::info(str::format("blessed: layer tracking: ", stats.passesOutOfOrder,
        " passes out of order by disjoint layers, ", stats.passesInOrder,
        " still in order (another attachment was in use), ", stats.rescues, " attachments"));
    }
  }

}
