// blessed: present-idle -- presents dxvk's frames through our own d3d12 flip-model swap chain on the game's window (BLESSED_PRESENT=bridge)
#pragma once

#include <memory>

#include "../dxvk_fence.h"
#include "../dxvk_image.h"
#include "../dxvk_presenter.h"

namespace dxvk {

  class DxvkDevice;

  /**
   * \brief DXGI present bridge
   *
   * nvidia's layered present makes a round trip per frame: the vulkan
   * queue hands the image to the driver's dxgi context and waits for it
   * to come back. The bridge makes that hop one-way. A d3d12 device on
   * the same adapter owns a flip-model swap chain on the window; the blit
   * renders into images shared with it and signals a shared d3d12 fence
   * (a vulkan timeline semaphore). The d3d12 queue waits on that fence,
   * then presents. The vulkan queue never waits on the d3d12 side; the
   * cpu waits before it reuses a buffer, which is already done unless
   * more than BLESSED_BRIDGE_BUFFERS frames are in flight.
   *
   * Zero-copy when the driver lets us share the swap chain's own buffers,
   * otherwise one d3d12 copy per frame from a shared image. Windowed
   * (borderless) only, sRGB colour space only (no HDR), tearing when the
   * sync interval is 0 and dxgi allows it.
   */
  class BlessedPresentBridge : public RcObject {

  public:

    BlessedPresentBridge(DxvkDevice* device, HWND window);

    ~BlessedPresentBridge();

    /// False if setup failed; the presenter then uses its own swap chain.
    bool valid() const;

    /// App thread, from Presenter::acquireNextImage. (Re)creates buffers
    /// when the extent or format changes.
    VkResult acquire(
            VkExtent2D          extent,
            VkSurfaceFormatKHR  format,
            PresenterSync&      sync,
            Rc<DxvkImage>&      image);

    /// Submission thread, from Presenter::presentImage, right after the
    /// blit's command list went to the graphics queue.
    VkResult present(uint32_t syncInterval);

    /// Waits for the d3d12 queue, before the presenter goes away.
    void waitIdle();

  private:

    struct Impl;
    std::unique_ptr<Impl> m;

  };

}
