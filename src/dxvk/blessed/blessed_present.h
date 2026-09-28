// blessed: present-idle -- where and when dxvk calls vkQueuePresentKHR (BLESSED_PRESENT)
#pragma once

#include <atomic>
#include <cstdint>

#include "../../util/thread.h"

namespace dxvk {

  /**
   * \brief Present placement
   *
   * Under nvidia's layered (dxgi) present, the present operation stalls
   * the queue it runs on for ~0.36 ms after the swapchain blit, before
   * the next frame's first command buffer can start. These modes move it
   * out of the way:
   *
   * - Queue: vkQueuePresentKHR runs on a second queue of the graphics
   *   family that carries nothing else. The graphics queue goes straight
   *   from the blit to the next frame; only the present queue waits.
   * - Defer: the present stays on the graphics queue but is issued after
   *   the next command list (or after BLESSED_PRESENT_DEFER_US, default
   *   4000, if none comes), so the stall's wait overlaps real work.
   * - Bridge: no vulkan swap chain at all; a d3d12 flip-model swap chain
   *   on the window presents, see blessed_present_bridge.h.
   */
  enum class BlessedPresentMode : uint32_t {
    Off   = 0u,
    Queue = 1u,
    Defer = 2u,
    Bridge = 3u,
  };

  /**
   * \brief Present placement switches
   *
   * BLESSED_PRESENT=queue|defer|bridge, read once on first use from a
   * function-local static. Whether the device actually got a present
   * queue is DxvkDevice::blessedPresentQueue (a null handle if not).
   */
  class BlessedPresent {
  public:

    static BlessedPresentMode Mode();

    /// Defer mode: how long a present may wait for the next command list.
    static uint64_t DeferNs();

    /// BLESSED_PRESENT_ACQUIRE=lazy: no acquire right after a present; the
    /// next frame acquires when it needs the image. Independent of Mode().
    static bool LazyAcquire();

  };

  /**
   * rief Defer mode: marks a thread waiting on the submission queue
   *
   * A held present must not keep synchronize() or waitForIdle() waiting,
   * so the submission thread issues it at once while any waiter exists.
   * Costs nothing unless BLESSED_PRESENT=defer.
   */
  class BlessedDeferWaiter {
  public:

    BlessedDeferWaiter(std::atomic<uint32_t>& count, dxvk::condition_variable& cond)
    : m_count(count), m_on(BlessedPresent::Mode() == BlessedPresentMode::Defer) {
      if (m_on) {
        m_count += 1u;
        cond.notify_all();
      }
    }

    ~BlessedDeferWaiter() {
      if (m_on)
        m_count -= 1u;
    }

    BlessedDeferWaiter(const BlessedDeferWaiter&) = delete;
    BlessedDeferWaiter& operator = (const BlessedDeferWaiter&) = delete;

  private:

    std::atomic<uint32_t>& m_count;
    bool                   m_on;

  };

}
