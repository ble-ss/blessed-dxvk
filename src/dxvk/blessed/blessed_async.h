// blessed: async compute -- env gates for running traced passes on a second queue (BLESSED_ASYNC)
#pragma once

#include <cstdint>

namespace dxvk {

  /**
   * \brief Which traced passes may move to the async queue
   */
  enum class BlessedAsyncPass : uint32_t {
    Gi  = 1u << 0,  // gi probe trace + ring copy, kicked at present
    Vol = 1u << 1,  // volumetrics trace half, kicked at the sao composite
  };

  /**
   * \brief Async compute switches
   *
   * What the environment asked for: BLESSED_ASYNC=1, BLESSED_ASYNC_QUEUE
   * (graphics: a second graphics-family queue instead of the dedicated
   * compute family, a/b only), BLESSED_ASYNC_PASSES (comma list of gi, vol
   * or none; default all). BLESSED_VOL_ASYNC=1 (vanilla-vol-async) also
   * requests the queue on its own, and always forces the graphics-family
   * mode: see src/d3d11/blessed_vol_async.h for why. Each is read once, on first use, from a
   * function-local static, so a static initializer in another file (the
   * d3d11 hooks) can call these safely. Whether the device actually got a
   * second queue is DxvkDevice::blessedHasAsyncQueue.
   */
  class BlessedAsync {
  public:

    static bool IsRequested();

    static bool WantsGraphicsFamily();

    static bool PassRequested(BlessedAsyncPass pass);

    // blessed: vanilla-vol-async -- BLESSED_VOL_ASYNC=1, its own switch,
    // independent of BLESSED_ASYNC_PASSES. See src/d3d11/blessed_vol_async.h.
    static bool VanillaVolRequested();

    // blessed: vol-async-3 -- BLESSED_VOL_ASYNC's value: 0 off, 1 the whole
    // window on a second graphics-family queue, 2 generate there and the
    // chain on a compute-family queue with queue family ownership
    // transfers on the two froxel volumes. See src/d3d11/blessed_vol_async.h.
    static uint32_t VanillaVolMode();

  };

}
