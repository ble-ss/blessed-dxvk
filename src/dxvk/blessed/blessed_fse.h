// blessed: present-fse-appcontrolled -- BLESSED_FSE mode switch
#pragma once

#include <cstdint>

namespace dxvk {

  /**
   * \brief Application-controlled exclusive fullscreen mode
   *
   * BLESSED_FSE=app requests VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT
   * instead of the driver-managed ALLOWED/DISALLOWED modes dxvk.allowFse picks
   * between. Off by default; costs one cached bool read when disabled.
   */
  enum class BlessedFseMode : uint32_t {
    Off = 0u,
    App = 1u,
  };

  class BlessedFse {
  public:

    static BlessedFseMode Mode();

  };

}
