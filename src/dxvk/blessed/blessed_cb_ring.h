// blessed: cb-ring -- one buffer rename into a constant-buffer ring block, as carried by the batched cs command
#pragma once

#include "../dxvk_buffer.h"

namespace dxvk {

  /**
   * \brief One pending rename of a dynamic constant buffer into a ring chunk
   *
   * Written by the d3d11 immediate context on Map(WRITE_DISCARD), stored
   * inline in the cs chunk as data of one batched rename command, and
   * applied on the cs thread by \c DxvkContext::blessedRenameBuffers in
   * the order the maps happened. The ring block itself is held once by
   * the command, not per entry.
   */
  struct DxvkBlessedCbRename {
    Rc<DxvkBuffer>  buffer;
    VkDeviceSize    offset = 0u;
  };

}
