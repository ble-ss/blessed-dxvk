// blessed: vrs -- d3d11 side of BLESSED_VRS: marks pixel shaders that must shade at 1x1 (see src/dxvk/blessed/blessed_vrs.h)
#pragma once

#include <cstddef>

namespace dxvk {

  class DxvkShader;

  /**
   * \brief Scans a DXBC pixel shader and tags its DxvkShader
   *
   * A coarse fragment's discard kills the whole 2x2 block (blocky leaf
   * edges); coarse depth, coverage or stencil exports are wrong in the
   * same way. Such shaders keep 1x1 through the per-draw combiner.
   * No-op unless BLESSED_VRS names a mode.
   */
  void BlessedVrsTagPixelShader(
          DxvkShader*             pShader,
    const void*                   pBytecode,
          size_t                  BytecodeLength);

}
