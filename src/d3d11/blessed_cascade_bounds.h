// blessed: cascade-cache -- where a moving caster lands in a cascade layer: per-mesh bounds and a draw's texel rectangle (the partial restore)
#pragma once

#include <cstdint>

#include "d3d11_context_state.h"

namespace dxvk {

  class DxvkDevice;
  struct BlessedCascadeBoundsArgs;

  /**
   * \brief Texel rectangles of cascade draws (BLESSED_CASCADE_CACHE_RESTORE=partial)
   *
   * A draw's footprint in a cascade layer is bounded from its mesh's
   * object-space aabb, measured once on the gpu from the game's own
   * index and vertex buffers (blessed_cascade_bounds.comp), and the
   * draw's transform:
   *
   * - skinned (the vs reads bones, b10): the aabb under every bone the
   *   mesh weights, unioned. A skinned vertex is a convex blend of its
   *   bones' transforms (weights sum to one), so it lies inside that union.
   * - b2 PerGeometry: the aabb under World, grown first by the most a
   *   wind-animated vs moves a vertex (1.1 * |TreeParams.z| * sqrt(3),
   *   Utility.hlsl) when the vs declares TreeParams.
   * - a dynamic position stream (actor heads): the aabb is read on the cpu
   *   from the mapped vertices, over the index range the gpu measured.
   *
   * Everything else, and every mesh not measured yet, is unknown: the
   * caller falls back to the full restore. App thread only.
   */
  namespace BlessedCascadeBounds {

    struct Rect {
      int32_t x0, y0, x1, y1;   ///< texels, half-open
    };

    enum class Result : uint32_t {
      Rect,       ///< \c out holds the rectangle
      Outside,    ///< the draw cannot touch the layer
      Unknown,    ///< no bound (see \c reason)
    };

    enum Reason : uint32_t {
      ReasonNone,
      ReasonNoPosition,   ///< no POSITION0 stream, or an unreadable format
      ReasonNoBuffers,    ///< missing or dynamic index buffer, dynamic skin streams
      ReasonPending,      ///< measured on the gpu, not read back yet
      ReasonFailed,       ///< measured, but nothing usable (or out of slots)
      ReasonNoTransform,  ///< neither bones nor PerGeometry World
      ReasonScan,         ///< dynamic positions out of range or too many
      ReasonCount,
    };

    const char* ReasonName(uint32_t r);

    /**
     * \brief The rectangle one DrawIndexed covers in the layer
     *
     * \param [in] vp The cascade's view-proj, row-major, world-relative
     * \param [in] posAdjust The cascade's CameraPosAdjust
     * \param [in] frame Present counter (per-frame caches, readback age)
     */
    Result DrawRect(
      const D3D11ContextState&  state,
            uint32_t            indexCount,
            uint32_t            startIndex,
            int32_t             baseVertex,
            uint64_t            frame,
      const double*             vp,
      const double*             posAdjust,
            uint32_t            width,
            uint32_t            height,
            Rect*               out,
            uint32_t*           reason);

    /**
     * \brief Hands the meshes queued this frame to one gpu dispatch
     *
     * \returns false when nothing is queued
     */
    bool TakeRequests(
            DxvkDevice*               device,
            BlessedCascadeBoundsArgs* args,
            uint64_t                  frame);

    /// Reads back meshes measured at least 8 presents ago
    void ReadBack(uint64_t frame);

    /// Meshes known, measured, pending
    void Counts(uint32_t* known, uint32_t* ready, uint32_t* pending);

  }

}
