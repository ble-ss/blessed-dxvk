// blessed: BLESSED_CASCADE_CACHE -- cached static sun shadow cascades: static casters drawn once, copied back each frame, moving casters redrawn on top
#pragma once

#include <cstdint>

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11DepthStencilView;
  class D3D11ImmediateContext;

  // blessed: backing flag for BlessedCascadeCache::IsEnabled(), set once at
  // dll load from BLESSED_CASCADE_CACHE (same split as
  // blessed_cascades_detail::g_enabled: env only, no Logger at static init)
  namespace blessed_cascade_cache_detail {
    extern const bool g_enabled;
  }

  /// Which draw entry point called \ref BlessedCascadeCache::OnDraw
  enum class BlessedCascadeDrawKind : uint32_t {
    Indexed,  ///< DrawIndexed: the only kind that can be cached
    Other,    ///< every other draw: always drawn
  };

  /**
   * \brief Cached static sun shadow cascades (BLESSED_CASCADE_CACHE=1)
   *
   * Skyrim redraws its two sun cascades (one 4096x4096 d16 array, one
   * layer per cascade) from scratch every frame, but the engine snaps the
   * cascade cameras to their texel grid and only moves the sun once a
   * second, so most frames draw the same static casters with the same
   * projection again. This keeps a copy of each layer's static depth:
   *
   * - learn: the sun cascade array is the d16 array a draw with a colour
   *   target samples at ps srv slot BLESSED_CASCADE_CACHE_SRV (default 4,
   *   the shadow-mask draw, whiterun-frame.md).
   * - per layer and frame, at the first draw after the engine's clear,
   *   pick a mode from the cascade's projection (vs b12 CameraViewProj +
   *   CameraPosAdjust) and the cache's health:
   *   - reuse: copy the cached static depth over the layer, then skip
   *     every draw whose key is in the cache and draw the rest on top.
   *   - build: the static draws go into the cache image, the moving ones
   *     into the layer, and a min-composite merges the cache into the
   *     layer at the end. The frame stays exact.
   *   - pass: plain vanilla (projection changed recently).
   * - a draw is static when it is a DrawIndexed with a PerGeometry cbuffer
   *   (vs b2), no skinning, only per-vertex slot 0 input, default/immutable
   *   vertex and index buffers, depth writes with LESS/LESS_EQUAL and a
   *   full viewport. Its key hashes the pipeline objects, the buffers,
   *   the draw range and the bytes of vs b0/b1/b2/b7 and ps b0/b1/b2/b11
   *   (so wind-animated trees, whose TreeParams change, never match).
   *
   * Depth testing is order-independent, so min(cached statics, this
   * frame's other draws) equals what vanilla draws, as long as every
   * cached key is drawn again this frame. A cached key that goes missing
   * (a static that moved or vanished) is seen only at the cascade's end;
   * that frame shows its old depth once, and the next frame rebuilds.
   *
   * Single-threaded by construction: every entry point runs on the
   * immediate context's thread under its device lock, like
   * BlessedCascadeSkip. Disabled cost: one predictable branch on a const
   * global per draw and clear.
   */
  class BlessedCascadeCache {

  public:

    static bool IsEnabled() {
      return blessed_cascade_cache_detail::g_enabled;
    }

    /**
     * \brief Pre-draw hook, immediate context only
     *
     * Learns the cascade array from the mask draw, ends an open cascade
     * when the target moves on, and inside a cascade redirects or skips.
     * \returns \c true: drop this draw, its depth is already in the layer
     */
    static bool OnDraw(
            D3D11ImmediateContext*  ctx,
      const D3D11ContextState&      state,
            BlessedCascadeDrawKind  kind,
            uint32_t                indexCount,
            uint32_t                startIndex,
            int32_t                 baseVertex);

    /**
     * \brief Before the context's own ClearDepthStencilView runs
     *
     * Ends an open cascade; a clear of a cascade layer opens the next one.
     * \returns \c true: drop this clear (the partial restore keeps the
     *    layer; the clear runs later if the cascade does not restore)
     */
    static bool OnClearDsv(
            D3D11ImmediateContext*  ctx,
            D3D11DepthStencilView*  dsv,
            uint32_t                clearFlags,
            float                   depth);

    /**
     * \brief Once per present, before the swap chain presents
     *
     * Ends an open cascade, reads back verify counters, writes
     * <BLESSED_PROBE_DIR>/cascade_cache.jsonl every 120 presents.
     */
    static void OnPresent(
            D3D11ImmediateContext*  ctx);

    /// A new swap chain size drops everything learned and cached
    static void NotifySwapchainExtent(
            uint32_t                width,
            uint32_t                height);

  };

}
