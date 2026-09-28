// blessed: vol-collapse, runs vanilla's ~90-dispatch volumetric z-integration chain as one dispatch
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;

  namespace blessed_vol_collapse_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief Collapses the vanilla volumetric integration chain
   *
   * After the froxel generate cs (cs.ab674eb1) the game integrates the
   * volume along z with one dispatch of cs.1c4ebb62 per slice (90 in the
   * whiterun dump), ping-ponging between two R16F 3d textures A and B.
   * Dispatch k writes dst[k-1] = src[k-1] and dst[k] = src[k-1] + src[k].
   * Each dispatch is tiny; the chain's cost is the 90 serialized
   * dispatches and the barriers between them.
   *
   * BLESSED_VOL_COLLAPSE=1 (off by default):
   *  - the first chain seen is only observed: its first k (cb0[1].x), its
   *    length, its two textures and its group counts are learned from a
   *    whole generate-to-generate cycle, and it runs as vanilla.
   *  - after that, at the first chain dispatch that follows a generate and
   *    matches what was learned, one dispatch (shaders/blessed_vol_collapse
   *    .hlsl) replays every store of the whole chain column by column on
   *    both textures, and every chain dispatch of that run is skipped
   *    (each one is checked against the learned k and texture order).
   *    Both textures end up holding exactly what the chain leaves in them,
   *    so every later reader (pass 138's draw, the lens-flare draw
   *    69c92902) sees vanilla's data.
   *  - a chain that deviates (another k, other textures, a shorter run)
   *    turns the collapse off for the process, with one warning.
   *
   * BLESSED_VOL_COLLAPSE_VERIFY=1: the first BLESSED_VOL_COLLAPSE_VERIFY_MAX
   * (default 8) chains run vanilla; the collapse runs on scratch copies of
   * A and B taken before the chain, and after the chain's last dispatch
   * both pairs are diffed with the shader verifier's kernel
   * (BlessedShaderVerify::CompareTextures, "<BLESSED_PROBE_DIR>/
   * shader-verify.jsonl" and the dxvk log). Later chains are collapsed.
   *
   * Only the chain itself changes: a mode that skips the generate and the
   * chain on some frames composes with this one (a frame without a
   * generate never collapses).
   *
   * Disabled: one cached-bool check per Dispatch.
   */
  class BlessedVolCollapse {

  public:

    static bool IsEnabled() {
      return blessed_vol_collapse_detail::g_enabled;
    }

    /**
     * \brief Pre-dispatch, immediate context only
     * \returns true if the dispatch must be skipped
     */
    static bool OnDispatch(
            D3D11ImmediateContext*  ctx,
      const D3D11ContextState&      state,
            UINT                    groupsX,
            UINT                    groupsY,
            UINT                    groupsZ);

    /**
     * \brief Post-dispatch, for dispatches that ran
     */
    static void OnDispatchDone(
            D3D11ImmediateContext*  ctx,
      const D3D11ContextState&      state);

  };

}
