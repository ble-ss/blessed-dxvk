// blessed: app-thread half of ray-traced gi -- patches vanilla's per-draw ambient cbuffer term
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;
  class DxvkContext;
  class DxvkDevice;

  // blessed: backing flag for BlessedGi::IsEnabled(), split out so the
  // hot-path gate (checked on every Draw/DrawIndexed/DrawInstanced/
  // DrawIndexedInstanced) is a trivial inline read -- same pattern as
  // blessed_dump_detail::g_enabled in blessed_dump.h. Set once at
  // static-init time from BLESSED_GI; never changes after.
  namespace blessed_gi_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief Gi v1: the index-buffer draw parameters OnDraw needs to key a mesh
   *
   * Only DrawIndexed/DrawIndexedInstanced (the same draws BlessedSceneCapture
   * keys off of) have these; \c valid is false from Draw/DrawInstanced call
   * sites, and OnDraw's mesh/albedo association is skipped entirely then --
   * a non-indexed draw was never going to match a scene-capture CacheKey
   * anyway.
   */
  struct BlessedGiDrawIndices {
    bool  valid       = false;
    UINT  indexCount  = 0;
    UINT  startIndex  = 0;
    INT   baseVertex  = 0;
  };

  /**
   * \brief gi-bounds: where one claimed draw's position stream lives
   *
   * Filled by \ref BlessedGi::OnDraw under BLESSED_GI_SAMPLE=bounds for an
   * indexed, single-instance, static (non-skinned) triangle-list draw; the
   * cs side (\ref BlessedGi::PatchOnCsBounds) reads the bound vb at
   * \c posBinding plus \c posOffset to key the mesh. \c valid stays false
   * in every other case (and always in origin mode), and the draw keeps
   * the plain gi batch and the origin sample. Draws batch together only
   * while the whole record is equal.
   */
  struct BlessedGiCsRecord {
    bool     valid      = false;
    VkFormat posFormat  = VK_FORMAT_UNDEFINED;
    uint32_t posBinding = 0;
    uint32_t posOffset  = 0;

    bool operator == (const BlessedGiCsRecord& o) const {
      return valid == o.valid && posFormat == o.posFormat
          && posBinding == o.posBinding && posOffset == o.posOffset;
    }
  };

  /**
   * \brief Replaces vanilla's DirectionalAmbient with traced (or constant) irradiance
   *
   * Split in two since gi-cs. \ref OnDraw is the app-thread half, called
   * from D3D11CommonContext<>::Draw* before the draw is batched: it only
   * checks the pass (cached per render-target generation) and sends the
   * per-mesh albedo note, and returns true when the draw should be patched.
   * The call site then batches the draw as BlessedBatchDraw*Gi instead of
   * BatchDraw*, whose cs command runs \ref PatchOnCs right before it
   * records the draw(s). PatchOnCs reads the draw's vs b2 World and ps b12
   * CameraPosAdjust from the *bound* uniform buffer slices' host mappings
   * (DxvkContext::blessedUniformBuffer), samples the probe grid, and writes
   * the row into the bound ps b2 slice's host mapping. On the cs thread the
   * bound slice is exactly this draw's discard allocation (invalidateBuffer
   * arrives in order, before the draw), and the gpu reads it only after the
   * command list is submitted, so the write lands in time.
   *
   * Stage 0 (BLESSED_GI=const): every draw whose ps b2 cbuffer is >= 224
   * bytes, while the main lit pass is bound (4 render targets, rtv0
   * R16G16B16A16_FLOAT, a dsv -- see docs/research/gi-route.md), has bytes
   * [176,224) of its bound ps b2 overwritten with a constant ambient
   * (BLESSED_GI_CONST, default solid red) in the DirectionalAmbient 3x4
   * form: constant column = the colour, direction columns zero.
   *
   * Stage 1 (BLESSED_GI=probes) additionally samples the traced probe grid
   * (BlessedGiState, dxvk-side) at the draw's world position and converts
   * its stored linear-in-normal fit to the same 3x4 form.
   *
   * All entry points are one cached-bool check unless BLESSED_GI is set.
   */
  class BlessedGi {
  public:

    // Cheap, cached: true once BLESSED_GI is "const" or "probes". Inline
    // read of a plain global -- no out-of-line call.
    static bool IsEnabled() {
      return blessed_gi_detail::g_enabled;
    }

    // Called from the immediate context's Draw*, before BatchDraw*. Takes
    // the owning DxvkDevice explicitly (m_device on the D3D11CommonContext
    // call site) rather than adding a public accessor to
    // D3D11ImmediateContext just for this. blessed: gi-cs -- returns true
    // when the caller must batch this draw with BlessedBatchDraw*Gi.
    // blessed: gi-bounds -- \p pRecord (indexed call sites only) is filled
    // under BLESSED_GI_SAMPLE=bounds, see BlessedGiCsRecord; untouched otherwise.
    static bool OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device,
      const BlessedGiDrawIndices& indices = BlessedGiDrawIndices(),
      BlessedGiCsRecord* pRecord = nullptr);

    // blessed: gi-cs -- cs thread, right before a claimed draw batch is
    // recorded: reads the bound cbuffers, samples, writes the ps b2 row.
    // \p drawCount is the batch size (for gi.jsonl's patched_per_frame).
    static void PatchOnCs(DxvkContext* ctx, size_t drawCount);

    // blessed: gi-bounds -- PatchOnCs for a batch with a valid record:
    // samples over the union of the batch's mesh bounds (see
    // BlessedGiState::sampleAmbientBounds), or at the origin while those
    // are not known yet.
    static void PatchOnCsBounds(DxvkContext* ctx, const BlessedGiCsRecord& record,
      const VkDrawIndexedIndirectCommand* draws, size_t drawCount);

    // Frame boundary: rolls the per-120-present gi.jsonl line and resets
    // the once-per-frame lighting-capture flag. \p device may be null.
    static void OnPresent(DxvkDevice* device);

    // blessed: mirrors BlessedSceneCapture::NotifySwapchainExtent -- called
    // from D3D11SwapChain::Present, own copy so this stays self-contained.
    static void NotifySwapchainExtent(uint32_t width, uint32_t height);

  private:

    /**
     * \brief Gi v1: resolves this draw's mesh identity + bound diffuse texture,
     *        and hands them to the cs thread if that mesh's texture changed
     *        since the last time this function saw it.
     *
     * A method (not a free function) so it inherits BlessedGi's own friend
     * access to D3D11CommonContext::EmitCs (see d3d11_context.h's
     * `friend class BlessedGi;`) -- \p ctx's EmitCs is protected.
     */
    static void ResolveMeshAlbedo(D3D11ImmediateContext* ctx, const D3D11ContextState& state,
      const BlessedGiDrawIndices& indices);

  };

}
