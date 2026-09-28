// blessed: scene-capture app-thread config, draw selector, and BLESSED_SCENE_LOG
#pragma once

#include <cstdint>
#include <memory>

#include "d3d11_context_state.h"

namespace dxvk {

  class DxvkContext; // blessed: scene-cs
  class DxvkDevice;
  struct BlessedSceneSkinnedDraw; // blessed: hook-cpu-2, see blessed_scene.h
  struct BlessedSceneSkinBatch;

  // blessed: backing flag for BlessedSceneCapture::IsEnabled(), split out
  // so the hot-path gate (checked on every DrawIndexed/DrawIndexedInstanced)
  // is a trivial inline read -- same pattern as blessed_dump_detail::g_enabled
  // in blessed_dump.h. Set once at static-init time from BLESSED_SCENE_VS /
  // BLESSED_SCENE_RULE; never changes after.
  namespace blessed_scene_capture_detail {
    extern const bool g_enabled;
  }

  /// blessed: reasons a would-be-eligible draw (selector matched) got skipped anyway
  enum class BlessedSceneSkip : uint8_t {
    NoPosition = 0,
    BadPositionFormat,
    DynamicVertexBuffer,
    DynamicIndexBuffer,
    BadTopology,
    BadIndexFormat,
    NegativeBaseVertex,
    // blessed: rt-lifetime bounds audit -- see D3D11CommonContext::BlessedSceneCaptureDraw
    BaseVertexOutOfRange,
    MisalignedIndexOffset,
    Count
  };

  /// blessed: actor-skinning -- reasons a skinned draw (BLESSED_SCENE_SKINNED=1,
  /// input layout has BLENDINDICES0/BLENDWEIGHT0) got skipped. Logged separately
  /// from BlessedSceneSkip -- see "skinned_skipped" in scene.jsonl.
  enum class BlessedSkinSkip : uint8_t {
    NoSkinPosition = 0,
    BadSkinPositionFormat,
    DynamicSkinBuffer,
    BadTopology,
    BadIndexFormat,
    NegativeBaseVertex,
    BaseVertexOutOfRange,
    NoBonesBuffer,
    BadBonesSize,
    HasWorldMatrix,
    NoCamPos,
    Count
  };

  /// blessed: BLESSED_SCENE_XFORM / BLESSED_SCENE_CAMPOS, "[<vs|ps>:]<cb slot>:<byte offset>"
  struct BlessedSceneCbufferSlot {
    bool            valid  = false;
    D3D11ShaderType stage  = D3D11ShaderType::eVertex;  // blessed: no prefix = vs, for compatibility
    uint32_t        slot   = 0;
    uint32_t        offset = 0;
    uint32_t        size   = 0;  // blessed: optional "...:<cbuffer size>"; 0 = any
  };

  /**
   * \brief scene-cs: what the app thread hands the cs thread for one batch of static draws
   *
   * Filled by D3D11CommonContext::BlessedSceneCaptureDraw once a draw has
   * passed the checks that only need the input layout and D3D11 state it
   * already holds (steps 1-4 of the old path: position stream, format,
   * binding, bound vb). Checks 6, 7 and 9 (topology, index format, index
   * offset alignment) are also worked out there, but their verdict rides
   * along in \c deferredA / \c deferredB, so that \ref CaptureOnCs can
   * record them in the old order, between the dynamic-buffer checks it can
   * only make on the cs thread. Draws batch together only while the whole
   * record is equal (see BlessedBatchDrawIndexedScene).
   */
  struct BlessedSceneCsRecord {
    VkFormat posFormat  = VK_FORMAT_UNDEFINED;
    uint32_t posBinding = 0;
    uint32_t posOffset  = 0;
    // BlessedSceneSkip::Count = none; else BadTopology or BadIndexFormat
    // (recorded after the dynamic-vb check) ...
    BlessedSceneSkip deferredA = BlessedSceneSkip::Count;
    // ... and MisalignedIndexOffset (recorded after the dynamic-ib check)
    BlessedSceneSkip deferredB = BlessedSceneSkip::Count;
    // this frame's depth-only pass index for the depth_passes log, or ~0u
    uint32_t passIndex  = ~0u;

    bool operator == (const BlessedSceneCsRecord& o) const {
      return posFormat == o.posFormat && posBinding == o.posBinding && posOffset == o.posOffset
          && deferredA == o.deferredA && deferredB == o.deferredB && passIndex == o.passIndex;
    }
  };

  /**
   * \brief App-thread half of scene capture: config, the draw selector, and BLESSED_SCENE_LOG
   *
   * Everything here is cheap when disabled: \ref IsEnabled is a cached
   * bool, and every other entry point either short-circuits on it or is
   * only ever called after a caller has already checked it (matching
   * BlessedDump's convention in blessed_dump.h).
   */
  class BlessedSceneCapture {
  public:

    // True once a selector env var (BLESSED_SCENE_VS or
    // BLESSED_SCENE_RULE=depthonly) is set. Inline read of a plain global --
    // no out-of-line call.
    static bool IsEnabled() {
      return blessed_scene_capture_detail::g_enabled;
    }

    // blessed: actor-skinning -- true once BLESSED_SCENE_SKINNED=1. Checked
    // by D3D11CommonContext::BlessedSceneCaptureDraw before routing a draw
    // whose input layout has BLENDINDICES0/BLENDWEIGHT0 into the skinned
    // path instead of the (unchanged) static one. One cached bool.
    static bool SkinnedEnabled();

    /**
     * \brief Whether this draw's shader/pipeline state matches the configured selector
     *
     * Does not check per-draw eligibility (position stream, buffer usage,
     * topology, index format) -- that needs the D3D11InputLayout and
     * buffer descs the immediate context already has to hand, so it stays
     * in D3D11CommonContext::DrawIndexed[Instanced] itself.
     */
    static bool MatchesSelector(const D3D11ContextState& state);

    static BlessedSceneCbufferSlot XformConfig();
    static BlessedSceneCbufferSlot CamPosConfig();

    // blessed: skin-repair -- BLESSED_SCENE_SKIN_PIVOT, default vs:12:640:
    // what the skinned vertex shader subtracts from the blended bones.
    static BlessedSceneCbufferSlot SkinPivotConfig();

    // blessed: skin-repair -- BLESSED_SCENE_SKIN_TRACE=1
    static bool SkinTraceEnabled();

    // blessed: the depthonly rule needs to know the swapchain's own size;
    // D3D11SwapChain::Present calls the setter once per present, before
    // any draw of the next frame can run.
    static void NotifySwapchainExtent(uint32_t width, uint32_t height);
    static bool IsSwapchainExtent(uint32_t width, uint32_t height);

    // blessed: skin-v2 -- depth-only pass tracking. A pass is a run of
    // selector-matching draws on one dsv, broken by any indexed draw that
    // does not match, a dsv change, the hooked (mask) draw, or a present.
    // Ids start at 1 and never repeat. Only moves under the depthonly rule.
    static uint32_t CurrentPass();

    // blessed: skin-v2 -- called by BlessedHook on a matching (mask) draw:
    // remembers the last depth-only pass before it (first match per frame).
    static void NoteMaskDraw();

    // blessed: skin-v2 -- once per present, before endFrame is queued: the
    // pass NoteMaskDraw saw this frame (0 = none), and resets the frame.
    static uint32_t TakeMaskPass();

    // blessed: hook-cpu-2 -- true when a skinned draw in the current pass
    // can never feed the tlas: the mask draw was seen after a depth-only
    // pass, this pass came later, and BLESSED_SCENE_SKIN_PASS is not "all".
    // BlessedScene::selectSkinnedDraws would drop it at endFrame anyway.
    static bool SkinnedDrawIsLate();

    // blessed: hook-cpu-2 -- counts a late skinned draw (see above) without
    // staging it; reported as skinned_other_pass, same as before.
    static void CountLateSkinnedDraw();

    // blessed: hook-cpu-2 -- appends one skinned draw to this frame's
    // batch. The bones are not copied: \p draw holds its b10 allocation.
    static void StageSkinnedDraw(BlessedSceneSkinnedDraw& draw);

    // blessed: hook-cpu-2 -- true for a buffer the skinned path could take
    // as b10 (dynamic cbuffer, 3,840 bytes) while BLESSED_SCENE_SKINNED=1;
    // D3D11Buffer then remembers its mapped allocation. Read at creation.
    static bool IsBonesBufferDesc(const D3D11_BUFFER_DESC& desc);

    // blessed: hook-cpu-2 -- once per present, next to TakeMaskPass: this
    // frame's batch (null if nothing was staged); the next draw starts a new one.
    static std::unique_ptr<BlessedSceneSkinBatch> TakeSkinBatch();

    /**
     * \brief scene-cs: the static path's cs-thread half, one batch of draws
     *
     * Runs inside the batched draw command, right before the draws are
     * recorded, so \p ctx's bound vertex/index buffers and vs/ps uniform
     * buffers are exactly the draws' own. Makes the remaining checks in the
     * old order (dynamic vb, \c deferredA, dynamic ib, \c deferredB,
     * negative base vertex, transform, lod scale, base vertex range),
     * counts every draw, and calls blessedSceneAddDraw for each one kept.
     */
    static void CaptureOnCs(DxvkContext* ctx, const BlessedSceneCsRecord& record,
      const VkDrawIndexedIndirectCommand* draws, size_t count);

    // blessed: scene-cs -- the cs thread's frame boundary for the static
    // per-pass counts (depth_passes): called from the present's ordered
    // chunk, just before blessedSceneEndFrame.
    static void EndFrameOnCs();

    // blessed: scene-cs -- this frame's depth-only pass index (for
    // BlessedSceneCsRecord::passIndex), ~0u if it is not logged.
    static uint32_t FramePassIndex();

    // BLESSED_SCENE_LOG=1: counters for the current 120-present window.
    // blessed: scene-cs -- static captures are counted by CaptureOnCs.
    static void RecordSkipped(BlessedSceneSkip reason);

    // blessed: actor-skinning -- same idea, separate counters (see
    // BlessedSkinSkip and "skinned_captured"/"skinned_skipped" in scene.jsonl).
    static void RecordSkinnedCaptured();
    static void RecordSkinnedSkipped(BlessedSkinSkip reason);

    // Call once per D3D11SwapChain::Present, alongside BlessedDump::OnPresent.
    // Writes <BLESSED_PROBE_DIR>/scene.jsonl every 120 presents when
    // BLESSED_SCENE_LOG=1; a no-op otherwise. \p device supplies the
    // cs-side counters (blas count/bytes/builds/instances) from its
    // BlessedScene, if it has one.
    static void OnPresent(DxvkDevice* device);

  };

}
