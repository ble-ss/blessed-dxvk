// blessed: scene-capture app-thread config, selector, and logging -- see blessed_scene_capture.h
#include "blessed_scene_capture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "d3d11_shader.h"
#include "d3d11_texture.h"

#include "../dxvk/dxvk_context.h" // blessed: scene-cs
#include "../dxvk/dxvk_device.h"
#include "../dxvk/dxvk_format.h"  // blessed: scene-cs
#include "../dxvk/blessed/blessed_scene.h"

#include "../util/util_blessed_probe.h" // blessed: hook-cpu-2
#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    enum class SelectorKind { None, VsHash, DepthOnly };

    std::vector<std::string> SplitCsv(const std::string& s) {
      std::vector<std::string> out;
      std::string cur;

      for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == ',') {
          if (!cur.empty())
            out.push_back(cur);
          cur.clear();
        } else {
          cur += s[i];
        }
      }

      return out;
    }

    // blessed: "vs.<hex>" -> "<hex>", also tolerates a bare hex token
    std::string HexPart(const std::string& s) {
      size_t dot = s.find('.');
      return dot != std::string::npos ? s.substr(dot + 1) : s;
    }

    // blessed: "[<vs|ps>:]<slot>:<offset>[:<size>]" -- no stage prefix = vs,
    // for compatibility with BLESSED_SCENE_XFORM/_CAMPOS values set before
    // the prefix existed.
    bool ParseSlotOffset(const std::string& s, BlessedSceneCbufferSlot& out) {
      std::string rest = s;
      out.stage = D3D11ShaderType::eVertex;

      if (rest.rfind("vs:", 0) == 0) {
        rest = rest.substr(3);
      } else if (rest.rfind("ps:", 0) == 0) {
        out.stage = D3D11ShaderType::ePixel;
        rest = rest.substr(3);
      }

      size_t colon = rest.find(':');
      if (colon == std::string::npos)
        return false;

      out.slot   = std::strtoul(rest.substr(0, colon).c_str(), nullptr, 10);
      out.offset = std::strtoul(rest.substr(colon + 1).c_str(), nullptr, 10);
      // blessed: a third field pins the cbuffer's byte size, so shaders with
      // a different layout in the same slot are skipped (whiterun: vs
      // 1f18fcd5's 208-byte b2 put garbage transforms in the tlas)
      size_t colon2 = rest.find(':', colon + 1);
      out.size   = colon2 == std::string::npos ? 0u
                 : uint32_t(std::strtoul(rest.substr(colon2 + 1).c_str(), nullptr, 10));
      out.valid  = true;
      return true;
    }

    struct SceneConfig {
      SelectorKind              kind = SelectorKind::None;
      std::vector<std::string>  vsHashes; // hex part only, lowercase as given
      BlessedSceneCbufferSlot   xform;
      BlessedSceneCbufferSlot   camPos;
      BlessedSceneCbufferSlot   skinPivot; // blessed: skin-repair
    };

    // blessed: pure function of BLESSED_SCENE_VS/BLESSED_SCENE_RULE alone --
    // no other static touched -- safe to run at static-init time (dll load)
    // for blessed_scene_capture_detail::g_enabled below. The *full* config
    // (GetConfig(), just below) stays lazy: nothing in it logs today, but
    // keeping the same shape as blessed_cascades/blessed_gi avoids relying
    // on that staying true, after a namespace-scope eager version of this
    // config already broke static init once for the sibling classes (see
    // ComputeCascadeEnabled()'s comment in blessed_cascades.cpp).
    bool ComputeSceneCaptureEnabled() {
      std::string vs = env::getEnvVar("BLESSED_SCENE_VS");
      std::string rule = env::getEnvVar("BLESSED_SCENE_RULE");
      return !vs.empty() || rule == "depthonly";
    }

    // blessed: function-local magic static -- lazy on purpose. Only ever
    // reached once blessed_scene_capture_detail::g_enabled is already known
    // true (see IsEnabled() in blessed_scene_capture.h), i.e. from a real
    // DrawIndexed call, well after static init -- so the guard cost is off
    // the disabled path this seat exists to make free.
    const SceneConfig& GetConfig() {
      static SceneConfig s_config = [] {
        SceneConfig c;

        std::string vs = env::getEnvVar("BLESSED_SCENE_VS");
        std::string rule = env::getEnvVar("BLESSED_SCENE_RULE");

        if (!vs.empty()) {
          c.kind = SelectorKind::VsHash;
          for (const std::string& tok : SplitCsv(vs))
            c.vsHashes.push_back(HexPart(tok));
        } else if (rule == "depthonly") {
          c.kind = SelectorKind::DepthOnly;
        }

        // default 2:0 -- see BLESSED_SCENE_XFORM in fork-scene-capture.md
        std::string xform = env::getEnvVar("BLESSED_SCENE_XFORM");
        if (xform.empty() || !ParseSlotOffset(xform, c.xform)) {
          c.xform.slot   = 2;
          c.xform.offset = 0;
          c.xform.valid  = true;
        }

        // unset by default -- transforms are treated as absolute
        std::string camPos = env::getEnvVar("BLESSED_SCENE_CAMPOS");
        if (!camPos.empty())
          ParseSlotOffset(camPos, c.camPos);

        // blessed: skin-repair -- the pivot Skinned::GetBoneTransformMatrix
        // subtracts is the skinned vertex shader's own b12 byte 640, so the
        // default reads the vs stage. independent of BLESSED_SCENE_CAMPOS.
        std::string pivot = env::getEnvVar("BLESSED_SCENE_SKIN_PIVOT");
        if (pivot.empty() || !ParseSlotOffset(pivot, c.skinPivot)) {
          c.skinPivot.stage  = D3D11ShaderType::eVertex;
          c.skinPivot.slot   = 12;
          c.skinPivot.offset = 640;
          c.skinPivot.valid  = true;
        }

        return c;
      }();

      return s_config;
    }

    // ---- swapchain extent (single immediate context; see blessed_dump.cpp's
    // own comment on why this needs no locking beyond the D3D10DeviceLock
    // every call site here is already made under) ----
    uint32_t g_swapchainWidth  = 0;
    uint32_t g_swapchainHeight = 0;

    // ---- BLESSED_SCENE_LOG counters, reset every 120 presents ----
    bool                  g_logEnabled       = false;
    bool                  g_logChecked       = false;
    uint32_t              g_presentsInWindow = 0;
    std::atomic<uint64_t> g_captured         { 0 };
    std::atomic<uint64_t> g_skipped[size_t(BlessedSceneSkip::Count)] = { };
    std::ofstream         g_logFile;

    // blessed: actor-skinning counters, same window as the static ones above.
    std::atomic<uint64_t> g_skinnedCaptured  { 0 };
    std::atomic<uint64_t> g_skinnedSkipped[size_t(BlessedSkinSkip::Count)] = { };

    // blessed: skin-v2 -- depth-only pass tracking (app thread only, same
    // locking argument as the swapchain extent above)
    uint32_t    g_passId        = 0;       // id of the current/last depth-only pass
    uint32_t    g_framePassBase = 1;       // first pass id of this frame
    bool        g_inDepthOnly   = false;   // last indexed draw matched the selector
    const void* g_passDsv       = nullptr; // dsv of the current pass
    uint32_t    g_maskPass      = 0;       // pass before this frame's mask draw
    bool        g_maskSeen      = false;

    // blessed: skin-v2 -- BLESSED_SCENE_LOG only: per depth-only pass of the
    // current frame {static draws, skinned draws}, and the last full frame's
    // copy (written to scene.jsonl as "depth_passes").
    constexpr uint32_t MaxLoggedPasses = 12;
    uint32_t g_framePasses[MaxLoggedPasses][2] = { };
    uint32_t g_framePassCount = 0;
    uint32_t g_frameMaskIndex = UINT32_MAX;
    uint32_t g_lastPasses[MaxLoggedPasses][2] = { };
    uint32_t g_lastPassCount = 0;
    uint32_t g_lastMaskIndex = UINT32_MAX;

    // blessed: scene-cs -- the static column of depth_passes is counted on
    // the cs thread now (a static draw is only known to be captured there):
    // g_csPasses is cs-thread-only, and EndFrameOnCs publishes it to
    // g_csLastPasses, which the app thread's log line reads.
    uint32_t              g_csPasses[MaxLoggedPasses] = { };
    uint32_t              g_csPassCount = 0;
    std::atomic<uint32_t> g_csLastPasses[MaxLoggedPasses] = { };
    std::atomic<uint32_t> g_csLastPassCount = { 0u };

    // blessed: hook-cpu -- MatchesDepthOnly only reads state.om (rtvs, dsv,
    // dsState), so its result is identical for every draw between two real
    // OM rebinds. Cache it keyed on D3D11ContextStateOM::blessedOmGeneration:
    // a draw whose generation matches the last one costs one uint64 compare
    // instead of the dsState->Desc() + GetViewInfo + GetCommonTexture +
    // MipLevelExtent + IsSwapchainExtent chain below. App thread only, same
    // no-locking argument as the rest of this file's statics.
    uint64_t g_matchGeneration = UINT64_MAX; // never equals a real initial generation (0)
    bool     g_matchResult     = false;

    // blessed: hook-cpu-2 -- this frame's staged skinned draws (app thread
    // only), handed to the cs thread whole by TakeSkinBatch at present
    std::unique_ptr<BlessedSceneSkinBatch> g_skinBatch;
    uint32_t                               g_lateSkinned = 0;

    // blessed: hook-cpu-2 round two -- most skinned draws any frame has
    // staged so far. A batch is reserved to this when it is taken, so a
    // fresh batch (pool empty) allocates once instead of regrowing draw by
    // draw, and a recycled one never grows at all.
    uint32_t                               g_skinHighWater = 0;

    void ReserveSkinBatch(BlessedSceneSkinBatch& batch, uint32_t draws) {
      batch.draws.reserve(draws);
    }

    uint32_t* FramePassSlot() {
      uint32_t index = g_passId - g_framePassBase;

      if (g_passId < g_framePassBase || index >= MaxLoggedPasses)
        return nullptr;

      g_framePassCount = std::max(g_framePassCount, index + 1u);
      return g_framePasses[index];
    }

    void NoteSelectorResult(bool matched, const void* dsv) {
      if (!matched) {
        g_inDepthOnly = false;
        return;
      }

      if (!g_inDepthOnly || dsv != g_passDsv) {
        g_passId++;
        g_passDsv     = dsv;
        g_inDepthOnly = true;
      }
    }

    bool LogEnabled() {
      if (!g_logChecked) {
        g_logChecked = true;
        g_logEnabled = env::getEnvVar("BLESSED_SCENE_LOG") == "1"
                    && !env::getEnvVar("BLESSED_PROBE_DIR").empty();
      }

      return g_logEnabled;
    }

  }


  // blessed: definition for the header's inline IsEnabled() -- see
  // blessed_scene_capture.h. Independent of GetConfig() above on purpose
  // (see ComputeSceneCaptureEnabled()'s comment).
  namespace blessed_scene_capture_detail {
    extern const bool g_enabled = ComputeSceneCaptureEnabled();
  }


  bool BlessedSceneCapture::SkinnedEnabled() {
    static bool s_enabled = env::getEnvVar("BLESSED_SCENE_SKINNED") == "1";
    return s_enabled;
  }


  namespace {

    bool MatchesDepthOnly(const D3D11ContextState& state) {
      // no rtv bound
      for (uint32_t i = 0; i < state.om.maxRtv; i++) {
        if (state.om.rtvs[i].ptr())
          return false;
      }

      auto* dsv = state.om.dsv.ptr();
      if (!dsv)
        return false;

      D3D11_DEPTH_STENCIL_DESC dsDesc = { };
      dsDesc.DepthEnable    = TRUE;
      dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;

      if (auto* dsState = state.om.dsState.ptr())
        dsDesc = dsState->Desc();

      if (!(dsDesc.DepthEnable && dsDesc.DepthWriteMask == D3D11_DEPTH_WRITE_MASK_ALL))
        return false;

      D3D11_VK_VIEW_INFO vi = dsv->GetViewInfo();

      if (D3D11CommonTexture* tex = GetCommonTexture(vi.pResource)) {
        VkExtent3D ext = tex->MipLevelExtent(vi.Image.MinLevel);
        return BlessedSceneCapture::IsSwapchainExtent(ext.width, ext.height);
      }

      return false;
    }

  }


  bool BlessedSceneCapture::MatchesSelector(const D3D11ContextState& state) {
    const SceneConfig& cfg = GetConfig();

    switch (cfg.kind) {
      case SelectorKind::VsHash: {
        if (!state.vs.ptr())
          return false;

        std::string hex = HexPart(state.vs->GetCommonShader()->GetName());

        for (const std::string& tok : cfg.vsHashes) {
          if (tok.size() >= 8 && hex.compare(0, tok.size(), tok) == 0)
            return true;
          if (tok == hex)
            return true;
        }

        return false;
      }

      case SelectorKind::DepthOnly: {
        // blessed: skin-v2 -- every result feeds the pass tracker
        auto* dsv = state.om.dsv.ptr();

        // blessed: hook-cpu -- see g_matchGeneration's comment above
        bool matched;
        if (state.om.blessedOmGeneration == g_matchGeneration) {
          matched = g_matchResult;
        } else {
          matched = MatchesDepthOnly(state);
          g_matchGeneration = state.om.blessedOmGeneration;
          g_matchResult     = matched;
        }

        NoteSelectorResult(matched, dsv);
        return matched;
      }

      default:
        return false;
    }
  }


  BlessedSceneCbufferSlot BlessedSceneCapture::XformConfig() {
    return GetConfig().xform;
  }


  BlessedSceneCbufferSlot BlessedSceneCapture::CamPosConfig() {
    return GetConfig().camPos;
  }


  BlessedSceneCbufferSlot BlessedSceneCapture::SkinPivotConfig() {
    return GetConfig().skinPivot;
  }


  bool BlessedSceneCapture::SkinTraceEnabled() {
    static bool s_enabled = env::getEnvVar("BLESSED_SCENE_SKIN_TRACE") == "1";
    return s_enabled;
  }


  void BlessedSceneCapture::NotifySwapchainExtent(uint32_t width, uint32_t height) {
    g_swapchainWidth  = width;
    g_swapchainHeight = height;
  }


  bool BlessedSceneCapture::IsSwapchainExtent(uint32_t width, uint32_t height) {
    return width == g_swapchainWidth && height == g_swapchainHeight;
  }


  uint32_t BlessedSceneCapture::CurrentPass() {
    return g_passId;
  }


  void BlessedSceneCapture::NoteMaskDraw() {
    // blessed: skin-v2 -- the mask draw ends the current depth-only pass,
    // so depth-only draws after it start a new one
    g_inDepthOnly = false;

    if (g_maskSeen)
      return;

    g_maskSeen = true;
    g_maskPass = g_passId >= g_framePassBase ? g_passId : 0u;
    g_frameMaskIndex = g_maskPass ? g_maskPass - g_framePassBase : UINT32_MAX;
  }


  uint32_t BlessedSceneCapture::TakeMaskPass() {
    uint32_t maskPass = g_maskPass;

    if (LogEnabled()) {
      std::memcpy(g_lastPasses, g_framePasses, sizeof(g_lastPasses));
      g_lastPassCount = g_framePassCount;
      g_lastMaskIndex = g_frameMaskIndex;
      std::memset(g_framePasses, 0, sizeof(g_framePasses));
      g_framePassCount = 0;
    }

    g_frameMaskIndex = UINT32_MAX;
    g_maskPass       = 0;
    g_maskSeen       = false;
    g_inDepthOnly    = false;
    g_framePassBase  = g_passId + 1u;
    return maskPass;
  }


  bool BlessedSceneCapture::SkinnedDrawIsLate() {
    // g_maskPass is what TakeMaskPass will hand endFrame as skinnedPass;
    // selectSkinnedDraws only counts passes <= it, and only when it is
    // non-zero (0 falls back to the largest pass of the whole frame)
    return g_maskSeen && g_maskPass != 0u && g_passId > g_maskPass
        && !BlessedScene::skinAllPasses();
  }


  void BlessedSceneCapture::CountLateSkinnedDraw() {
    g_lateSkinned++;
  }


  bool BlessedSceneCapture::IsBonesBufferDesc(const D3D11_BUFFER_DESC& desc) {
    // the skinned path only accepts a b10 buffer of exactly this size (see
    // D3D11CommonContext::BlessedSceneCaptureSkinnedDraw's BadBonesSize)
    return desc.Usage == D3D11_USAGE_DYNAMIC
        && (desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER)
        && desc.ByteWidth == sizeof(float) * BlessedSceneSkinnedDraw::BonesFloats
        && IsEnabled() && SkinnedEnabled();
  }


  void BlessedSceneCapture::StageSkinnedDraw(BlessedSceneSkinnedDraw& draw) {
    using blessed::Call;
    using blessed::CallScope;

    if (!g_skinBatch) {
      CallScope<true> probe(Call::DrawIndexedSceneSkinnedAcquire);
      g_skinBatch = BlessedScene::acquireSkinBatch();
      ReserveSkinBatch(*g_skinBatch, g_skinHighWater);
    }

    BlessedSceneSkinBatch& batch = *g_skinBatch;

    if (batch.draws.size() == batch.draws.capacity()) {
      CallScope<true> probe(Call::DrawIndexedSceneSkinnedGrow);
      ReserveSkinBatch(batch, uint32_t(batch.draws.size()) * 2u + 16u);
    }

    CallScope<true> probe(Call::DrawIndexedSceneSkinnedPush);
    batch.draws.push_back(std::move(draw));
  }


  std::unique_ptr<BlessedSceneSkinBatch> BlessedSceneCapture::TakeSkinBatch() {
    if (g_lateSkinned) {
      if (!g_skinBatch)
        g_skinBatch = BlessedScene::acquireSkinBatch();

      g_skinBatch->lateDraws = g_lateSkinned;
      g_lateSkinned = 0;
    }

    if (g_skinBatch)
      g_skinHighWater = std::max(g_skinHighWater, uint32_t(g_skinBatch->draws.size()));

    return std::move(g_skinBatch);
  }


  void BlessedSceneCapture::RecordSkipped(BlessedSceneSkip reason) {
    if (LogEnabled())
      g_skipped[size_t(reason)].fetch_add(1, std::memory_order_relaxed);
  }


  uint32_t BlessedSceneCapture::FramePassIndex() {
    if (!LogEnabled())
      return ~0u;

    uint32_t index = g_passId - g_framePassBase;

    if (g_passId < g_framePassBase || index >= MaxLoggedPasses)
      return ~0u;

    return index;
  }


  namespace {

    // blessed: scene-cs -- LogEnabled's lazy init runs on the app thread
    // (FramePassIndex, called for every draw the static path hands over),
    // before the cs thread can see any record, so reading it here is safe.
    void CsCountSkipped(BlessedSceneSkip reason, size_t count) {
      if (g_logEnabled)
        g_skipped[size_t(reason)].fetch_add(count, std::memory_order_relaxed);
    }

    void CsCountCaptured(uint32_t passIndex, size_t count) {
      if (!g_logEnabled)
        return;

      g_captured.fetch_add(count, std::memory_order_relaxed);

      if (passIndex < MaxLoggedPasses) {
        g_csPasses[passIndex] += uint32_t(count);
        g_csPassCount = std::max(g_csPassCount, passIndex + 1u);
      }
    }

    // blessed: scene-cs -- the host mapping of the uniform buffer slice
    // bound at \p binding, if \p end bytes from the slice start fit in the
    // buffer; also returns the buffer's byte width. Same test the app
    // thread made against the D3D11Buffer (GetMapPtr is the buffer's
    // current allocation, which is what the cs thread has bound by the
    // time the draw is recorded; DxvkBuffer::info().size is ByteWidth).
    const uint8_t* BoundCbuffer(DxvkContext* ctx, uint32_t binding, uint32_t end, VkDeviceSize* byteWidth) {
      const DxvkBufferSlice& slice = ctx->blessedUniformBuffer(binding);
      const auto* base = reinterpret_cast<const uint8_t*>(slice.mapPtr(0));

      if (!base)
        return nullptr;

      *byteWidth = slice.buffer()->info().size;

      if (slice.offset() + end > *byteWidth)
        return nullptr;

      // blessed: perf-halfrate -- read on the cpu: keep its ring chunks cached
      slice.buffer()->blessedMarkCpuRead();
      return base;
    }

  }


  void BlessedSceneCapture::CaptureOnCs(DxvkContext* ctx, const BlessedSceneCsRecord& record,
    const VkDrawIndexedIndirectCommand* draws, size_t count) {
    // 5. dynamic vertex buffer -- the app thread already checked that a
    // buffer with a non-zero stride is bound at posBinding
    const DxvkBufferSlice& vbSlice = ctx->blessedVertexBuffer(record.posBinding);

    if (unlikely(vbSlice.buffer() == nullptr)) {
      CsCountSkipped(BlessedSceneSkip::NoPosition, count); // cannot happen: the bindings were applied
      return;
    }

    if (vbSlice.buffer()->info().blessedDynamic) {
      CsCountSkipped(BlessedSceneSkip::DynamicVertexBuffer, count);
      return;
    }

    // 6./7. topology, index format (worked out on the app thread)
    if (record.deferredA != BlessedSceneSkip::Count) {
      CsCountSkipped(record.deferredA, count);
      return;
    }

    // 8. dynamic index buffer -- bound, since 7. passed
    const DxvkBufferSlice& ibSlice = ctx->blessedIndexBuffer();

    if (unlikely(ibSlice.buffer() == nullptr)) {
      CsCountSkipped(BlessedSceneSkip::BadIndexFormat, count); // cannot happen, as above
      return;
    }

    if (ibSlice.buffer()->info().blessedDynamic) {
      CsCountSkipped(BlessedSceneSkip::DynamicIndexBuffer, count);
      return;
    }

    // 9. index offset alignment (worked out on the app thread)
    if (record.deferredB != BlessedSceneSkip::Count) {
      CsCountSkipped(record.deferredB, count);
      return;
    }

    // 10. negative base vertex, per draw. Every later check up to the
    // transform is the same for the whole batch, so the draws that pass
    // here are counted once below.
    size_t nonNegative = 0;

    for (size_t i = 0; i < count; i++) {
      if (draws[i].vertexOffset < 0)
        CsCountSkipped(BlessedSceneSkip::NegativeBaseVertex, 1u);
      else
        nonNegative++;
    }

    if (!nonNegative)
      return;

    // 11. transform (required), off the bound vs cbuffer -- always the vs
    // stage, as before, whatever stage prefix BLESSED_SCENE_XFORM carries.
    // These returns count nothing, as before.
    const SceneConfig& cfg = GetConfig();

    if (!cfg.xform.valid || cfg.xform.slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
      return;

    VkDeviceSize xformWidth = 0;
    const uint8_t* xformBase = BoundCbuffer(ctx,
      D3D11ShaderResourceMapping::computeCbvBinding(D3D11ShaderType::eVertex, cfg.xform.slot),
      cfg.xform.offset + 48u, &xformWidth);

    if (!xformBase)
      return;

    // blessed: layout check, see BlessedSceneCbufferSlot::size
    if (cfg.xform.size && xformWidth != cfg.xform.size)
      return;

    BlessedSceneDraw sceneDraw;
    std::memcpy(sceneDraw.transform, xformBase + cfg.xform.offset, 48u);

    // 12. skip distant lod. lod terrain/objects carry a 16x scale and a
    // transform that is not camera-relative like the rest (whiterun: a
    // ceiling of lod chunks at z +3525 swallowed every sun ray). scale above
    // BLESSED_SCENE_MAXSCALE (default 4) is not captured, for now.
    static const float s_maxScale = [] {
      std::string v = env::getEnvVar("BLESSED_SCENE_MAXSCALE");
      return v.empty() ? 4.0f : float(std::atof(v.c_str()));
    }();

    for (uint32_t r = 0; r < 3; r++) {
      const float* row = &sceneDraw.transform[4 * r];
      if (row[0] * row[0] + row[1] * row[1] + row[2] * row[2] > s_maxScale * s_maxScale)
        return;
    }

    // 13. camera position (optional), stage-aware -- see the old app-thread
    // comment: the real camera lives in the pixel shader's per-frame cbuffer.
    if (cfg.camPos.valid && cfg.camPos.slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
      VkDeviceSize camWidth = 0;
      const uint8_t* camBase = BoundCbuffer(ctx,
        D3D11ShaderResourceMapping::computeCbvBinding(cfg.camPos.stage, cfg.camPos.slot),
        cfg.camPos.offset + 12u, &camWidth);

      if (camBase) {
        std::memcpy(sceneDraw.camPos, camBase + cfg.camPos.offset, 12u);
        sceneDraw.hasCamPos = true;
      }
    }

    // 14. vertex count from the vb's byte width. The last vertex only needs
    // its own element, not a whole stride: (width - offset - element) /
    // stride + 1.
    VkDeviceSize vbByteWidth = vbSlice.buffer()->info().size;
    uint32_t     vbStride    = ctx->blessedVertexStride(record.posBinding);
    VkDeviceSize vbAbsOffset = vbSlice.offset() + record.posOffset;
    VkDeviceSize posElemSize = lookupFormatInfo(record.posFormat)->elementSize;
    uint32_t     vertexCount = vbAbsOffset + posElemSize <= vbByteWidth
      ? uint32_t((vbByteWidth - vbAbsOffset - posElemSize) / vbStride + 1u) : 0u;

    // same slice D3D11Buffer::GetBufferSlice(offset, length) made: offset
    // (already <= width here whenever a draw survives the range check
    // below), length clamped to what is left of the buffer
    VkDeviceSize vbLength = std::min(VkDeviceSize(vertexCount) * vbStride,
      vbAbsOffset <= vbByteWidth ? vbByteWidth - vbAbsOffset : VkDeviceSize(0));

    // the ib slice D3D11Buffer::GetBufferSlice(ib.offset) made
    VkDeviceSize ibByteWidth = ibSlice.buffer()->info().size;
    VkDeviceSize ibOffset    = std::min(VkDeviceSize(ibSlice.offset()), ibByteWidth);
    DxvkBufferSlice ib(ibSlice.buffer(), ibOffset, ibByteWidth - ibOffset);

    sceneDraw.vbStride    = vbStride;
    sceneDraw.vbFormat    = record.posFormat;
    sceneDraw.vertexCount = vertexCount;
    sceneDraw.indexType   = ctx->blessedIndexType();

    size_t captured = 0;

    for (size_t i = 0; i < count; i++) {
      const VkDrawIndexedIndirectCommand& draw = draws[i];

      if (draw.vertexOffset < 0)
        continue; // counted above

      // blessed: rt-lifetime bounds audit -- see the old app-thread comment:
      // a base vertex at or past vertexCount reads out of bounds even for
      // index value 0, and the real max index is not known cheaply.
      if (uint32_t(draw.vertexOffset) >= vertexCount) {
        CsCountSkipped(BlessedSceneSkip::BaseVertexOutOfRange, 1u);
        continue;
      }

      BlessedSceneDraw d = sceneDraw;
      d.vb         = DxvkBufferSlice(vbSlice.buffer(), vbAbsOffset, vbLength);
      d.ib         = ib;
      d.indexCount = draw.indexCount;
      d.startIndex = draw.firstIndex;
      d.baseVertex = draw.vertexOffset;

      ctx->blessedSceneAddDraw(d);
      captured++;
    }

    if (captured)
      CsCountCaptured(record.passIndex, captured);
  }


  void BlessedSceneCapture::EndFrameOnCs() {
    if (!g_logEnabled)
      return;

    for (uint32_t i = 0; i < MaxLoggedPasses; i++)
      g_csLastPasses[i].store(g_csPasses[i], std::memory_order_relaxed);

    g_csLastPassCount.store(g_csPassCount, std::memory_order_relaxed);

    std::memset(g_csPasses, 0, sizeof(g_csPasses));
    g_csPassCount = 0;
  }


  void BlessedSceneCapture::RecordSkinnedCaptured() {
    if (LogEnabled()) {
      g_skinnedCaptured.fetch_add(1, std::memory_order_relaxed);

      if (uint32_t* slot = FramePassSlot())
        slot[1]++;
    }
  }


  void BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip reason) {
    if (LogEnabled())
      g_skinnedSkipped[size_t(reason)].fetch_add(1, std::memory_order_relaxed);
  }


  void BlessedSceneCapture::OnPresent(DxvkDevice* device) {
    if (!IsEnabled() || !LogEnabled())
      return;

    if (++g_presentsInWindow < 120)
      return;

    g_presentsInWindow = 0;

    std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    if (!g_logFile.is_open())
      g_logFile.open(dir + "/scene.jsonl", std::ios::out | std::ios::app);

    if (!g_logFile.is_open())
      return;

    static const char* reasonNames[] = {
      "no_position", "bad_position_format", "dynamic_vb", "dynamic_ib",
      "bad_topology", "bad_index_format", "negative_base_vertex",
      "base_vertex_out_of_range", "misaligned_index_offset",
    };

    // blessed: actor-skinning -- see BlessedSkinSkip
    static const char* skinReasonNames[] = {
      "no_skin_position", "bad_skin_position_format", "dynamic_skin_buffer",
      "bad_topology", "bad_index_format", "negative_base_vertex", "base_vertex_out_of_range",
      "no_bones_buffer", "bad_bones_size", "has_world_matrix", "no_pivot",
    };

    uint64_t captured = g_captured.exchange(0, std::memory_order_relaxed);

    std::string line = "{\"captured\":" + std::to_string(captured) + ",\"skipped\":{";
    for (size_t i = 0; i < size_t(BlessedSceneSkip::Count); i++) {
      uint64_t n = g_skipped[i].exchange(0, std::memory_order_relaxed);
      if (i) line += ',';
      line += "\""; line += reasonNames[i]; line += "\":" + std::to_string(n);
    }
    line += "}";

    uint64_t skinnedCaptured = g_skinnedCaptured.exchange(0, std::memory_order_relaxed);
    line += ",\"skinned_captured\":" + std::to_string(skinnedCaptured) + ",\"skinned_skipped\":{";
    for (size_t i = 0; i < size_t(BlessedSkinSkip::Count); i++) {
      uint64_t n = g_skinnedSkipped[i].exchange(0, std::memory_order_relaxed);
      if (i) line += ',';
      line += "\""; line += skinReasonNames[i]; line += "\":" + std::to_string(n);
    }
    line += "}";

    // blessed: skin-v2 -- the last frame's depth-only passes in order, each
    // [static draws, skinned draws, 1 if it is the pass before the mask draw]
    //
    // blessed: scene-cs -- the static column comes from the cs thread's
    // last finished frame (g_csLastPasses), the skinned column and the mask
    // flag from the app thread's; with the cs thread behind, the two can be
    // one frame apart.
    uint32_t passCount = std::max(g_lastPassCount, g_csLastPassCount.load(std::memory_order_relaxed));

    line += ",\"depth_passes\":[";
    for (uint32_t i = 0; i < passCount; i++) {
      if (i) line += ',';
      line += "[" + std::to_string(g_csLastPasses[i].load(std::memory_order_relaxed)) + "," + std::to_string(g_lastPasses[i][1])
            + "," + (i == g_lastMaskIndex ? "1" : "0") + "]";
    }
    line += "]";

    // blessed: cs-side counters -- BlessedScene::m_stats is all atomics, so
    // reading (and, for the windowed ones, resetting) it from this thread
    // is safe even though BlessedScene itself lives on the cs thread.
    if (BlessedScene* scene = device ? device->blessedScene() : nullptr) {
      BlessedSceneStats& stats = scene->stats();

      line += ",\"blas_count\":" + std::to_string(stats.blasCount.load(std::memory_order_relaxed));
      line += ",\"blas_bytes\":" + std::to_string(stats.blasBytes.load(std::memory_order_relaxed));
      line += ",\"builds_this_window\":" + std::to_string(stats.buildsThisWindow.exchange(0, std::memory_order_relaxed));
      line += ",\"tlas_instances\":" + std::to_string(stats.tlasInstances.load(std::memory_order_relaxed));
      line += ",\"blas_dropped_cap\":" + std::to_string(stats.blasDroppedCap.exchange(0, std::memory_order_relaxed));
      line += ",\"skinned_blas\":" + std::to_string(stats.skinnedBlasCount.load(std::memory_order_relaxed));
      line += ",\"skinned_refits\":" + std::to_string(stats.skinnedRefits.exchange(0, std::memory_order_relaxed));
      line += ",\"skinned_dropped_cap\":" + std::to_string(stats.skinnedDroppedCap.exchange(0, std::memory_order_relaxed));

      // blessed: skin-repair -- shared-mesh instances, same-pose repeats,
      // and the average gpu time of skin dispatch + blas build/refit per frame
      line += ",\"skinned_shared\":" + std::to_string(stats.skinnedShared.exchange(0, std::memory_order_relaxed));
      line += ",\"skinned_duplicates\":" + std::to_string(stats.skinnedDuplicates.exchange(0, std::memory_order_relaxed));

      // blessed: skin-v2 -- draws dropped for being from another depth-only
      // pass, frames that fell back to the largest pass (no mask draw), and
      // full rebuilds of existing entries (a refit constraint changed)
      line += ",\"skinned_other_pass\":" + std::to_string(stats.skinnedOtherPass.exchange(0, std::memory_order_relaxed));
      line += ",\"skinned_no_mask\":" + std::to_string(stats.skinnedNoMask.exchange(0, std::memory_order_relaxed));
      line += ",\"skinned_rebuilds\":" + std::to_string(stats.skinnedRebuilds.exchange(0, std::memory_order_relaxed));
      // blessed: hook-cpu-2 -- should stay 0
      line += ",\"skinned_bones_unaddressable\":" + std::to_string(stats.skinnedBonesUnaddressable.exchange(0, std::memory_order_relaxed));

      uint64_t skinNs      = stats.skinnedGpuNs.exchange(0, std::memory_order_relaxed);
      uint32_t skinSamples = stats.skinnedGpuSamples.exchange(0, std::memory_order_relaxed);
      double   skinMs      = skinSamples ? double(skinNs) / double(skinSamples) / 1.0e6 : 0.0;
      line += ",\"skinned_gpu_ms\":" + str::format(skinMs);
    }

    line += "}";

    g_logFile << line << '\n';
    g_logFile.flush();
  }

}
