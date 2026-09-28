// blessed: app-thread half of ray-traced gi -- see blessed_gi.h
#include <algorithm>
#include <cfloat> // blessed: gi-bounds
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>

#include "blessed_gi.h"

#include "d3d11_context_imm.h"
#include "d3d11_buffer.h"
#include "d3d11_texture.h"
#include "d3d11_input_layout.h"
#include "d3d11_view_srv.h"
#include "d3d11_shader.h"

#include "../dxvk/dxvk_context.h"
#include "../dxvk/dxvk_device.h"
#include "../dxvk/dxvk_image.h"
#include "../dxvk/blessed/blessed_gi.h"
#include "../dxvk/blessed/blessed_scene.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/util_blessed_probe.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // blessed: gi v1 -- app-thread-only dedup cache: identifies a mesh the
    // same way BlessedScene::CacheKey would (D3D11Buffer* pointers here,
    // not yet resolved to DxvkBuffer*, since that resolution is cs-thread
    // work this cache exists to avoid paying every draw) and remembers the
    // last SRV pointer sent for it. A repeat draw of the same mesh with the
    // same bound texture (the overwhelming steady-state case: ~700-1,300 of
    // these draws a frame, same meshes every frame) short-circuits to a
    // single hash lookup + pointer compare -- no EmitCs, no Rc touch, no
    // heap allocation. Only a new mesh or a changed texture pays the cost
    // of crossing to the cs thread. Never cleared: a stale entry for a mesh
    // that stopped being drawn just goes unread, same tradeoff as
    // BlessedScene's own m_cache.
    struct GiMeshDedupKey {
      D3D11Buffer* vb         = nullptr;
      D3D11Buffer* ib         = nullptr;
      uint32_t     vbOffset   = 0;
      uint32_t     ibOffset   = 0;
      uint32_t     indexCount = 0;
      uint32_t     startIndex = 0;
      int32_t      baseVertex = 0;

      bool operator == (const GiMeshDedupKey& o) const {
        return vb == o.vb && ib == o.ib && vbOffset == o.vbOffset && ibOffset == o.ibOffset
            && indexCount == o.indexCount && startIndex == o.startIndex && baseVertex == o.baseVertex;
      }
    };

    struct GiMeshDedupKeyHash {
      size_t operator () (const GiMeshDedupKey& k) const {
        size_t h = std::hash<const void*>()(k.vb);
        auto mix = [&h] (size_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
        mix(std::hash<const void*>()(k.ib));
        mix(k.vbOffset);
        mix(k.ibOffset);
        mix(k.indexCount);
        mix(k.startIndex);
        mix(size_t(k.baseVertex));
        return h;
      }
    };

    std::unordered_map<GiMeshDedupKey, D3D11ShaderResourceView*, GiMeshDedupKeyHash> g_giMeshDedup;

    // blessed: hook-cpu -- single-entry fast path in front of the hash map
    // above: draw order from the engine is usually grouped by mesh, so the
    // overwhelmingly common case is "same key, same srv as the draw right
    // before this one". That case now costs one key comparison (7 scalar
    // fields) instead of hashing the key and walking the map's bucket. A
    // miss here still falls through to the real map below unchanged.
    GiMeshDedupKey            g_lastGiKey;
    D3D11ShaderResourceView*  g_lastGiSrv = nullptr;

    // blessed: gi v1 -- true for a unorm-typed diffuse texture (assumed
    // srgb-authored, Skyrim's own convention -- decoded manually in
    // blessed_gi_albedo.comp); false for one whose view format is already
    // srgb-typed (the hardware sampler already linearizes it).
    bool NeedsManualSrgbDecode(VkFormat fmt) {
      switch (fmt) {
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
          return false;
        default:
          return true;
      }
    }

  }


  /**
   * Gi v1: resolves this draw's mesh identity + bound diffuse texture, and
   * hands them to the cs thread if (and only if) that mesh's texture
   * changed since the last time this function saw it.
   *
   * Mirrors D3D11CommonContext::BlessedSceneCaptureDraw's own position-
   * stream/index-buffer acceptance checks (see d3d11_context.cpp) closely
   * enough that a mesh accepted there resolves to the identical
   * BlessedScene::CacheKey here -- deliberately not shared code (that
   * function also does camera/transform work and per-reason skip counters
   * this one has no use for); see the seat report for the risk this
   * duplication accepts (the two checks silently drifting). A method (not
   * a free function) so it inherits BlessedGi's own friend access to
   * D3D11CommonContext::EmitCs -- see the header's doc comment.
   */
  void BlessedGi::ResolveMeshAlbedo(D3D11ImmediateContext* ctx, const D3D11ContextState& state,
    const BlessedGiDrawIndices& indices) {
    if (!ctx)
      return;

    auto* layout = state.ia.inputLayout.ptr();
    if (!layout || !layout->HasBlessedPosition() || layout->HasBlessedSkinning())
      return; // no position stream, or an actor (no static geometry/albedo path -- see v0's fallback)

    const DxvkVertexAttribute& posAttr = layout->GetBlessedPosition();
    if (posAttr.format != VK_FORMAT_R32G32B32_SFLOAT
     && posAttr.format != VK_FORMAT_R32G32B32A32_SFLOAT
     && posAttr.format != VK_FORMAT_R16G16B16A16_SFLOAT)
      return;

    if (posAttr.binding >= state.ia.maxVbCount)
      return;

    const auto& vb = state.ia.vertexBuffers[posAttr.binding];
    if (!vb.buffer.ptr() || vb.stride == 0)
      return;

    if (state.ia.primitiveTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
      return;

    const auto& ib = state.ia.indexBuffer;
    if (!ib.buffer.ptr()
     || (ib.format != DXGI_FORMAT_R16_UINT && ib.format != DXGI_FORMAT_R32_UINT))
      return;

    if (indices.baseVertex < 0)
      return;

    D3D11ShaderResourceView* srv = state.srv[D3D11ShaderType::ePixel].views[0].ptr();
    if (!srv || srv->GetResourceType() != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
      return;

    UINT vbAbsOffset = vb.offset + posAttr.offset;

    GiMeshDedupKey key;
    key.vb         = vb.buffer.ptr();
    key.ib         = ib.buffer.ptr();
    key.vbOffset   = vbAbsOffset;
    key.ibOffset   = ib.offset;
    key.indexCount = indices.indexCount;
    key.startIndex = indices.startIndex;
    key.baseVertex = indices.baseVertex;

    // blessed: hook-cpu -- see g_lastGiKey's comment above
    if (key == g_lastGiKey && srv == g_lastGiSrv)
      return;

    g_lastGiKey = key;
    g_lastGiSrv = srv;

    auto it = g_giMeshDedup.find(key);
    if (it != g_giMeshDedup.end() && it->second == srv)
      return; // steady state: same mesh, same texture as last time -- nothing to send

    if (it != g_giMeshDedup.end())
      it->second = srv;
    else
      g_giMeshDedup.emplace(key, srv);

    Rc<DxvkImageView> view = srv->GetImageView();
    if (view == nullptr)
      return;

    BlessedGiMeshAlbedoNote note;
    note.vb              = vb.buffer->GetBufferSlice(vbAbsOffset);
    note.vbStride        = vb.stride;
    note.vbFormat        = posAttr.format;
    note.ib              = ib.buffer->GetBufferSlice(ib.offset);
    note.indexType       = ib.format == DXGI_FORMAT_R16_UINT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    note.indexCount      = indices.indexCount;
    note.startIndex      = indices.startIndex;
    note.baseVertex      = indices.baseVertex;
    note.textureView     = view;
    note.needsSrgbDecode = NeedsManualSrgbDecode(view->info().format);

    ctx->EmitCs([cNote = std::move(note)] (DxvkContext* dxvkCtx) mutable {
      dxvkCtx->blessedGiNoteMeshAlbedo(cNote);
    });
  }


  namespace {

    enum class BlessedGiMode : uint8_t {
      Off,
      Const,   // BLESSED_GI=const -- stage 0
      Probes,  // BLESSED_GI=probes -- stage 1
    };

    // blessed: swapchain extent, refreshed every present -- see
    // BlessedGi::NotifySwapchainExtent (mirrors BlessedSceneCapture's own
    // copy; each blessed consumer keeps its own to stay a self-contained,
    // zero-cost-when-disabled unit).
    uint32_t g_swapchainWidth  = 0;
    uint32_t g_swapchainHeight = 0;

    bool ParseFloats3(const std::string& s, float out[3], float defR, float defG, float defB) {
      out[0] = defR; out[1] = defG; out[2] = defB;
      if (s.empty())
        return false;

      size_t pos = 0;
      for (uint32_t i = 0; i < 3; i++) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (tok.empty())
          return false;
        out[i] = std::strtof(tok.c_str(), nullptr);
        if (comma == std::string::npos) {
          if (i + 1 != 3)
            return false;
        } else {
          pos = comma + 1;
        }
      }
      return true;
    }

    // blessed: "<vs|ps>:<slot>:<byte offset>" -- same convention as
    // blessed_shadow.cpp's BlessedShadowSlot, duplicated locally rather
    // than shared (each blessed consumer's config stays self-contained).
    struct BlessedGiSlot {
      bool             valid  = false;
      D3D11ShaderType  stage  = D3D11ShaderType::ePixel;
      uint32_t         slot   = 0;
      uint32_t         offset = 0;
      uint32_t         binding = 0; // blessed: gi-cs, DxvkContext uniform buffer index
    };

    // blessed: gi-cs -- the DxvkContext uniform buffer index D3D11 binds
    // stage/slot to (same mapping as D3D11CommonContext::BindConstantBuffer).
    // A slot past D3D11's 14 cbuffer slots can never be bound: invalid.
    BlessedGiSlot ResolveBinding(BlessedGiSlot s) {
      if (s.slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
        s.valid = false;
      if (s.valid)
        s.binding = D3D11ShaderResourceMapping::computeCbvBinding(s.stage, s.slot);
      return s;
    }

    BlessedGiSlot ParseSlot(const std::string& s, const BlessedGiSlot& fallback) {
      if (s.empty())
        return fallback;

      size_t firstColon = s.find(':');
      size_t secondColon = firstColon == std::string::npos ? std::string::npos : s.find(':', firstColon + 1);
      if (firstColon == std::string::npos || secondColon == std::string::npos)
        return fallback;

      std::string stageStr = s.substr(0, firstColon);
      BlessedGiSlot out;
      out.stage  = stageStr == "vs" ? D3D11ShaderType::eVertex : D3D11ShaderType::ePixel;
      out.slot   = std::strtoul(s.substr(firstColon + 1, secondColon - firstColon - 1).c_str(), nullptr, 10);
      out.offset = std::strtoul(s.substr(secondColon + 1).c_str(), nullptr, 10);
      out.valid  = true;
      return out;
    }

    // blessed: gi-cs -- the host mapping of the uniform buffer slice bound
    // at \p binding on the cs thread right now, if \p bytes from its start
    // fit in the buffer. Same test the app-thread version made against the
    // D3D11Buffer: slice offset (constantOffset * 16) + bytes <= ByteWidth
    // (DxvkBuffer::info().size is ByteWidth), and the mapping is the
    // buffer's current allocation, i.e. what D3D11Buffer::GetMapPtr gave
    // the app for the map this draw was issued after.
    uint8_t* BoundSliceBytes(DxvkContext* ctx, uint32_t binding, uint32_t bytes) {
      const DxvkBufferSlice& slice = ctx->blessedUniformBuffer(binding);
      auto* base = reinterpret_cast<uint8_t*>(slice.mapPtr(0));
      if (!base)
        return nullptr;

      if (slice.offset() + bytes > slice.buffer()->info().size)
        return nullptr;

      // blessed: perf-halfrate -- read on the cpu: keep its ring chunks cached
      slice.buffer()->blessedMarkCpuRead();
      return base;
    }

    // blessed: reads `count` floats out of a bound cbuffer -- same pattern
    // as blessed_shadow.cpp's ReadCbufferFloats, on the cs thread (gi-cs).
    bool ReadCbufferFloats(DxvkContext* ctx, const BlessedGiSlot& slot, float* out, uint32_t count) {
      if (!slot.valid)
        return false;

      const uint8_t* base = BoundSliceBytes(ctx, slot.binding, slot.offset + count * uint32_t(sizeof(float)));
      if (!base)
        return false;

      std::memcpy(out, base + slot.offset, count * sizeof(float));
      return true;
    }

    struct BlessedGiConfig {
      bool           enabled = false;
      BlessedGiMode  mode    = BlessedGiMode::Off;
      float          constColor[3] = { 1.0f, 0.0f, 0.0f }; // BLESSED_GI_CONST
      BlessedGiSlot  xform;   // BLESSED_GI_XFORM -- vs World translation
      BlessedGiSlot  camPos;  // BLESSED_GI_CAMPOS -- CameraPosAdjust
      bool           debugIrradiance = false; // BLESSED_GI_DEBUG=irradiance
      uint32_t       psB2Binding = 0; // blessed: gi-cs, where the patch goes
      bool           sampleBounds = false; // blessed: gi-bounds, BLESSED_GI_SAMPLE=bounds (probes only)
      uint32_t       boundsRefresh = 8;    // blessed: gi-bounds, BLESSED_GI_BOUNDS_REFRESH (frames, >= 1)

      static const BlessedGiConfig& Get();
    };

    // blessed: pure function of BLESSED_GI alone -- no Logger call, no
    // other static touched -- safe to run at static-init time (dll load)
    // for blessed_gi_detail::g_enabled below. The *full* config (Get(),
    // just below) has to stay lazy: its Logger::info call would otherwise
    // run during dxvk's own static init, before Logger's statics are ready
    // -- the crash a first deploy of this seat's earlier draft hit (game
    // exits before the main menu, a 0-byte, wrongly-named log file). See
    // ComputeCascadeEnabled()'s comment in blessed_cascades.cpp for the
    // same reasoning.
    bool ComputeGiEnabled() {
      std::string mode = env::getEnvVar("BLESSED_GI");
      return mode == "const" || mode == "probes";
    }

    // blessed: function-local magic static -- lazy on purpose. Only ever
    // reached once blessed_gi_detail::g_enabled is already known true (see
    // IsEnabled() in blessed_gi.h), i.e. from a real OnDraw call, well
    // after static init -- so the guard cost is off the disabled path, and
    // Logger::info here runs at a point where Logger is fully alive.
    const BlessedGiConfig& BlessedGiConfig::Get() {
      static BlessedGiConfig s_config = [] {
        BlessedGiConfig c;

        std::string mode = env::getEnvVar("BLESSED_GI");
        if (mode == "const")
          c.mode = BlessedGiMode::Const;
        else if (mode == "probes")
          c.mode = BlessedGiMode::Probes;
        else
          return c; // "off", unset, or unrecognized: stay disabled

        c.enabled = true;

        std::string constStr = env::getEnvVar("BLESSED_GI_CONST");
        ParseFloats3(constStr, c.constColor, 1.0f, 0.0f, 0.0f);

        // community shaders' Lighting.hlsl documents World at vs b2
        // packoffset(c0) -- gi-route.md step 4 ("verify in a dump, the
        // utility shader differs" -- not independently verified here).
        BlessedGiSlot xformDefault;
        xformDefault.valid = true;
        xformDefault.stage = D3D11ShaderType::eVertex;
        xformDefault.slot  = 2;
        xformDefault.offset = 0;
        c.xform = ResolveBinding(ParseSlot(env::getEnvVar("BLESSED_GI_XFORM"), xformDefault));

        // same CameraPosAdjust slot the shadow pass defaults to
        // (BLESSED_SHADOW_CAMPOS) -- PROJECT.md's lesson: only a *later*
        // pass's ps b12 holds it, and pass 115 is one.
        BlessedGiSlot camPosDefault;
        camPosDefault.valid = true;
        camPosDefault.stage = D3D11ShaderType::ePixel;
        camPosDefault.slot  = 12;
        camPosDefault.offset = 640;
        c.camPos = ResolveBinding(ParseSlot(env::getEnvVar("BLESSED_GI_CAMPOS"), camPosDefault));
        c.psB2Binding = D3D11ShaderResourceMapping::computeCbvBinding(D3D11ShaderType::ePixel, 2u);

        c.debugIrradiance = env::getEnvVar("BLESSED_GI_DEBUG") == "irradiance";

        // blessed: gi-bounds -- the dxvk side reads the same variable
        // (BlessedGiState), so both halves agree on the mode
        c.sampleBounds = c.mode == BlessedGiMode::Probes
          && env::getEnvVar("BLESSED_GI_SAMPLE") != "origin";

        std::string refreshStr = env::getEnvVar("BLESSED_GI_BOUNDS_REFRESH");
        if (!refreshStr.empty())
          c.boundsRefresh = std::clamp(uint32_t(std::strtoul(refreshStr.c_str(), nullptr, 10)), 1u, 1024u); // well under the bounds store's eviction age

        Logger::info(str::format("BlessedGi: enabled, mode=", mode,
          " const=(", c.constColor[0], ",", c.constColor[1], ",", c.constColor[2], ")",
          " sample=", c.sampleBounds ? "bounds" : "origin",
          c.sampleBounds ? str::format(" bounds_refresh=", c.boundsRefresh) : std::string()));

        return c;
      }();

      return s_config;
    }

    // blessed: per-120-present accounting for gi.jsonl
    struct LogState {
      bool          initDone       = false;
      std::ofstream file;
      dxvk::high_resolution_clock::time_point processStart;
      uint64_t      framesInWindow = 0;
      uint64_t      patchedAtWindowStart = 0; // blessed: gi-cs, see g_patchedTotal
    };

    LogState g_log;

    // blessed: gi-cs -- patched draws so far. Written by the cs thread only
    // (a relaxed load + store, no locked add), read by the app thread's
    // OnPresent, which logs the difference per 120-present window.
    std::atomic<uint64_t> g_patchedTotal = { 0u };

    // blessed: hook-cpu-2 -- the last sampleAmbient call's input and result.
    // Several draws of one object (its sub-meshes) share one World
    // translation, so the same absolute position is often sampled many
    // times in a row; sampleAmbient only reads that position and the cs
    // cache, and the cs cache only changes in refreshCsCache, which bumps
    // its generation. A bitwise-equal position under the same generation
    // reuses the stored row, so the result is exactly what a fresh sample
    // would return. gi-cs: cs-thread-only, like everything in CsState.
    struct SampleMemo {
      bool     valid      = false;
      bool     ok         = false;
      uint64_t generation = 0;
      float    pos[3]     = { };
      float    row[12]    = { };
    };

    // blessed: gi-cs -- PatchOnCs's own state, cs thread only. The lighting
    // capture happens once per cs cache generation, i.e. once per frame
    // (refreshCsCache runs in every present's ordered gi chunk), which is
    // what the app thread's once-per-present flag used to do.
    struct CsState {
      uint64_t   lightingGeneration = UINT64_MAX;
      SampleMemo memo;
    };

    CsState g_cs;

    // blessed: gi-bounds -- the cross-frame row memo in front of the
    // bounds sample, cs thread only. Keyed by a 64-bit fingerprint of the
    // batch's mesh key(s) (buffer cookies, offsets, stride, format, index
    // ranges), the World's 3x3 bits and the absolute translation (World
    // translation + camera, quantized to 1/4 unit: World is camera-relative,
    // so its own bits change whenever the camera moves). A row is reused
    // across frames until it is due:
    //   - its age reaches BLESSED_GI_BOUNDS_REFRESH frames. A new entry
    //     starts with a per-entry phase of age already (fingerprint mod N),
    //     so entries made in the same frame age out over N frames, not all
    //     at once, and stay staggered after that.
    //   - the grid scrolled since it was sampled, on its own phase frame
    //     (within N frames, again staggered), or at once when the grid
    //     jumped more than one cell on any axis (a teleport or load door).
    // Within one frame a row is reused only for a bitwise-equal absolute
    // position, so N = 1 recomputes every row every frame and writes the
    // same bits as the in-frame memo it replaces. Rows are only stored
    // once every mesh in the batch has settled bounds (known or failed).
    struct BoundsRowMemo {
      uint64_t fingerprint = 0;
      uint64_t sampledGen  = 0;
      float    exactPos[6] = { }; // World translation, camera
      int32_t  origin[3]   = { };
      uint32_t agePad      = 0;
      bool     used        = false;
      bool     ok          = false;
      float    row[12]     = { };
    };

    constexpr uint32_t BoundsRowMemoCount = 8192; // power of two
    BoundsRowMemo g_boundsRowMemo[BoundsRowMemoCount];

    inline uint64_t MixFingerprint(uint64_t h, uint64_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      return h * 0xff51afd7ed558ccdull;
    }

    inline uint32_t FloatBits(float f) {
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits));
      return bits;
    }

    // blessed: gi-bounds -- the app-thread half of the record: the same
    // position-stream and index checks ResolveMeshAlbedo makes (and
    // BlessedSceneCaptureDraw before it), minus the texture. A draw that
    // fails any of them stays on the origin sample.
    void FillBoundsRecord(const D3D11ContextState& state, const BlessedGiDrawIndices& indices,
      BlessedGiCsRecord& record) {
      if (!indices.valid || indices.baseVertex < 0)
        return;

      auto* layout = state.ia.inputLayout.ptr();
      if (!layout || !layout->HasBlessedPosition() || layout->HasBlessedSkinning())
        return;

      const DxvkVertexAttribute& posAttr = layout->GetBlessedPosition();
      if (posAttr.format != VK_FORMAT_R32G32B32_SFLOAT
       && posAttr.format != VK_FORMAT_R32G32B32A32_SFLOAT
       && posAttr.format != VK_FORMAT_R16G16B16A16_SFLOAT)
        return;

      if (posAttr.binding >= state.ia.maxVbCount)
        return;

      const auto& vb = state.ia.vertexBuffers[posAttr.binding];
      if (!vb.buffer.ptr() || vb.stride == 0)
        return;

      if (state.ia.primitiveTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
        return;

      const auto& ib = state.ia.indexBuffer;
      if (!ib.buffer.ptr()
       || (ib.format != DXGI_FORMAT_R16_UINT && ib.format != DXGI_FORMAT_R32_UINT))
        return;

      record.valid      = true;
      record.posFormat  = posAttr.format;
      record.posBinding = posAttr.binding;
      record.posOffset  = posAttr.offset;
    }

    void EnsureLogInit() {
      if (g_log.initDone)
        return;

      g_log.initDone    = true;
      g_log.processStart = dxvk::high_resolution_clock::now();

      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
      if (!dir.empty()) {
        std::string path = dir + env::PlatformDirSlash + "gi.jsonl";
        g_log.file.open(str::topath(path.c_str()), std::ios::out | std::ios::trunc);
      }
    }

    // blessed: overwrites bytes [176,224) of the bound ps b2 slice \p b2
    // (already checked to be >= 224 bytes by BoundSliceBytes -- gate on
    // cbuffer *shape*, not a shader-hash allowlist: ~25+ Lighting.hlsl
    // permutations share it, gi-route.md section 2's histogram) with \p row
    // (row-major float3x4, row c = (a_x,a_y,a_z,a_const)).
    //
    // gi-cs: \p b2 is the host mapping of the slice bound on the cs thread,
    // right before the draw is recorded: the DxvkResourceAllocation the
    // cbuffer's most recent D3D11_MAP_WRITE_DISCARD handed out (the
    // matching invalidateBuffer reached the cs thread in order, before
    // this draw). dxvk's memory allocator only ever calls a memory type
    // "cached" (its cachedDynamicResources profile, on for SkyrimSE.exe --
    // PROJECT.md) if it is *also* host-coherent
    // (DxvkMemoryAllocator::determineMemoryTypesWithPropertyFlags,
    // dxvk_memory.cpp), so a plain CPU store here needs no explicit flush
    // to become visible to the GPU, as long as it lands before the command
    // buffer this draw lives in is submitted -- and it lands before the
    // draw is even recorded.
    void WritePatchRow(uint8_t* b2, const float row[12]) {
      std::memcpy(b2 + 176u, row, 48u);
    }

    void PatchAmbient(uint8_t* b2, const float colour[3]) {
      // row_major float3x4: row r = (a_x, a_y, a_z, a_const). constant
      // ambient: a_const = colour[r], direction weights zero.
      float patch[12] = { };
      patch[3]  = colour[0];
      patch[7]  = colour[1];
      patch[11] = colour[2];
      WritePatchRow(b2, patch);
    }

    // blessed: captures this frame's pre-zero DirLightDirection (c0),
    // DirLightColor (c1) and DirectionalAmbient (c11-c13) off the *same*
    // ps b2 cbuffer the patch is about to overwrite -- all three live in
    // one pass-115 draw's cbuffer (gi-route.md section 2). Only the first
    // patched draw each frame bothers (see CsState::lightingGeneration).
    void ReadFrameLighting(const uint8_t* b2, BlessedGiFrameLighting& out) {
      std::memcpy(out.sunDir, b2 + 0u, 12u);
      std::memcpy(out.sunColor, b2 + 16u, 12u);
      std::memcpy(out.ambientRow, b2 + 176u, 48u);
      out.valid = true;
    }

    // blessed: identifies the main forward-lit pass (pass 115 in the frame
    // dump): 4 bound render targets, rtv0 R16G16B16A16_FLOAT at swapchain
    // size, a bound dsv. See docs/research/gi-route.md section 1.
    bool IsMainLitPassBound(const D3D11ContextState& state) {
      if (state.om.maxRtv < 4u || !state.om.dsv.ptr())
        return false;

      D3D11RenderTargetView* rtv0 = state.om.rtvs[0].ptr();
      if (!rtv0)
        return false;

      D3D11_RENDER_TARGET_VIEW_DESC desc = { };
      rtv0->GetDesc(&desc);
      if (desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT)
        return false;

      if (g_swapchainWidth == 0u || g_swapchainHeight == 0u)
        return false; // no swapchain seen yet -- can't confirm size

      const D3D11_VK_VIEW_INFO& vi = rtv0->GetViewInfo();
      D3D11CommonTexture* tex = GetCommonTexture(vi.pResource);
      if (!tex)
        return false;

      VkExtent3D ext = tex->MipLevelExtent(vi.Image.MinLevel);
      return ext.width == g_swapchainWidth && ext.height == g_swapchainHeight;
    }

    // blessed: hook-cpu -- IsMainLitPassBound only reads state.om, so its
    // result is identical for every draw between two real OM rebinds (see
    // D3D11ContextStateOM::blessedOmGeneration's comment). Cached the same
    // way BlessedSceneCapture caches MatchesDepthOnly: a draw outside the
    // main lit pass costs one uint64 compare instead of GetDesc + GetViewInfo
    // + GetCommonTexture + MipLevelExtent. App thread only (same as the rest
    // of this file's statics).
    uint64_t g_litPassGeneration = UINT64_MAX;
    bool     g_litPassResult     = false;

    bool IsMainLitPassBoundCached(const D3D11ContextState& state) {
      if (state.om.blessedOmGeneration == g_litPassGeneration)
        return g_litPassResult;

      g_litPassResult     = IsMainLitPassBound(state);
      g_litPassGeneration = state.om.blessedOmGeneration;
      return g_litPassResult;
    }

  }


  // blessed: definition for the header's inline IsEnabled() -- see
  // blessed_gi.h. Independent of BlessedGiConfig::Get() above on purpose
  // (see ComputeGiEnabled()'s comment).
  namespace blessed_gi_detail {
    extern const bool g_enabled = ComputeGiEnabled();
  }


  bool BlessedGi::OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device,
    const BlessedGiDrawIndices& indices, BlessedGiCsRecord* pRecord) {
    const BlessedGiConfig& cfg = BlessedGiConfig::Get();
    if (!cfg.enabled)
      return false;

    // blessed: gi-bounds -- a stale record must never ride on this draw
    if (cfg.sampleBounds && pRecord)
      pRecord->valid = false;

    EnsureLogInit();

    if (!IsMainLitPassBoundCached(state))
      return false;

    // blessed: gi-cs -- what's left on the app thread once the main lit
    // pass is bound: the mode check and the mesh/albedo note. The cbuffer
    // reads, the probe sample and the write all run in PatchOnCs.
    ::dxvk::blessed::CallScope<true> blessedProbe_giEmit(::dxvk::blessed::Call::DrawIndexedGiEmit);

    if (cfg.mode == BlessedGiMode::Const)
      return true;

    // Probes: without the dxvk-side state nothing could be sampled, and the
    // lighting note had nowhere to go either (blessedGiNoteLighting was a
    // no-op then), so the draw is left alone, as before.
    BlessedGiState* gi = device ? device->blessedGi() : nullptr;
    if (!gi)
      return false;

    // blessed: gi v1 -- this draw's mesh/texture, only when this mode
    // actually uses it (skip entirely under BLESSED_GI_V0=1 and for
    // non-indexed draws, which BlessedSceneCapture never keys either).
    // See ResolveMeshAlbedo's own dedup cache for why this is cheap in
    // the steady state.
    if (!gi->v0Mode() && indices.valid) {
      ::dxvk::blessed::CallScope<true> blessedProbe_giResolve(::dxvk::blessed::Call::DrawIndexedGiResolve); // blessed: hook-cpu-2
      ResolveMeshAlbedo(ctx, state, indices);
    }

    // blessed: gi-bounds -- where the cs side finds this draw's positions
    if (cfg.sampleBounds && pRecord && gi->sampleBounds())
      FillBoundsRecord(state, indices, *pRecord);

    return true;
  }


  void BlessedGi::PatchOnCs(DxvkContext* ctx, size_t drawCount) {
    const BlessedGiConfig& cfg = BlessedGiConfig::Get();

    uint8_t* b2 = BoundSliceBytes(ctx, cfg.psB2Binding, 224u);
    if (!b2)
      return;

    bool patched = false;

    if (cfg.mode == BlessedGiMode::Const) {
      PatchAmbient(b2, cfg.constColor);
      patched = true;
    } else {
      BlessedGiState* gi = ctx->blessedGi();
      if (!gi)
        return;

      uint64_t generation = gi->csCacheGeneration();

      // blessed: capture this frame's pre-zero lighting once, off the first
      // patched draw, before its own patch lands -- see BlessedGiFrameLighting.
      if (g_cs.lightingGeneration != generation) {
        g_cs.lightingGeneration = generation;

        BlessedGiFrameLighting lighting;
        ReadFrameLighting(b2, lighting);
        gi->noteFrameLighting(lighting);
      }

      // blessed: sample the traced probe grid at this draw's world
      // position -- vs b2's World translation (BLESSED_GI_XFORM, row-major
      // 3x4: translation is column 3 of each row, i.e. bytes 12/28/44)
      // plus this draw's own camera position (BLESSED_GI_CAMPOS) gives a
      // true absolute world position, independent of which frame's camera
      // either side happened to read it relative to.
      float world3x4[12];
      float camPosAbs[3];

      if (!ReadCbufferFloats(ctx, cfg.xform, world3x4, 12)
       || !ReadCbufferFloats(ctx, cfg.camPos, camPosAbs, 3))
        return;

      float absPos[3] = {
        world3x4[3]  + camPosAbs[0],
        world3x4[7]  + camPosAbs[1],
        world3x4[11] + camPosAbs[2],
      };

      // blessed: hook-cpu-2 -- see SampleMemo
      SampleMemo& memo = g_cs.memo;
      if (!memo.valid || memo.generation != generation || std::memcmp(memo.pos, absPos, sizeof(absPos))) {
        memo.ok         = gi->sampleAmbient(absPos, memo.row);
        memo.valid      = true;
        memo.generation = generation;
        std::memcpy(memo.pos, absPos, sizeof(absPos));
      }

      if (!memo.ok)
        return;

      float row[12];
      std::memcpy(row, memo.row, sizeof(row));

      if (cfg.debugIrradiance || gi->debugIrradiance()) {
        // blessed: BLESSED_GI_DEBUG=irradiance -- albedo x irradiance
        // with a white albedo isn't available here (the shader still
        // multiplies by the game's own diffuse texture), so boost the
        // written ambient instead to make probe differences visible.
        for (float& v : row)
          v *= 4.0f;
      }

      WritePatchRow(b2, row);
      patched = true;
    }

    if (patched) {
      g_patchedTotal.store(g_patchedTotal.load(std::memory_order_relaxed) + drawCount,
        std::memory_order_relaxed);
    }
  }


  void BlessedGi::PatchOnCsBounds(DxvkContext* ctx, const BlessedGiCsRecord& record,
    const VkDrawIndexedIndirectCommand* draws, size_t drawCount) {
    // blessed: gi-bounds -- probes mode only (a record is only ever valid
    // then). The lighting capture, cbuffer reads and origin fallback are
    // PatchOnCs's own, kept as a copy so origin mode's path stays untouched.
    const BlessedGiConfig& cfg = BlessedGiConfig::Get();

    uint8_t* b2 = BoundSliceBytes(ctx, cfg.psB2Binding, 224u);
    if (!b2)
      return;

    BlessedGiState* gi = ctx->blessedGi();
    if (!gi)
      return;

    uint64_t generation = gi->csCacheGeneration();

    if (g_cs.lightingGeneration != generation) {
      g_cs.lightingGeneration = generation;

      BlessedGiFrameLighting lighting;
      ReadFrameLighting(b2, lighting);
      gi->noteFrameLighting(lighting);
    }

    float world3x4[12];
    float camPosAbs[3];

    if (!ReadCbufferFloats(ctx, cfg.xform, world3x4, 12)
     || !ReadCbufferFloats(ctx, cfg.camPos, camPosAbs, 3))
      return;

    const DxvkBufferSlice& vbSlice = ctx->blessedVertexBuffer(record.posBinding);
    const DxvkBufferSlice& ibSlice = ctx->blessedIndexBuffer();
    bool haveBuffers = vbSlice.buffer() != nullptr && ibSlice.buffer() != nullptr;

    float row[12];
    bool  ok = false;

    // exact inputs of the same-frame check: the World translation and the
    // camera, the two terms every sample adds (see BoundsRowMemo)
    float exactPos[6] = {
      world3x4[3], world3x4[7], world3x4[11],
      camPosAbs[0], camPosAbs[1], camPosAbs[2],
    };

    BoundsRowMemo* memo = nullptr;
    uint64_t fingerprint = 0;
    const int32_t* origin = gi->csCacheOriginCell();

    if (haveBuffers) {
      uint64_t h = 0x243f6a8885a308d3ull;
      h = MixFingerprint(h, vbSlice.buffer()->cookie());
      h = MixFingerprint(h, ibSlice.buffer()->cookie());
      h = MixFingerprint(h, uint64_t(vbSlice.offset() + record.posOffset));
      h = MixFingerprint(h, uint64_t(ibSlice.offset()));
      h = MixFingerprint(h, uint64_t(ctx->blessedVertexStride(record.posBinding))
                          | uint64_t(record.posFormat) << 32);
      h = MixFingerprint(h, uint64_t(ctx->blessedIndexType()));

      for (size_t i = 0; i < drawCount; i++) {
        h = MixFingerprint(h, uint64_t(draws[i].indexCount) | uint64_t(draws[i].firstIndex) << 32);
        h = MixFingerprint(h, uint64_t(uint32_t(draws[i].vertexOffset)));
      }

      for (uint32_t r = 0; r < 3; r++) {
        const float* m = &world3x4[4 * r];
        h = MixFingerprint(h, uint64_t(FloatBits(m[0])) | uint64_t(FloatBits(m[1])) << 32);
        // quantized absolute translation: 1/4 unit, see BoundsRowMemo
        int32_t q = int32_t(std::floor((m[3] + camPosAbs[r]) * 4.0f));
        h = MixFingerprint(h, uint64_t(FloatBits(m[2])) | uint64_t(uint32_t(q)) << 32);
      }

      fingerprint = h;
      memo = &g_boundsRowMemo[uint32_t(h) & (BoundsRowMemoCount - 1u)];

      if (memo->used && memo->fingerprint == fingerprint) {
        bool reuse;

        if (memo->sampledGen == generation) {
          // same frame: only for bitwise-equal translation and camera
          reuse = !std::memcmp(memo->exactPos, exactPos, sizeof(exactPos));
        } else {
          uint32_t n   = cfg.boundsRefresh;
          uint64_t age = generation - memo->sampledGen + memo->agePad;
          bool scrolled = memo->origin[0] != origin[0] || memo->origin[1] != origin[1] || memo->origin[2] != origin[2];
          bool jumped   = std::abs(memo->origin[0] - origin[0]) > 1
                       || std::abs(memo->origin[1] - origin[1]) > 1
                       || std::abs(memo->origin[2] - origin[2]) > 1;
          uint32_t phase = uint32_t(fingerprint >> 32) % n;
          bool due = age >= n || jumped || (scrolled && (generation + phase) % n == 0);
          reuse = !due;
        }

        if (reuse) {
          if (!memo->ok)
            return;

          std::memcpy(row, memo->row, sizeof(row));
          ok = true;
        }
      }
    }

    if (!ok) {
      // the union of every known mesh aabb in the batch: the draws share
      // one bound ps b2, so they get one row, and the vb/ib bindings, so
      // only the index range differs between them
      float bounds[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
      bool  haveBounds = false;
      bool  allSettled = haveBuffers;

      if (haveBuffers) {
        BlessedGiBoundsRequest req;
        req.vb        = vbSlice.buffer().ptr();
        req.vbOffset  = vbSlice.offset() + record.posOffset;
        req.vbStride  = ctx->blessedVertexStride(record.posBinding);
        req.vbFormat  = record.posFormat;
        req.ib        = ibSlice.buffer().ptr();
        req.ibOffset  = ibSlice.offset();
        req.indexType = ctx->blessedIndexType();

        for (size_t i = 0; i < drawCount; i++) {
          req.indexCount = draws[i].indexCount;
          req.startIndex = draws[i].firstIndex;
          req.baseVertex = draws[i].vertexOffset;

          bool settled = true;
          const float* b = gi->lookupMeshBounds(req, &settled);
          allSettled &= settled;

          if (!b)
            continue;

          for (int a = 0; a < 3; a++) {
            bounds[a]     = std::min(bounds[a],     b[a]);
            bounds[a + 3] = std::max(bounds[a + 3], b[a + 3]);
          }

          haveBounds = true;
        }
      }

      if (haveBounds)
        ok = gi->sampleAmbientBounds(world3x4, camPosAbs, bounds, bounds + 3, row);

      if (!ok) {
        // the origin sample, exactly as PatchOnCs takes it
        float absPos[3] = {
          world3x4[3]  + camPosAbs[0],
          world3x4[7]  + camPosAbs[1],
          world3x4[11] + camPosAbs[2],
        };

        SampleMemo& originMemo = g_cs.memo;
        if (!originMemo.valid || originMemo.generation != generation || std::memcmp(originMemo.pos, absPos, sizeof(absPos))) {
          originMemo.ok         = gi->sampleAmbient(absPos, originMemo.row);
          originMemo.valid      = true;
          originMemo.generation = generation;
          std::memcpy(originMemo.pos, absPos, sizeof(absPos));
        }

        ok = originMemo.ok;
        if (ok)
          std::memcpy(row, originMemo.row, sizeof(row));
      }

      // store only a final answer: every mesh settled, so a later frame
      // would take the same path
      if (memo && allSettled) {
        bool fresh = !memo->used || memo->fingerprint != fingerprint;

        memo->used        = true;
        memo->fingerprint = fingerprint;
        memo->agePad      = fresh ? uint32_t(fingerprint >> 32) % cfg.boundsRefresh : 0u;
        memo->sampledGen  = generation;
        memo->ok          = ok;
        std::memcpy(memo->exactPos, exactPos, sizeof(exactPos));
        std::memcpy(memo->origin, origin, sizeof(memo->origin));
        if (ok)
          std::memcpy(memo->row, row, sizeof(row));
      }

      if (!ok)
        return;
    }

    if (cfg.debugIrradiance || gi->debugIrradiance()) {
      for (float& v : row)
        v *= 4.0f;
    }

    WritePatchRow(b2, row);

    g_patchedTotal.store(g_patchedTotal.load(std::memory_order_relaxed) + drawCount,
      std::memory_order_relaxed);
  }


  void BlessedGi::OnPresent(DxvkDevice* device) {
    const BlessedGiConfig& cfg = BlessedGiConfig::Get();
    if (!cfg.enabled)
      return;

    // blessed: gi-cs -- the per-frame lighting flag, the sample memo and the
    // probe cache refresh all moved to the cs thread (PatchOnCs and
    // BlessedGiState::refreshCsCache, run from DxvkContext::blessedRunGiTrace).

    g_log.framesInWindow++;
    if (g_log.framesInWindow < 120)
      return;

    // blessed: gi-cs -- counted on the cs thread, so this window's count
    // lags the app thread by however far the cs thread is behind (at most
    // a frame or so), and evens out across windows.
    uint64_t patchedTotal    = g_patchedTotal.load(std::memory_order_relaxed);
    uint64_t patchedInWindow = patchedTotal - g_log.patchedAtWindowStart;
    g_log.patchedAtWindowStart = patchedTotal;

    if (g_log.file.is_open()) {
      double t = std::chrono::duration<double>(
        dxvk::high_resolution_clock::now() - g_log.processStart).count();

      std::string line = str::format("{\"t\":", t,
        ",\"frames\":", g_log.framesInWindow,
        ",\"patched_per_frame\":", double(patchedInWindow) / double(g_log.framesInWindow),
        ",\"mode\":\"", (cfg.mode == BlessedGiMode::Const ? "const" : "probes"), "\"");

      // blessed: probes-only fields the dxvk-side tracer owns -- see
      // BlessedGiState::probesActive/raysPerProbe/lastTraceGpuMs.
      if (cfg.mode == BlessedGiMode::Probes) {
        BlessedGiState* gi = device ? device->blessedGi() : nullptr;
        if (gi) {
          line += str::format(",\"probes_active\":", gi->probesActive(),
            ",\"rays_per_probe\":", gi->raysPerProbe(),
            ",\"trace_gpu_ms\":", gi->lastTraceGpuMs(),
            // blessed: gi v1
            ",\"albedo_textures\":", gi->albedoTexturesActive(),
            ",\"probes_invalid\":", gi->countInvalidProbes());

          // blessed: gi-bounds -- only under sample=bounds, so origin's
          // line stays as it was
          if (gi->sampleBounds()) {
            line += str::format(",\"sample\":\"bounds\"",
              ",\"meshes_with_bounds\":", gi->meshesWithBounds(),
              ",\"probe_openness_mean\":", gi->meanOpenness());
          }
        }

        // blessed: gi v1 -- BlessedScene's own count (cs-thread-owned map,
        // read here the same way every other BlessedSceneStats field is).
        if (BlessedScene* scene = device ? device->blessedScene() : nullptr) {
          line += str::format(",\"meshes_with_albedo\":",
            scene->stats().giMeshesWithAlbedo.load(std::memory_order_relaxed));
        }
      }

      line += "}\n";
      g_log.file << line;
      g_log.file.flush();
    }

    g_log.framesInWindow  = 0;
  }


  void BlessedGi::NotifySwapchainExtent(uint32_t width, uint32_t height) {
    g_swapchainWidth  = width;
    g_swapchainHeight = height;
  }

}
