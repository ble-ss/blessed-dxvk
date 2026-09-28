// blessed: cached static sun shadow cascades, d3d11 side -- see blessed_cascade_cache.h
#include "blessed_cascade_cache.h"
#include "blessed_cascade_bounds.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d11_buffer.h"
#include "d3d11_context_imm.h"
#include "d3d11_depth_stencil.h"
#include "d3d11_input_layout.h"
#include "d3d11_rasterizer.h"
#include "d3d11_shader.h"
#include "d3d11_texture.h"
#include "d3d11_view_dsv.h"
#include "d3d11_view_srv.h"

#include "../dxvk/blessed/blessed_cascade_cache.h"

#include "../util/util_bit.h"
#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  // blessed: the immediate context's protected members this feature
  // uses. d3d11_context.h names this struct a friend.
  struct BlessedCascadeCacheCtx {
    template<typename Fn>
    static void Emit(D3D11ImmediateContext* ctx, Fn&& fn) {
      ctx->EmitCs(std::forward<Fn>(fn));
    }

    static void Rebind(D3D11ImmediateContext* ctx) {
      ctx->BindFramebuffer();
    }

    static void Reset(D3D11ImmediateContext* ctx) {
      ctx->ResetDirtyTracking();
      ctx->ResetCommandListState();
    }

    static void Restore(D3D11ImmediateContext* ctx) {
      ctx->RestoreCommandListState();
    }

    static void XferEvent(D3D11ImmediateContext* ctx) {
      ctx->BlessedGpuPassEvent(BlessedGpuPassKind::Xfer);
    }

    static DxvkDevice* Device(D3D11ImmediateContext* ctx) {
      return ctx->m_device.ptr();
    }
  };

  namespace {

    constexpr uint32_t MaxSlices = 4u;

    // b12 (PerFrame): CameraViewProj at c8, CameraPosAdjust at c40
    // (vendor/community-shaders/package/Shaders/Common/FrameBuffer.hlsli)
    constexpr uint32_t PerFrameSlot     = 12u;
    constexpr uint32_t ViewProjOffset   = 128u;
    constexpr uint32_t ViewProjBytes    = 64u;
    constexpr uint32_t PosAdjustOffset  = 640u;
    constexpr uint32_t PosAdjustBytes   = 16u;
    constexpr uint32_t MaxProjBytes     = 1024u;

    // Utility.hlsl's PerGeometry (World, TreeParams) and the bone buffers
    // (Common/Skinned.hlsli). A key hashes every cbuffer the draw's
    // shaders read except b12, which the projection check covers.
    constexpr uint32_t GeometrySlot = 2u;
    constexpr uint32_t BoneSlots    = (1u << 9u) | (1u << 10u);
    constexpr uint32_t MaxCbBytes   = 4096u;

    // Utility.hlsl PerGeometry: row_major float4x4 World at c1, so its
    // translation sits at c1.w, c2.w, c3.w
    constexpr uint32_t GeometryBytes = 64u;
    constexpr std::array<uint32_t, 3> TranslationBytes = { 28u, 44u, 60u };

    /// What StaticKey learns about a cacheable draw
    struct StaticDraw {
      uint64_t       key      = 0u;  ///< everything but World's translation
      uint64_t       identity = 0u;  ///< the mesh alone
      uint64_t       instance = 0u;  ///< key + world position, quantized
      double         pos[3]   = { }; ///< world position (translation + posAdjust)
      const uint8_t* geom     = nullptr;
      uint32_t       geomSize = 0u;
    };

    enum class Mode : uint32_t {
      None,     // no cascade open on this layer
      Pending,  // cleared, waiting for the first draw to pick a mode
      Pass,
      Build,
      Reuse,
      Verify,   // reuse's bookkeeping, but the cached keys are redrawn into scratch and compared
    };

    enum class Debug : uint32_t {
      Off,
      Redrawn,  // reuse without the copy: only redrawn casters shadow
      Cached,   // reuse without the redraws: only cached casters shadow
      Build,    // rebuild every frame (exercises the composite path)
      Pass,     // never cache (prices the hooks)
    };

    enum Reason : uint32_t {
      ReasonFirst,    // nothing cached yet
      ReasonProj,     // the cascade's projection changed
      ReasonMissing,  // a cached key was not drawn again
      ReasonMoved,    // a cached mesh came back with a different key
      ReasonPromote,  // new static draws stayed long enough to absorb
      ReasonTolerate, // missing keys outlived BLESSED_CASCADE_CACHE_TOLERATE
      ReasonDebug,
      ReasonCount,
    };

    const char* ReasonName(uint32_t r) {
      switch (r) {
        case ReasonFirst:    return "first";
        case ReasonProj:     return "proj";
        case ReasonMissing:  return "missing";
        case ReasonMoved:    return "moved";
        case ReasonPromote:  return "promote";
        case ReasonTolerate: return "tolerate";
        case ReasonDebug:    return "debug";
        default:             return "?";
      }
    }

    struct CascadeConfig {
      uint32_t  srvSlot       = 4u;
      uint32_t  stable        = 2u;
      uint32_t  promoteFrames = 30u;
      uint32_t  tolerate      = 0u;
      uint32_t  hotFrames     = 120u;
      uint32_t  verifyEvery   = 0u;
      uint32_t  sliceMask     = ~0u;
      uint32_t  minSize       = 1024u;
      bool      projFull      = false;
      float     tolTexels     = 0.5f;   // 0: exact (bitwise projection match)
      float     tolDepth      = -1.0f;  // d16 steps; < 0: half the casters' own depth bias
      float     posTol        = 0.05f;  // world units a cached caster may sit from its build position
      bool      restoreDraw   = false;  // restore by a fullscreen depth write instead of a copy
      bool      restorePartial = false; // restore only what last frame's other casters touched
      float     partialMaxCover = 0.4f; // above this share of the layer: the full restore
      uint32_t  partialMaxRects = 128u;
      Debug     debugMode     = Debug::Off;
      std::vector<std::string> skipVs;
    };

    float EnvF32(const char* name, float def) {
      std::string v = env::getEnvVar(name);
      return v.empty() || v == "auto" ? def : std::strtof(v.c_str(), nullptr);
    }

    uint32_t EnvU32(const char* name, uint32_t def) {
      std::string v = env::getEnvVar(name);
      return v.empty() ? def : uint32_t(std::strtoul(v.c_str(), nullptr, 10));
    }

    std::vector<std::string> EnvList(const char* name) {
      std::vector<std::string> out;
      std::string v = env::getEnvVar(name);
      size_t pos = 0;

      while (pos < v.size()) {
        size_t comma = v.find(',', pos);
        std::string tok = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);

        if (!tok.empty())
          out.push_back(tok);

        if (comma == std::string::npos)
          break;
        pos = comma + 1;
      }

      return out;
    }

    // blessed: env only, no Logger -- runs at dll load (see the header)
    bool ComputeEnabled() {
      return env::getEnvVar("BLESSED_CASCADE_CACHE") == "1"
          && env::getEnvVar("BLESSED_SKIP_CASCADES") != "1";
    }

    // blessed: lazy, first reached from a real draw (Logger is alive then)
    const CascadeConfig& GetConfig() {
      static CascadeConfig s_config = [] {
        CascadeConfig c;
        c.srvSlot       = EnvU32("BLESSED_CASCADE_CACHE_SRV", 4u);
        c.stable        = std::max(1u, EnvU32("BLESSED_CASCADE_CACHE_STABLE", 2u));
        c.promoteFrames = EnvU32("BLESSED_CASCADE_CACHE_PROMOTE", 30u);
        c.tolerate      = EnvU32("BLESSED_CASCADE_CACHE_TOLERATE", 0u);
        c.hotFrames     = EnvU32("BLESSED_CASCADE_CACHE_HOT", 120u);
        c.tolTexels     = EnvF32("BLESSED_CASCADE_CACHE_TOL_TEXELS", 0.5f);
        c.tolDepth      = EnvF32("BLESSED_CASCADE_CACHE_TOL_DEPTH", -1.0f);
        c.posTol        = EnvF32("BLESSED_CASCADE_CACHE_POS_TOL", 0.05f);
        c.restoreDraw   = env::getEnvVar("BLESSED_CASCADE_CACHE_RESTORE") == "draw";
        c.restorePartial = env::getEnvVar("BLESSED_CASCADE_CACHE_RESTORE") == "partial";
        c.partialMaxCover = EnvF32("BLESSED_CASCADE_CACHE_PARTIAL_MAX", 0.4f);
        c.partialMaxRects = EnvU32("BLESSED_CASCADE_CACHE_PARTIAL_RECTS", 128u);
        c.verifyEvery   = EnvU32("BLESSED_CASCADE_CACHE_VERIFY", 0u);
        c.minSize       = EnvU32("BLESSED_CASCADE_CACHE_MIN_SIZE", 1024u);
        c.projFull      = env::getEnvVar("BLESSED_CASCADE_CACHE_PROJ") == "full";
        c.skipVs        = EnvList("BLESSED_CASCADE_CACHE_SKIP_VS");

        std::vector<std::string> slices = EnvList("BLESSED_CASCADE_CACHE_SLICES");
        if (!slices.empty()) {
          c.sliceMask = 0u;
          for (const auto& s : slices)
            c.sliceMask |= 1u << (std::strtoul(s.c_str(), nullptr, 10) & 31u);
        }

        std::string dbg = env::getEnvVar("BLESSED_CASCADE_CACHE_DEBUG");
        if (dbg == "redrawn")     c.debugMode = Debug::Redrawn;
        else if (dbg == "cached") c.debugMode = Debug::Cached;
        else if (dbg == "build")  c.debugMode = Debug::Build;
        else if (dbg == "pass")   c.debugMode = Debug::Pass;

        Logger::info(str::format("BlessedCascadeCache: enabled, srv slot ", c.srvSlot,
          ", stable ", c.stable, ", promote ", c.promoteFrames, ", tolerate ", c.tolerate,
          ", verify ", c.verifyEvery, ", proj ", c.projFull ? "full" : "vp",
          ", tol ", c.tolTexels, " texels / ", c.tolDepth, " d16 steps (< 0: auto), pos tol ", c.posTol,
          ", debug ", dbg.empty() ? "off" : dbg));
        return c;
      }();

      return s_config;
    }

    // ---- hashing ----

    struct Hasher {
      uint64_t h = 0x9e3779b97f4a7c15ull;

      void u64(uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
      }

      void ptr(const void* p) {
        u64(uint64_t(reinterpret_cast<uintptr_t>(p)));
      }

      void bytes(const void* data, size_t size) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        size_t i = 0;

        for (; i + 8 <= size; i += 8) {
          uint64_t v;
          std::memcpy(&v, p + i, 8);
          u64(v);
        }

        uint64_t tail = size;
        for (; i < size; i++)
          tail = (tail << 8) | p[i];
        u64(tail);
      }
    };

    // the bytes a bound constant buffer holds right now, or false when
    // they cannot be read from the cpu (a default-usage buffer)
    bool CbBytes(const D3D11ConstantBufferBinding& b, const uint8_t** data, uint32_t* size) {
      D3D11Buffer* buffer = b.buffer.ptr();

      if (!buffer)
        return false;

      const uint8_t* base = reinterpret_cast<const uint8_t*>(buffer->GetMapPtr());

      if (!base)
        return false;

      uint32_t width  = buffer->Desc()->ByteWidth;
      uint32_t offset = b.constantOffset * 16u;

      if (offset >= width)
        return false;

      uint32_t bytes = width - offset;

      if (b.constantCount)
        bytes = std::min(bytes, b.constantCount * 16u);

      *data = base + offset;
      *size = std::min(bytes, MaxCbBytes);
      return true;
    }

    // hashes the current bytes of every cbuffer in \c mask except b12;
    // false when one of them cannot be read from the cpu
    bool HashCbvs(Hasher& k, const D3D11ShaderStageCbvBinding& cbv, uint32_t mask,
                  const D3D11CommonShader* shader, bool dropTranslation) {
      mask &= ~(1u << PerFrameSlot);

      for (uint32_t m = mask; m; m &= m - 1u) {
        uint32_t slot = bit::tzcnt(m);

        if (slot >= cbv.maxCount || !cbv.buffers[slot].buffer.ptr()) {
          k.u64(0xff00u | slot);
          continue;
        }

        const uint8_t* data = nullptr;
        uint32_t size = 0u;

        if (!CbBytes(cbv.buffers[slot], &data, &size))
          return false;

        // blessed: only the part the shader declares (cbN[size]). Skyrim
        // writes per-frame values past it: Utility.hlsl's TreeParams (b2
        // c7) advances every frame on every static caster, but only the
        // wind-animated variants declare that far.
        if (uint32_t declared = shader->BlessedCbvSize(slot))
          size = std::min(size, declared * 16u);

        k.u64(slot);

        const uint8_t* src = data;
        uint8_t copy[MaxCbBytes];

        if (dropTranslation && slot == GeometrySlot && size >= GeometryBytes) {
          std::memcpy(copy, data, size);

          for (uint32_t off : TranslationBytes)
            std::memset(copy + off, 0, 4);

          src = copy;
        }

        // blessed: only the registers the shader reads. A declared range
        // can hold registers it never reads: the shadow-map vs declares b2
        // up to World (c1-c4) and so covers c0 (ShadowFadeParam), which
        // only the shadow-mask permutations read (Utility.hlsl).
        if (shader->BlessedCbvWhole(slot)) {
          k.bytes(src, size);
        } else {
          uint64_t regs = shader->BlessedCbvRegs(slot);
          k.u64(regs);

          for (uint64_t r = regs; r; r &= r - 1u) {
            uint32_t reg = bit::tzcnt(r);

            if ((reg + 1u) * 16u > size)
              break;

            k.bytes(src + reg * 16u, 16u);
          }
        }
      }

      return true;
    }


    // ---- per layer ----

    enum RejectReason : uint32_t {
      RejectKind,        // not DrawIndexed
      RejectLayout,      // no layout, or a skinned one
      RejectStages,      // hs/ds/gs bound
      RejectStreams,     // a vertex stream other than per-vertex slot 0
      RejectBuffers,     // dynamic or staging vb/ib
      RejectVsBindings,  // no b2, bones, or vs textures
      RejectUav,
      RejectDepthState,
      RejectViewport,
      RejectSkipVs,
      RejectCbRead,      // a cbuffer the cpu cannot read
      RejectCount,
    };

    const char* RejectName(uint32_t r) {
      switch (r) {
        case RejectKind:       return "kind";
        case RejectLayout:     return "layout";
        case RejectStages:     return "stages";
        case RejectStreams:    return "streams";
        case RejectBuffers:    return "buffers";
        case RejectVsBindings: return "vs_bindings";
        case RejectUav:        return "uav";
        case RejectDepthState: return "depth_state";
        case RejectViewport:   return "viewport";
        case RejectSkipVs:     return "skip_vs";
        case RejectCbRead:     return "cb_read";
        default:               return "?";
      }
    }

    bool Reject(uint32_t* out, uint32_t reason) {
      *out = reason;
      return false;
    }


    struct ProjError {
      double texels = 0.0;  ///< largest xy move of a point of the cascade volume, in texels
      double depth  = 0.0;  ///< largest depth change of such a point, in d16 steps
      bool   ok     = false;
    };

    /// One cached draw of a key: where it stood, and the cascade that
    /// last matched it (State::serial)
    struct KeyInst {
      double   pos[3] = { };
      uint64_t serial = 0u;
    };

    /// The cached draws that share a key (same mesh, material, rotation)
    struct KeyEntry {
      std::vector<KeyInst> insts;
    };

    struct Target {
      ID3D11Texture2D*          tex     = nullptr;
      ID3D11DepthStencilView*   dsv     = nullptr;
      ID3D11ShaderResourceView* srv     = nullptr;
      Rc<DxvkImage>             image;
      Rc<DxvkImageView>         depthView;
      Rc<DxvkImageView>         sampledView;

      bool valid() const { return tex != nullptr; }

      void release() {
        depthView   = nullptr;
        sampledView = nullptr;
        image       = nullptr;
        if (srv) { srv->Release(); srv = nullptr; }
        if (dsv) { dsv->Release(); dsv = nullptr; }
        if (tex) { tex->Release(); tex = nullptr; }
      }
    };

    struct PendingVerify {
      Rc<DxvkBuffer> counters;
      uint64_t       frame      = 0u;
      uint32_t       slice      = 0u;
      uint32_t       missingKeys = 0u;
      bool           restoreCheck = false;  // layer vs cache right after a partial restore: must be equal
      double         errTexels   = 0.0;  // bound between the cache's projection and that frame's
      double         errDepth    = 0.0;
    };

    struct Slice {
      // the engine's view of this layer (compare only) and its image view
      D3D11DepthStencilView*  dsv = nullptr;
      Rc<DxvkImageView>       dsvView;
      VkExtent2D              extent = { };

      Target                  cache;
      Target                  scratch;   // verify only
      bool                    createFailed = false;

      // the cache
      bool                    valid = false;
      std::vector<uint8_t>    cacheProj;
      std::unordered_map<uint64_t, KeyEntry> keys;
      std::unordered_map<uint64_t, uint32_t> identities;
      uint32_t                keyDraws = 0u;
      bool                    rebuild = false;
      uint32_t                rebuildReason = ReasonFirst;

      // projection history
      std::vector<uint8_t>    lastProj;
      uint32_t                stableFrames = 0u;
      uint32_t                missingFrames = 0u;
      uint64_t                reuseFrames = 0u;

      // this frame's cascade
      Mode                    mode = Mode::None;
      float                   clearDepth = 1.0f;
      bool                    boundToAlt = false;  // the cache or scratch is bound instead of the layer
      uint64_t                altGeneration = 0u;  // blessedOmGeneration when it was bound
      std::vector<uint8_t>    proj;
      std::unordered_map<uint64_t, KeyEntry> building;
      std::unordered_map<uint64_t, uint32_t> buildingIds;
      uint32_t                buildDraws = 0u;
      uint32_t                seenDraws = 0u;
      std::vector<std::pair<uint64_t, uint64_t>> newIdentityHits;  // (identity, key) of cached meshes drawn with a new key

      // meshes seen moving: drawn as moving casters until the keys of all
      // their draws hold still for BLESSED_CASCADE_CACHE_HOT frames
      struct Hot {
        uint64_t sumPrev = 0u;  // order-free sum of the keys, previous cascade
        uint64_t sumCur  = 0u;  // the same, the cascade in \c serial
        uint64_t serial  = 0u;
        uint64_t until   = 0u;
        // blessed: the reheat log -- the first draw of the last two cascades
        std::vector<uint8_t> geomCur, geomPrev;
        double   posCur[3]  = { };
        double   posPrev[3] = { };
      };
      uint32_t                reheatLogged = 0u;
      std::unordered_map<uint64_t, Hot> hot;
      std::unordered_map<uint64_t, uint32_t> uncachedAge;
      std::unordered_map<uint64_t, uint32_t> uncachedNext;

      // stats window
      uint64_t                frames[6] = { };
      uint64_t                skipped = 0u;
      uint64_t                redrawnMoving = 0u;
      uint64_t                redrawnNew = 0u;
      uint64_t                reasons[ReasonCount] = { };
      uint64_t                missingKeys = 0u;
      uint64_t                rejects[RejectCount + 1u] = { };
      uint64_t                projChanges = 0u;       // this window
      uint64_t                restores = 0u;          // this window: reuse frames that restored the cache
      uint64_t                restoresSkipped = 0u;   // reuse frames with nothing cached: no restore
      uint64_t                hotReheats = 0u;        // hot meshes whose draws changed again

      // blessed: the partial restore. The layer keeps last frame's content
      // (its clear is dropped) when that content is exactly the cache plus
      // what last frame's other draws wrote, inside dirtyTiles.
      static constexpr uint32_t Tiles = 64u;              // per side
      std::array<uint64_t, Tiles> dirtyTiles = { };       // last cascade's other draws
      std::array<uint64_t, Tiles> dirtyNext  = { };       // this cascade's
      bool                    dirtyValid = false;         // layer == cache + dirtyTiles
      bool                    dirtyNextAll = false;       // a draw this cascade had no bound
      bool                    clearDeferred = false;      // the engine's clear was dropped, not run yet
      float                   clearDepthLast = -1.0f;     // last clear value (a changed one clears for real)
      Rc<DxvkImageView>       layerSampled;               // restore check
      uint64_t                partialRestores = 0u;       // this window
      uint64_t                partialTiles = 0u;
      uint64_t                partialRects = 0u;
      uint64_t                partialFallbackCover = 0u;
      uint64_t                partialFallbackRects = 0u;
      uint64_t                partialNotDeferred = 0u;    // reuse frames whose layer was cleared (no valid dirty set)
      uint64_t                partialChecks = 0u;
      uint64_t                boundsUnknown[BlessedCascadeBounds::ReasonCount] = { };
      double                  posAdjust[3] = { };     // this cascade's CameraPosAdjust
      int32_t                 buildBiasMin = INT32_MAX;
      int32_t                 cacheBias = 0;          // smallest DepthBias among the cached draws
      ProjError               lastErr;                // cache vs this cascade's projection
      double                  errTexelsMax = 0.0;     // this window, reuse frames
      double                  errDepthMax  = 0.0;
      // blessed: vs b2 bytes per cached mesh, for the moved-mesh log
      std::unordered_map<uint64_t, std::vector<uint8_t>> buildGeom;
      std::unordered_map<uint64_t, std::vector<uint8_t>> cacheGeom;
      uint32_t                movedLogged = 0u;
      uint64_t                draws = 0u;             // draws into this layer, any mode
      uint64_t                projChangesTotal = 0u;
      uint64_t                lastProjChangeFrame = 0u;
    };

    struct State {
      DxvkImage*              array = nullptr;    // compare only
      Rc<DxvkImage>           arrayRef;           // keeps the pointer from being reused
      VkExtent2D              arrayExtent = { };
      uint64_t                arrayLastClear = 0u;
      D3D11ShaderResourceView* lastSrv = nullptr;

      std::array<Slice, MaxSlices> slices;
      Slice*                  open = nullptr;
      uint32_t                openLayer = 0u;

      Rc<BlessedCascadeCachePass> pass;
      bool                    passFailed = false;

      std::deque<PendingVerify> verifies;
      uint64_t                frame = 1u;
      uint64_t                serial = 1u;       // one per opened cascade

      uint32_t                swapW = 0u;
      uint32_t                swapH = 0u;

      // log
      uint64_t                presentsInWindow = 0u;
      std::ofstream           log;
      bool                    logTried = false;
      dxvk::high_resolution_clock::time_point start;
      bool                    startSet = false;
      std::string             verifyLines;
      std::string             projLines;
      D3D11DepthStencilView*  strayMiss = nullptr;   // compare only
      std::ofstream           projLog;
    };

    // blessed: never destroyed. It holds Rc<> references to dxvk images,
    // views and buffers; releasing them from a static destructor at
    // process exit could run after the device is gone.
    State& g_state = *new State();


    void ResetSlices() {
      g_state.strayMiss = nullptr;

      for (auto& s : g_state.slices) {
        s.cache.release();
        s.scratch.release();
        s = Slice();
      }

      g_state.open = nullptr;
    }


    void Learn(DxvkImage* image) {
      const auto& info = image->info();

      if (image == g_state.array)
        return;

      if (g_state.array) {
        bool bigger = info.extent.width > g_state.arrayExtent.width;
        bool stale  = g_state.frame - g_state.arrayLastClear > 300u;

        if (!bigger && !stale)
          return;
      }

      ResetSlices();

      g_state.array       = image;
      g_state.arrayRef    = image;
      g_state.arrayExtent = { info.extent.width, info.extent.height };
      g_state.arrayLastClear = g_state.frame;

      Logger::info(str::format("BlessedCascadeCache: learned the sun cascade array, ",
        info.extent.width, "x", info.extent.height, " x", info.numLayers, " layers, format ", info.format));
    }


    void TryLearn(const D3D11ContextState& state) {
      const CascadeConfig& c = GetConfig();
      const auto& srvs = state.srv[D3D11ShaderType::ePixel];

      if (c.srvSlot >= srvs.maxCount)
        return;

      D3D11ShaderResourceView* srv = srvs.views[c.srvSlot].ptr();

      if (!srv || srv == g_state.lastSrv)
        return;

      g_state.lastSrv = srv;

      Rc<DxvkImageView> view = srv->GetImageView();

      if (view == nullptr)
        return;

      DxvkImage* image = view->image();
      const auto& info = image->info();

      bool depth = info.format == VK_FORMAT_D16_UNORM || info.format == VK_FORMAT_D32_SFLOAT;

      if (depth && info.numLayers >= 2u && info.numLayers <= MaxSlices
       && info.extent.width >= c.minSize && info.extent.width == info.extent.height
       && info.sampleCount == VK_SAMPLE_COUNT_1_BIT && info.mipLevels == 1u)
        Learn(image);
    }


    // ---- targets ----

    bool CreateTargetOnce(D3D11ImmediateContext* ctx, Slice& s, Target& t);

    bool CreateTarget(D3D11ImmediateContext* ctx, Slice& s, Target& t) {
      if (t.valid())
        return true;

      if (s.createFailed)
        return false;

      if (!CreateTargetOnce(ctx, s, t)) {
        s.createFailed = true;
        Logger::warn("BlessedCascadeCache: could not create a cache image, this layer stays vanilla");
        return false;
      }

      return true;
    }


    bool CreateTargetOnce(D3D11ImmediateContext* ctx, Slice& s, Target& t) {

      Com<ID3D11Resource> res;
      s.dsv->GetResource(&res);

      Com<ID3D11Texture2D> engineTex;
      if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&engineTex))))
        return false;

      D3D11_TEXTURE2D_DESC desc = { };
      engineTex->GetDesc(&desc);

      D3D11_DEPTH_STENCIL_VIEW_DESC engineDsv = { };
      s.dsv->GetDesc(&engineDsv);

      DXGI_FORMAT srvFormat;
      switch (engineDsv.Format) {
        case DXGI_FORMAT_D16_UNORM:   srvFormat = DXGI_FORMAT_R16_UNORM; break;
        case DXGI_FORMAT_D32_FLOAT:   srvFormat = DXGI_FORMAT_R32_FLOAT; break;
        default: return false;
      }

      desc.ArraySize      = 1u;
      desc.MipLevels      = 1u;
      desc.MiscFlags      = 0u;
      desc.CPUAccessFlags = 0u;
      desc.Usage          = D3D11_USAGE_DEFAULT;
      desc.BindFlags      = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;

      Com<ID3D11Device> device;
      ctx->GetDevice(&device);

      Com<ID3D11Texture2D> tex;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &tex)))
        return false;

      D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = { };
      dsvDesc.Format        = engineDsv.Format;
      dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

      Com<ID3D11DepthStencilView> dsv;
      if (FAILED(device->CreateDepthStencilView(tex.ptr(), &dsvDesc, &dsv)))
        return false;

      D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
      srvDesc.Format              = srvFormat;
      srvDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
      srvDesc.Texture2D.MipLevels = 1u;

      Com<ID3D11ShaderResourceView> srv;
      if (FAILED(device->CreateShaderResourceView(tex.ptr(), &srvDesc, &srv)))
        return false;

      t.tex = tex.ref();
      t.dsv = dsv.ref();
      t.srv = srv.ref();
      t.image       = GetCommonTexture(t.tex)->GetImage();
      t.depthView   = static_cast<D3D11DepthStencilView*>(t.dsv)->GetImageView();
      t.sampledView = static_cast<D3D11ShaderResourceView*>(t.srv)->GetImageView();
      return true;
    }


    // ---- projection ----

    bool ReadProj(const D3D11ContextState& state, std::vector<uint8_t>& out) {
      const auto& cbv = state.cbv[D3D11ShaderType::eVertex];

      if (PerFrameSlot >= cbv.maxCount)
        return false;

      const uint8_t* data = nullptr;
      uint32_t size = 0u;

      if (!CbBytes(cbv.buffers[PerFrameSlot], &data, &size))
        return false;

      if (GetConfig().projFull) {
        size = std::min(size, MaxProjBytes);
        out.assign(data, data + size);
        return true;
      }

      if (size < PosAdjustOffset + PosAdjustBytes)
        return false;

      out.resize(ViewProjBytes + PosAdjustBytes);
      std::memcpy(out.data(), data + ViewProjOffset, ViewProjBytes);
      std::memcpy(out.data() + ViewProjBytes, data + PosAdjustOffset, PosAdjustBytes);
      return true;
    }


    // ---- projection math ----

    /// The compared projection as numbers: world-relative view-proj
    /// (row-major, clip = vp * (world - posAdjust)) and posAdjust
    struct ProjView {
      double vp[16] = { };
      double pos[3] = { };
      bool   ok     = false;
    };

    ProjView ParseProj(const std::vector<uint8_t>& p) {
      ProjView v;
      size_t vpOffset  = GetConfig().projFull ? ViewProjOffset : 0u;
      size_t posOffset = GetConfig().projFull ? PosAdjustOffset : ViewProjBytes;

      if (p.size() < posOffset + 12u || p.size() < vpOffset + 64u)
        return v;

      for (uint32_t i = 0; i < 16; i++) {
        float f;
        std::memcpy(&f, p.data() + vpOffset + 4 * i, 4);
        v.vp[i] = f;
      }

      for (uint32_t i = 0; i < 3; i++) {
        float f;
        std::memcpy(&f, p.data() + posOffset + 4 * i, 4);
        v.pos[i] = f;
      }

      v.ok = true;
      return v;
    }

    bool Invert4(const double* m, double* out) {
      double a[4][8];

      for (uint32_t r = 0; r < 4; r++) {
        for (uint32_t c = 0; c < 4; c++) {
          a[r][c]     = m[4 * r + c];
          a[r][c + 4] = r == c ? 1.0 : 0.0;
        }
      }

      for (uint32_t c = 0; c < 4; c++) {
        uint32_t best = c;

        for (uint32_t r = c + 1; r < 4; r++) {
          if (std::abs(a[r][c]) > std::abs(a[best][c]))
            best = r;
        }

        if (std::abs(a[best][c]) < 1e-300)
          return false;

        if (best != c) {
          for (uint32_t k = 0; k < 8; k++)
            std::swap(a[c][k], a[best][k]);
        }

        double inv = 1.0 / a[c][c];

        for (uint32_t k = 0; k < 8; k++)
          a[c][k] *= inv;

        for (uint32_t r = 0; r < 4; r++) {
          if (r == c)
            continue;

          double f = a[r][c];

          for (uint32_t k = 0; k < 8; k++)
            a[r][k] -= f * a[c][k];
        }
      }

      for (uint32_t r = 0; r < 4; r++) {
        for (uint32_t c = 0; c < 4; c++)
          out[4 * r + c] = a[r][c + 4];
      }

      return true;
    }


    /**
     * rief How far any point of cascade volume \c a lands under \c b
     *
     * Takes the eight corners of \c a's clip box (x, y in -1..1, z in
     * 0..1), puts them back in world space with \c a, projects them
     * with \c b and returns the largest move. Both maps are affine
     * (orthographic), so the largest move over the box sits at a corner:
     * this bounds every texel of a depth map drawn with \c a and read as
     * if drawn with \c b.
     */
    ProjError ProjDistance(const ProjView& a, const ProjView& b, VkExtent2D extent) {
      ProjError e;
      double inv[16];

      if (!a.ok || !b.ok || !Invert4(a.vp, inv))
        return e;

      for (uint32_t i = 0; i < 8; i++) {
        double c[4] = { (i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0, (i & 4) ? 1.0 : 0.0, 1.0 };
        double w[4] = { };

        for (uint32_t r = 0; r < 4; r++)
          w[r] = inv[4 * r + 0] * c[0] + inv[4 * r + 1] * c[1] + inv[4 * r + 2] * c[2] + inv[4 * r + 3] * c[3];

        if (std::abs(w[3]) < 1e-300)
          return e;

        // back to world space, then relative to b's posAdjust
        double rel[4] = {
          w[0] / w[3] + a.pos[0] - b.pos[0],
          w[1] / w[3] + a.pos[1] - b.pos[1],
          w[2] / w[3] + a.pos[2] - b.pos[2],
          1.0 };

        double o[4];

        for (uint32_t r = 0; r < 4; r++)
          o[r] = b.vp[4 * r + 0] * rel[0] + b.vp[4 * r + 1] * rel[1] + b.vp[4 * r + 2] * rel[2] + b.vp[4 * r + 3] * rel[3];

        if (std::abs(o[3]) < 1e-300)
          return e;

        double dx = std::abs(o[0] / o[3] - c[0]) * 0.5 * double(extent.width);
        double dy = std::abs(o[1] / o[3] - c[1]) * 0.5 * double(extent.height);
        double dz = std::abs(o[2] / o[3] - c[2]) * 65535.0;

        e.texels = std::max(e.texels, std::max(dx, dy));
        e.depth  = std::max(e.depth, dz);
      }

      e.ok = true;
      return e;
    }


    // a float as json: non-finite values (stale cbuffer bytes can be
    // anything) as null
    std::string JsonNum(float v) {
      return std::isfinite(v) ? str::format(v) : std::string("null");
    }


    std::string Hex64(uint64_t v) {
      char buf[24];
      std::snprintf(buf, sizeof(buf), "\"0x%llx\"", static_cast<unsigned long long>(v));
      return buf;
    }


    // blessed: cascade_cache_proj.jsonl -- each static-caster vs once:
    // the declared size of every cbuffer it reads
    void NoteVs(D3D11VertexShader* vs) {
      static std::unordered_map<const void*, bool> s_seen;

      if (s_seen.size() >= 64u || !s_seen.emplace(vs, true).second)
        return;

      const D3D11CommonShader* cs = vs->GetCommonShader();
      uint32_t mask = cs->GetBindingMask().cbvMask;
      std::string sizes;

      for (uint32_t m = mask; m; m &= m - 1u) {
        uint32_t slot = bit::tzcnt(m);
        sizes += str::format(sizes.empty() ? "" : ",", "[", slot, ",", cs->BlessedCbvSize(slot), ",",
          cs->BlessedCbvWhole(slot) ? std::string("\"whole\"") : Hex64(cs->BlessedCbvRegs(slot)), "]");
      }

      g_state.projLines += str::format("{\"vs\":\"", cs->GetName(), "\",\"cb_vec4s\":[", sizes, "]}\n");
    }


    // ---- classification ----

    bool SkipVs(D3D11VertexShader* vs) {
      const CascadeConfig& c = GetConfig();

      if (c.skipVs.empty())
        return false;

      static std::unordered_map<const void*, bool> s_cache;
      auto it = s_cache.find(vs);

      if (it != s_cache.end())
        return it->second;

      std::string name = vs->GetCommonShader()->GetName();
      bool skip = false;

      for (const auto& tok : c.skipVs)
        skip |= name.find(tok) != std::string::npos;

      s_cache.emplace(vs, skip);
      return skip;
    }


    bool IsStaticBuffer(D3D11Buffer* buffer) {
      return buffer && buffer->Desc()->Usage != D3D11_USAGE_DYNAMIC
                    && buffer->Desc()->Usage != D3D11_USAGE_STAGING;
    }




    /**
     * \brief The key of a cacheable static caster, or false
     *
     * \param [out] key Everything that decides what this draw writes
     * \param [out] identity The mesh alone: pipeline, buffers, range
     */
    bool StaticKey(
      const D3D11ContextState&      state,
      const Slice&                  s,
            BlessedCascadeDrawKind  kind,
            uint32_t                indexCount,
            uint32_t                startIndex,
            int32_t                 baseVertex,
            StaticDraw*             out,
            uint32_t*               reject) {
      if (kind != BlessedCascadeDrawKind::Indexed)
        return Reject(reject, RejectKind);

      D3D11VertexShader* vs = state.vs.ptr();
      D3D11InputLayout* layout = state.ia.inputLayout.ptr();

      if (!vs || !layout || layout->HasBlessedSkinning())
        return Reject(reject, RejectLayout);

      if (state.hs.ptr() || state.ds.ptr() || state.gs.ptr())
        return Reject(reject, RejectStages);

      // every vertex stream must be slot 0, per vertex
      for (uint32_t i = 0; i < layout->GetBindingCount(); i++) {
        DxvkVertexBinding b = layout->GetInput(layout->GetAttributeCount() + i).binding();

        if (b.binding != 0u || b.inputRate != VK_VERTEX_INPUT_RATE_VERTEX)
          return Reject(reject, RejectStreams);
      }

      const auto& vb = state.ia.vertexBuffers[0];
      const auto& ib = state.ia.indexBuffer;

      if (!IsStaticBuffer(vb.buffer.ptr()) || !IsStaticBuffer(ib.buffer.ptr()))
        return Reject(reject, RejectBuffers);

      // what the shaders actually read (dxvk's binding masks): a static
      // caster's vs reads its PerGeometry b2 and no bones (b9/b10), and
      // neither stage reads a uav or a vertex-stage texture
      const D3D11BindingMask vsMask = vs->GetCommonShader()->GetBindingMask();

      if (!(vsMask.cbvMask & (1u << GeometrySlot)) || (vsMask.cbvMask & BoneSlots)
       || vsMask.uavMask || vsMask.srvMask[0] || vsMask.srvMask[1])
        return Reject(reject, RejectVsBindings);

      D3D11PixelShader* ps = state.ps.ptr();
      D3D11BindingMask psMask = { };

      if (ps) {
        psMask = ps->GetCommonShader()->GetBindingMask();

        if (psMask.uavMask)
          return Reject(reject, RejectUav);
      }

      const auto& vsCbv = state.cbv[D3D11ShaderType::eVertex];

      if (GeometrySlot >= vsCbv.maxCount || !vsCbv.buffers[GeometrySlot].buffer.ptr())
        return Reject(reject, RejectVsBindings);

      if (!CbBytes(vsCbv.buffers[GeometrySlot], &out->geom, &out->geomSize))
        return Reject(reject, RejectCbRead);

      if (uint32_t declared = vs->GetCommonShader()->BlessedCbvSize(GeometrySlot))
        out->geomSize = std::min(out->geomSize, declared * 16u);

      NoteVs(vs);

      // depth writes, LESS or LESS_EQUAL, no stencil: min-composable
      D3D11DepthStencilState* dss = state.om.dsState.ptr();

      if (!dss)
        return Reject(reject, RejectDepthState);

      const auto& dsDesc = dss->Desc();

      if (!dsDesc.DepthEnable || dsDesc.DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ALL || dsDesc.StencilEnable
       || (dsDesc.DepthFunc != D3D11_COMPARISON_LESS && dsDesc.DepthFunc != D3D11_COMPARISON_LESS_EQUAL))
        return Reject(reject, RejectDepthState);

      // one full viewport over the layer
      if (state.rs.numViewports != 1u)
        return Reject(reject, RejectViewport);

      const D3D11_VIEWPORT& vp = state.rs.viewports[0];

      if (vp.TopLeftX != 0.0f || vp.TopLeftY != 0.0f
       || vp.Width != float(s.extent.width) || vp.Height != float(s.extent.height)
       || vp.MinDepth != 0.0f || vp.MaxDepth != 1.0f)
        return Reject(reject, RejectViewport);

      // blessed: graphics uavs only. state.uav is the compute stage's, and
      // its maxCount never shrinks once a compute pass bound one (see
      // CSSetUnorderedAccessViews), so testing it rejected every draw.
      for (uint32_t i = state.om.minUav; i < state.om.maxUav; i++) {
        if (state.om.uavs[i].ptr())
          return Reject(reject, RejectUav);
      }

      if (SkipVs(vs))
        return Reject(reject, RejectSkipVs);

      D3D11RasterizerState* rs = state.rs.state.ptr();

      Hasher id;
      id.ptr(vs);
      id.ptr(layout);
      id.ptr(ib.buffer.ptr());
      id.u64(ib.offset);
      id.u64(uint64_t(ib.format));
      id.ptr(vb.buffer.ptr());
      id.u64(vb.offset);
      id.u64(vb.stride);
      id.u64(indexCount);
      id.u64(startIndex);
      id.u64(uint64_t(uint32_t(baseVertex)));
      id.u64(uint64_t(state.ia.primitiveTopology));

      Hasher k = id;
      k.ptr(ps);
      k.ptr(rs);
      k.ptr(dss);

      if (rs && rs->Desc().ScissorEnable) {
        if (state.rs.numScissors)
          k.bytes(&state.rs.scissors[0], sizeof(D3D11_RECT));
        else
          k.u64(0u);
      }

      // blessed: World's translation (b2 c1.w, c2.w, c3.w) is relative to
      // CameraPosAdjust, which moves with every cascade camera step: the
      // key leaves it out, and the draw carries its world position
      // (translation + posAdjust) for a tolerance match instead
      bool movable = out->geomSize >= GeometryBytes;

      if (!HashCbvs(k, state.cbv[D3D11ShaderType::eVertex], vsMask.cbvMask, vs->GetCommonShader(), movable))
        return Reject(reject, RejectCbRead);

      if (movable) {
        for (uint32_t i = 0; i < 3; i++) {
          float t;
          std::memcpy(&t, out->geom + TranslationBytes[i], 4);
          out->pos[i] = double(t) + s.posAdjust[i];
        }
      }

      if (ps) {
        if (!HashCbvs(k, state.cbv[D3D11ShaderType::ePixel], psMask.cbvMask, ps->GetCommonShader(), false))
          return Reject(reject, RejectCbRead);

        const auto& psSrv = state.srv[D3D11ShaderType::ePixel];
        const auto& psSmp = state.samplers[D3D11ShaderType::ePixel];

        for (uint32_t w = 0; w < 2u; w++) {
          for (uint64_t m = psMask.srvMask[w]; m; m &= m - 1u) {
            uint32_t slot = 64u * w + uint32_t(bit::tzcnt(m));
            k.u64(slot);
            k.ptr(slot < psSrv.maxCount ? psSrv.views[slot].ptr() : nullptr);
          }
        }

        for (uint32_t m = psMask.samplerMask; m; m &= m - 1u) {
          uint32_t slot = bit::tzcnt(m);
          k.u64(slot);
          k.ptr(slot < psSmp.maxCount ? psSmp.samplers[slot].ptr() : nullptr);
        }
      }

      out->key      = k.h;
      out->identity = id.h;

      // key + world position to a quarter unit: what the hot and promote
      // bookkeeping compare across frames
      Hasher inst = k;
      for (uint32_t i = 0; i < 3; i++)
        inst.u64(uint64_t(std::llround(out->pos[i] * 4.0)));
      out->instance = inst.h;
      return true;
    }


    // ---- binding ----

    void BindAlt(D3D11ImmediateContext* ctx, Slice& s, const Target& t, const D3D11ContextState& state) {
      if (s.boundToAlt)
        return;

      s.altGeneration = state.om.blessedOmGeneration;

      BlessedCascadeCacheCtx::Emit(ctx, [cView = t.depthView] (DxvkContext* dxvkCtx) {
        DxvkRenderTargets rt;
        rt.depth.view = cView;
        dxvkCtx->bindRenderTargets(std::move(rt), 0u);
      });

      s.boundToAlt = true;
    }


    void BindLayer(D3D11ImmediateContext* ctx, Slice& s) {
      if (!s.boundToAlt)
        return;

      // the context's own targets (still the layer, or whatever the game
      // bound since) go back
      BlessedCascadeCacheCtx::Rebind(ctx);
      s.boundToAlt = false;
    }


    void ClearTarget(D3D11ImmediateContext* ctx, const Target& t, float depth) {
      VkClearValue value = { };
      value.depthStencil.depth = depth;

      DxvkAttachment attachment = { };
      attachment.view = t.depthView;

      BlessedCascadeCacheCtx::Emit(ctx, [cAttachment = std::move(attachment), cValue = value] (DxvkContext* dxvkCtx) {
        dxvkCtx->clearRenderTarget(cAttachment, VK_IMAGE_ASPECT_DEPTH_BIT, cValue, 0u);
      });
    }


    bool EnsurePass(D3D11ImmediateContext* ctx) {
      if (g_state.pass != nullptr)
        return true;

      if (g_state.passFailed)
        return false;

      g_state.pass = new BlessedCascadeCachePass(BlessedCascadeCacheCtx::Device(ctx));
      return true;
    }


    // dst = min(dst, src) over the whole layer; the game's state is reset
    // before and restored after, as the video blit does
    void Composite(D3D11ImmediateContext* ctx, Slice& s, const Target& src,
                   VkCompareOp compareOp = VK_COMPARE_OP_LESS_OR_EQUAL) {
      BlessedCascadeCacheCtx::Reset(ctx);

      BlessedCascadeCacheCtx::Emit(ctx, [
        cPass   = g_state.pass,
        cDst    = s.dsvView,
        cSrc    = src.sampledView,
        cExtent = s.extent,
        cOp     = compareOp
      ] (DxvkContext* dxvkCtx) {
        cPass->composite(dxvkCtx, cDst, cSrc, cExtent, cOp);
      });

      BlessedCascadeCacheCtx::Restore(ctx);
    }


    void RunVerifyViews(D3D11ImmediateContext* ctx, Slice& s, uint32_t layer, uint32_t missingKeys,
                        const Rc<DxvkImageView>& a, const Rc<DxvkImageView>& b, bool restoreCheck);

    void RunVerify(D3D11ImmediateContext* ctx, Slice& s, uint32_t layer, uint32_t missingKeys) {
      RunVerifyViews(ctx, s, layer, missingKeys, s.cache.sampledView, s.scratch.sampledView, false);
    }

    void RunVerifyViews(D3D11ImmediateContext* ctx, Slice& s, uint32_t layer, uint32_t missingKeys,
                        const Rc<DxvkImageView>& a, const Rc<DxvkImageView>& b, bool restoreCheck) {
      Rc<DxvkDevice> dxvkDevice = BlessedCascadeCacheCtx::Device(ctx);

      DxvkBufferCreateInfo info = { };
      info.size      = sizeof(BlessedCascadeVerifyCounters);
      info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      info.access    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      info.debugName = "blessed cascade verify";

      Rc<DxvkBuffer> counters = dxvkDevice->createBuffer(info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      std::memset(counters->getSliceInfo().mapPtr, 0, sizeof(BlessedCascadeVerifyCounters));

      BlessedCascadeCacheCtx::Reset(ctx);

      BlessedCascadeCacheCtx::Emit(ctx, [
        cPass     = g_state.pass,
        cCached   = a,
        cFresh    = b,
        cCounters = counters,
        cExtent   = s.extent
      ] (DxvkContext* dxvkCtx) {
        cPass->verify(dxvkCtx, cCached, cFresh, cCounters, cExtent);
      });

      BlessedCascadeCacheCtx::Restore(ctx);

      PendingVerify pv;
      pv.counters    = std::move(counters);
      pv.frame       = g_state.frame;
      pv.slice       = layer;
      pv.missingKeys = missingKeys;
      pv.restoreCheck = restoreCheck;
      pv.errTexels   = s.lastErr.texels;
      pv.errDepth    = s.lastErr.depth;
      g_state.verifies.push_back(std::move(pv));
    }


    // blessed: cascade_cache_proj.jsonl -- which of the compared floats
    // changed, old and new, for the first 64 changes of a layer and every
    // 16th after. Float i is byte 4i of what ReadProj compares: 0-15
    // CameraViewProj (row-major, b12 c8-c11), 16-19 CameraPosAdjust (c40);
    // with BLESSED_CASCADE_CACHE_PROJ=full, byte 4i of b12.
    void NoteProjChange(Slice& s, uint32_t layer) {
      uint64_t n = ++s.projChangesTotal;
      uint64_t since = g_state.frame - s.lastProjChangeFrame;
      s.lastProjChangeFrame = g_state.frame;
      s.projChanges++;

      if (n > 64u && (n % 16u))
        return;

      size_t count = std::min(s.proj.size(), s.lastProj.size()) / sizeof(float);
      std::string diff;
      uint32_t changed = 0u;

      for (size_t i = 0; i < count; i++) {
        float a, b;
        std::memcpy(&a, s.lastProj.data() + 4 * i, 4);
        std::memcpy(&b, s.proj.data() + 4 * i, 4);

        if (std::memcmp(&a, &b, 4) == 0)
          continue;

        if (changed++ < 24u) {
          diff += str::format(diff.empty() ? "" : ",", "[", i, ",", JsonNum(a), ",", JsonNum(b), "]");
        }
      }

      g_state.projLines += str::format("{\"frame\":", g_state.frame, ",\"layer\":", layer,
        ",\"change\":", n, ",\"frames_since_last\":", since,
        ",\"changed_floats\":", changed, ",\"diff\":[", diff, "]}\n");
    }


    // blessed: cascade_cache_proj.jsonl -- a cached mesh drawn with a
    // new key or position: which vs b2 floats differ from the cached
    // draw's, and how far posAdjust moved since the build. First 16 per
    // layer, then every 64th.
    void NoteMoved(Slice& s, uint32_t layer, const StaticDraw& d) {
      uint32_t n = ++s.movedLogged;

      if (n > 16u && (n % 64u))
        return;

      auto g = s.cacheGeom.find(d.identity);

      if (g == s.cacheGeom.end() || !d.geom)
        return;

      ProjView built = ParseProj(s.cacheProj);
      size_t count = std::min<size_t>(g->second.size(), std::min(d.geomSize, 128u)) / 4u;
      std::string diff;
      uint32_t changed = 0u;

      for (size_t i = 0; i < count; i++) {
        float a, b;
        std::memcpy(&a, g->second.data() + 4 * i, 4);
        std::memcpy(&b, d.geom + 4 * i, 4);

        if (std::memcmp(&a, &b, 4) != 0 && changed++ < 24u)
          diff += str::format(diff.empty() ? "" : ",", "[", i, ",", JsonNum(a), ",", JsonNum(b), "]");
      }

      g_state.projLines += str::format("{\"moved\":", n, ",\"frame\":", g_state.frame, ",\"layer\":", layer,
        ",\"pos_adjust_delta\":[", s.posAdjust[0] - built.pos[0], ",", s.posAdjust[1] - built.pos[1], ",",
        s.posAdjust[2] - built.pos[2], "],\"world_pos\":[", d.pos[0], ",", d.pos[1], ",", d.pos[2],
        "],\"changed_floats\":", changed, ",\"diff\":[", diff, "]}\n");
    }


    // ---- the cascade's first draw ----
    // blessed: the engine's clear of the layer, run late (it was dropped at
    // ClearDepthStencilView for a partial restore that did not happen)
    void EmitLayerClear(D3D11ImmediateContext* ctx, Slice& s) {
      if (!s.clearDeferred)
        return;

      s.clearDeferred = false;

      VkClearValue value = { };
      value.depthStencil.depth = s.clearDepth;

      DxvkAttachment attachment = { };
      attachment.view = s.dsvView;

      BlessedCascadeCacheCtx::XferEvent(ctx);
      BlessedCascadeCacheCtx::Emit(ctx, [cAttachment = std::move(attachment), cValue = value] (DxvkContext* dxvkCtx) {
        dxvkCtx->clearRenderTarget(cAttachment, VK_IMAGE_ASPECT_DEPTH_BIT, cValue, 0u);
      });
    }


    void MarkTiles(Slice& s, const BlessedCascadeBounds::Rect& r) {
      uint32_t tw = std::max(1u, s.extent.width / Slice::Tiles);
      uint32_t th = std::max(1u, s.extent.height / Slice::Tiles);
      uint32_t x0 = std::min(uint32_t(r.x0) / tw, Slice::Tiles - 1u);
      uint32_t x1 = std::min((uint32_t(r.x1) + tw - 1u) / tw, Slice::Tiles);
      uint32_t y0 = std::min(uint32_t(r.y0) / th, Slice::Tiles - 1u);
      uint32_t y1 = std::min((uint32_t(r.y1) + th - 1u) / th, Slice::Tiles);

      if (x1 <= x0 || y1 <= y0)
        return;

      uint64_t bits = (x1 - x0 >= 64u) ? ~0ull : (((1ull << (x1 - x0)) - 1ull) << x0);

      for (uint32_t y = y0; y < y1; y++)
        s.dirtyNext[y] |= bits;
    }


    // a draw that writes the layer (not the cache): where it lands
    void TrackLayerDraw(Slice& s, const D3D11ContextState& state, BlessedCascadeDrawKind kind,
                        uint32_t indexCount, uint32_t startIndex, int32_t baseVertex) {
      if (s.dirtyNextAll)
        return;

      if (kind != BlessedCascadeDrawKind::Indexed) {
        s.dirtyNextAll = true;
        s.boundsUnknown[BlessedCascadeBounds::ReasonNoPosition]++;
        return;
      }

      ProjView v = ParseProj(s.proj);

      if (!v.ok) {
        s.dirtyNextAll = true;
        return;
      }

      BlessedCascadeBounds::Rect rect = { };
      uint32_t reason = 0u;
      auto r = BlessedCascadeBounds::DrawRect(state, indexCount, startIndex, baseVertex, g_state.frame,
        v.vp, v.pos, s.extent.width, s.extent.height, &rect, &reason);

      if (r == BlessedCascadeBounds::Result::Rect) {
        MarkTiles(s, rect);
      } else if (r == BlessedCascadeBounds::Result::Unknown) {
        s.dirtyNextAll = true;
        s.boundsUnknown[std::min<uint32_t>(reason, BlessedCascadeBounds::ReasonCount - 1u)]++;
      }
    }


    struct TileRect { uint32_t x0, y0, x1, y1; };  // tiles, half-open

    // dirty tiles as rectangles: runs per tile row, merged down while the
    // run below spans the same columns. false: too many or too much
    bool PlanRects(const Slice& s, std::vector<TileRect>& out, uint32_t* tiles) {
      const CascadeConfig& c = GetConfig();
      out.clear();
      *tiles = 0u;

      std::vector<TileRect> open;

      for (uint32_t y = 0; y <= Slice::Tiles; y++) {
        uint64_t row = y < Slice::Tiles ? s.dirtyTiles[y] : 0ull;
        std::vector<TileRect> runs;

        for (uint32_t x = 0; x < Slice::Tiles; ) {
          if (!(row & (1ull << x))) { x++; continue; }
          uint32_t x0 = x;
          while (x < Slice::Tiles && (row & (1ull << x))) x++;
          runs.push_back({ x0, y, x, y + 1u });
          *tiles += x - x0;
        }

        std::vector<TileRect> next;

        for (auto& r : runs) {
          auto it = std::find_if(open.begin(), open.end(),
            [&] (const TileRect& o) { return o.x0 == r.x0 && o.x1 == r.x1; });

          if (it != open.end()) {
            TileRect grown = *it;
            grown.y1 = y + 1u;
            next.push_back(grown);
            open.erase(it);
          } else {
            next.push_back(r);
          }
        }

        for (auto& o : open)
          out.push_back(o);

        open = std::move(next);
      }

      if (double(*tiles) > double(c.partialMaxCover) * double(Slice::Tiles * Slice::Tiles))
        return false;

      return out.size() <= c.partialMaxRects;
    }


    Rc<DxvkImageView> LayerSampledView(Slice& s, uint32_t layer) {
      if (s.layerSampled == nullptr) {
        DxvkImageViewKey key = { };
        key.viewType   = VK_IMAGE_VIEW_TYPE_2D;
        key.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
        key.format     = g_state.arrayRef->info().format;
        key.aspects    = VK_IMAGE_ASPECT_DEPTH_BIT;
        key.mipIndex   = 0u;
        key.mipCount   = 1u;
        key.layerIndex = uint16_t(layer);
        key.layerCount = 1u;
        s.layerSampled = g_state.arrayRef->createView(key);
      }

      return s.layerSampled;
    }


    /**
     * \brief The partial restore: copy back the dirty tiles only
     *
     * \returns false: not possible this frame, the caller restores in full
     */
    bool PartialRestore(D3D11ImmediateContext* ctx, Slice& s, uint32_t layer) {
      const CascadeConfig& c = GetConfig();

      if (!s.clearDeferred || !s.dirtyValid) {
        s.partialNotDeferred++;
        return false;
      }

      std::vector<TileRect> rects;
      uint32_t tiles = 0u;

      if (!PlanRects(s, rects, &tiles)) {
        if (double(tiles) > double(c.partialMaxCover) * double(Slice::Tiles * Slice::Tiles))
          s.partialFallbackCover++;
        else
          s.partialFallbackRects++;
        return false;
      }

      uint32_t tw = s.extent.width / Slice::Tiles;
      uint32_t th = s.extent.height / Slice::Tiles;

      std::vector<VkRect2D> regions;
      regions.reserve(rects.size());

      for (const auto& r : rects) {
        VkRect2D reg;
        reg.offset = { int32_t(r.x0 * tw), int32_t(r.y0 * th) };
        reg.extent = { std::min((r.x1 - r.x0) * tw, s.extent.width  - r.x0 * tw),
                       std::min((r.y1 - r.y0) * th, s.extent.height - r.y0 * th) };
        regions.push_back(reg);
      }

      // blessed: the layer is not cleared this frame; it holds the cache
      // plus last frame's other casters, and those sit inside these tiles
      s.clearDeferred = false;

      if (!regions.empty()) {
        BlessedCascadeCacheCtx::XferEvent(ctx);
        BlessedCascadeCacheCtx::Emit(ctx, [
          cDst     = g_state.arrayRef,
          cLayer   = layer,
          cSrc     = s.cache.image,
          cRegions = std::move(regions)
        ] (DxvkContext* dxvkCtx) {
          for (const auto& r : cRegions) {
            VkOffset3D offset = { r.offset.x, r.offset.y, 0 };
            dxvkCtx->copyImage(
              cDst, { VK_IMAGE_ASPECT_DEPTH_BIT, 0u, cLayer, 1u }, offset,
              cSrc, { VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 0u, 1u }, offset,
              VkExtent3D { r.extent.width, r.extent.height, 1u });
          }
        });
      }

      s.partialRestores++;
      s.partialTiles += tiles;
      s.partialRects += rects.size();

      // blessed: the proof. every VERIFY-th partial restore compares the
      // whole layer with the cache right now: they must be equal texel for
      // texel, whatever the projection tolerance
      if (c.verifyEvery && (++s.partialChecks % c.verifyEvery) == 0u && EnsurePass(ctx))
        RunVerifyViews(ctx, s, layer, 0u, s.cache.sampledView, LayerSampledView(s, layer), true);

      return true;
    }


    // blessed: cascade_cache_proj.jsonl -- a hot mesh whose draws changed
    // between the last two cascades: the vs b2 floats of its first draw
    // in each (declared range), its world position, and the vs
    void NoteReheat(Slice& s, uint32_t layer, const StaticDraw& d, const Slice::Hot& e) {
      s.reheatLogged++;

      size_t count = std::min(e.geomPrev.size(), e.geomCur.size()) / 4u;
      std::string diff;
      uint32_t changed = 0u;

      for (size_t i = 0; i < count; i++) {
        float a, b;
        std::memcpy(&a, e.geomPrev.data() + 4 * i, 4);
        std::memcpy(&b, e.geomCur.data() + 4 * i, 4);

        if (std::memcmp(&a, &b, 4) != 0 && changed++ < 24u)
          diff += str::format(diff.empty() ? "" : ",", "[", i, ",", JsonNum(a), ",", JsonNum(b), "]");
      }

      g_state.projLines += str::format("{\"reheat\":", s.reheatLogged, ",\"frame\":", g_state.frame,
        ",\"layer\":", layer, ",\"identity\":", Hex64(d.identity),
        ",\"pos_prev\":[", e.posPrev[0], ",", e.posPrev[1], ",", e.posPrev[2],
        "],\"pos_cur\":[", e.posCur[0], ",", e.posCur[1], ",", e.posCur[2],
        "],\"b2_bytes\":", e.geomCur.size(), ",\"changed_floats\":", changed, ",\"diff\":[", diff, "]}\n");
    }


    bool Tolerant() {
      return GetConfig().tolTexels > 0.0f;
    }

    /// The depth tolerance in d16 steps: the env value, or half the
    /// smallest constant depth bias the cached casters were drawn with
    /// (DepthBias is in d16 steps for a d16 target), at least one step
    double DepthTolerance(const Slice& s) {
      float t = GetConfig().tolDepth;

      if (t >= 0.0f)
        return t;

      return std::max(1.0, 0.5 * double(s.cacheBias));
    }

    bool WithinTolerance(const Slice& s, const ProjError& e) {
      return e.ok && e.texels <= double(GetConfig().tolTexels) && e.depth <= DepthTolerance(s);
    }


    // ---- the cascade's first draw ----

    void DecideMode(D3D11ImmediateContext* ctx, Slice& s, uint32_t layer, const D3D11ContextState& state);

    void Decide(D3D11ImmediateContext* ctx, Slice& s, uint32_t layer, const D3D11ContextState& state) {
      DecideMode(ctx, s, layer, state);

      // whatever did not restore partially gets the engine's clear back
      EmitLayerClear(ctx, s);
    }

    void DecideMode(D3D11ImmediateContext* ctx, Slice& s, uint32_t layer, const D3D11ContextState& state) {
      const CascadeConfig& c = GetConfig();

      s.mode = Mode::Pass;
      s.seenDraws = 0u;
      s.buildDraws = 0u;
      s.newIdentityHits.clear();
      s.boundToAlt = false;
      s.uncachedNext.clear();

      if (!ReadProj(state, s.proj)) {
        s.rejects[RejectCount]++;
        s.lastProj.clear();
        s.stableFrames = 0u;
        return;
      }

      ProjView now = ParseProj(s.proj);

      for (uint32_t i = 0; i < 3; i++)
        s.posAdjust[i] = now.pos[i];

      bool identical = !s.lastProj.empty() && s.lastProj == s.proj;

      if (!identical && !s.lastProj.empty())
        NoteProjChange(s, layer);

      // exact mode: stable means bitwise equal. with a tolerance, a
      // projection that drifted less than it from last frame's counts too
      bool same = identical;

      if (!same && Tolerant() && !s.lastProj.empty())
        same = WithinTolerance(s, ProjDistance(ParseProj(s.lastProj), now, s.extent));

      s.stableFrames = same ? s.stableFrames + 1u : 1u;
      s.lastProj = s.proj;

      if (c.debugMode == Debug::Pass || !(c.sliceMask & (1u << layer)))
        return;

      if (s.valid && s.cacheProj != s.proj) {
        // blessed: the cache was drawn with cacheProj. Exact mode drops it
        // on any change. With a tolerance it stays while no point of the
        // cascade volume moved more than BLESSED_CASCADE_CACHE_TOL_TEXELS
        // texels sideways or the depth tolerance in depth between the two
        // projections (see ProjDistance).
        ProjError err = Tolerant()
          ? ProjDistance(ParseProj(s.cacheProj), now, s.extent)
          : ProjError();
        s.lastErr = err;

        if (!Tolerant() || !WithinTolerance(s, err)) {
          s.valid = false;
          s.rebuild = true;
          s.rebuildReason = ReasonProj;
        } else {
          s.errTexelsMax = std::max(s.errTexelsMax, err.texels);
          s.errDepthMax  = std::max(s.errDepthMax, err.depth);
        }
      } else {
        s.lastErr = ProjError();
        s.lastErr.ok = true;
      }

      if (c.debugMode == Debug::Build && s.valid) {
        s.rebuild = true;
        s.rebuildReason = ReasonDebug;
      }

      if (s.valid && !s.rebuild) {
        s.mode = Mode::Reuse;

        if (c.verifyEvery && (++s.reuseFrames % c.verifyEvery) == 0u
         && EnsurePass(ctx) && CreateTarget(ctx, s, s.scratch))
          s.mode = Mode::Verify;

        if (s.mode == Mode::Verify) {
          ClearTarget(ctx, s.scratch, s.clearDepth);
        } else if (c.debugMode == Debug::Off && c.restorePartial && s.keyDraws
                && PartialRestore(ctx, s, layer)) {
          // blessed: BLESSED_CASCADE_CACHE_RESTORE=partial, done above
        } else if (c.debugMode != Debug::Redrawn && !s.keyDraws) {
          // blessed: nothing cached (every static caster hot or new): the
          // engine's clear already left the layer as it should start
          s.restoresSkipped++;
        } else if (c.debugMode != Debug::Redrawn && c.restoreDraw) {
          // blessed: BLESSED_CASCADE_CACHE_RESTORE=draw -- the restore as a
          // fullscreen depth write, the layer stays a depth attachment
          s.restores++;
          Composite(ctx, s, s.cache, VK_COMPARE_OP_ALWAYS);
        } else if (c.debugMode != Debug::Redrawn) {
          // blessed: the restore. The engine's clear already ran (or was
          // dropped): this copy overwrites the whole layer with the cached
          // statics.
          s.restores++;
          s.clearDeferred = false;
          BlessedCascadeCacheCtx::XferEvent(ctx);
          BlessedCascadeCacheCtx::Emit(ctx, [
            cDst    = g_state.arrayRef,
            cLayer  = layer,
            cSrc    = s.cache.image,
            cExtent = s.extent
          ] (DxvkContext* dxvkCtx) {
            dxvkCtx->copyImage(
              cDst, { VK_IMAGE_ASPECT_DEPTH_BIT, 0u, cLayer, 1u }, VkOffset3D { 0, 0, 0 },
              cSrc, { VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 0u, 1u }, VkOffset3D { 0, 0, 0 },
              VkExtent3D { cExtent.width, cExtent.height, 1u });
          });
        }

        return;
      }

      bool stableEnough = s.stableFrames >= c.stable || c.debugMode == Debug::Build;

      if (!stableEnough || !EnsurePass(ctx) || !CreateTarget(ctx, s, s.cache))
        return;

      if (!s.valid && !s.rebuild)
        s.rebuildReason = ReasonFirst;

      s.reasons[s.rebuildReason]++;
      s.mode = Mode::Build;
      s.valid = false;
      s.building.clear();
      s.buildingIds.clear();
      s.buildGeom.clear();
      s.buildBiasMin = INT32_MAX;
      ClearTarget(ctx, s.cache, s.clearDepth);
    }


    // ---- the cascade's end ----

    void CloseOpen(D3D11ImmediateContext* ctx) {
      Slice* sp = g_state.open;

      if (!sp)
        return;

      g_state.open = nullptr;

      Slice& s = *sp;
      uint32_t layer = g_state.openLayer;
      const CascadeConfig& c = GetConfig();

      Mode mode = s.mode;
      s.mode = Mode::None;

      // blessed: partial restore bookkeeping. After a build or a reuse the
      // layer is the cache plus this cascade's other draws, all bounded;
      // anything else leaves it unknown and the next frame clears it.
      if (c.restorePartial) {
        EmitLayerClear(ctx, s);

        bool known = (mode == Mode::Build || mode == Mode::Reuse) && !s.dirtyNextAll;
        s.dirtyValid = known;
        s.dirtyTiles = s.dirtyNext;
        s.dirtyNext.fill(0ull);
        s.dirtyNextAll = false;
      }

      if (mode == Mode::None || mode == Mode::Pending) {
        s.frames[uint32_t(Mode::Pass)]++;
        return;
      }

      s.frames[uint32_t(mode)]++;

      if (mode == Mode::Build || mode == Mode::Verify)
        BindLayer(ctx, s);

      if (mode == Mode::Build) {
        Composite(ctx, s, s.cache);

        s.keys = std::move(s.building);
        s.identities = std::move(s.buildingIds);
        s.cacheGeom = std::move(s.buildGeom);
        s.building = { };
        s.buildingIds = { };
        s.buildGeom = { };
        s.cacheBias = s.buildBiasMin == INT32_MAX ? 0 : s.buildBiasMin;
        s.keyDraws = s.buildDraws;
        s.cacheProj = s.proj;
        s.valid = true;
        s.rebuild = false;
        s.missingFrames = 0u;
        s.uncachedAge.clear();
        return;
      }

      if (mode != Mode::Reuse && mode != Mode::Verify)
        return;

      uint32_t missing = s.keyDraws - std::min(s.keyDraws, s.seenDraws);
      s.missingKeys += missing;

      if (mode == Mode::Verify) {
        // the frame itself stays exact: the redrawn cached keys merge in
        Composite(ctx, s, s.scratch);
        RunVerify(ctx, s, layer, missing);
      }

      if (missing) {
        if (!s.newIdentityHits.empty()) {
          // a cached mesh came back with a new key while a cached key went
          // missing: it moved. Draw that mesh as a moving caster from now on.
          for (const auto& h : s.newIdentityHits)
            s.hot[h.first].until = g_state.frame + c.hotFrames;

          s.rebuild = true;
          s.rebuildReason = ReasonMoved;
        } else if (!c.tolerate) {
          s.rebuild = true;
          s.rebuildReason = ReasonMissing;
        } else if (++s.missingFrames > c.tolerate) {
          s.rebuild = true;
          s.rebuildReason = ReasonTolerate;
        }
      } else {
        s.missingFrames = 0u;
      }

      // new static draws that keep coming back get absorbed by a rebuild
      std::swap(s.uncachedAge, s.uncachedNext);
      s.uncachedNext.clear();

      if (c.promoteFrames && !s.rebuild) {
        for (const auto& e : s.uncachedAge) {
          if (e.second >= c.promoteFrames) {
            s.rebuild = true;
            s.rebuildReason = ReasonPromote;
            break;
          }
        }
      }
    }


    Slice* SliceFor(D3D11DepthStencilView* dsv, uint32_t* layer) {
      if (!dsv || !g_state.array)
        return nullptr;

      Rc<DxvkImageView> view = dsv->GetImageView();

      if (view == nullptr || view->image() != g_state.array)
        return nullptr;

      DxvkImageViewKey key = view->info();

      if (key.layerCount != 1u || key.layerIndex >= MaxSlices || key.mipIndex != 0u)
        return nullptr;

      *layer = key.layerIndex;
      return &g_state.slices[key.layerIndex];
    }


    // blessed: a depth write into a cascade layer outside its cascade (no
    // clear before it): the layer is no longer the cache plus the tracked
    // tiles, so the next frame restores it in full
    void NoteStrayWrite(const D3D11ContextState& state) {
      D3D11DepthStencilView* dsv = state.om.dsv.ptr();

      if (!dsv || !GetConfig().restorePartial || !g_state.array)
        return;

      D3D11DepthStencilState* dss = state.om.dsState.ptr();

      if (dss && (!dss->Desc().DepthEnable || dss->Desc().DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ALL))
        return;

      for (auto& sl : g_state.slices) {
        if (sl.dsv == dsv) {
          sl.dirtyValid = false;
          return;
        }
      }

      // the depth views a frame writes are few and long-lived: remember the
      // last one that is not ours (reset whenever the array is relearned)
      if (dsv == g_state.strayMiss)
        return;

      uint32_t layer = 0u;

      if (Slice* other = SliceFor(dsv, &layer))
        other->dirtyValid = false;
      else
        g_state.strayMiss = dsv;
    }


    std::string UnknownBoundsJson(const Slice& s) {
      std::string out = "{";

      for (uint32_t r = 1u; r < BlessedCascadeBounds::ReasonCount; r++)
        out += str::format(r > 1u ? "," : "", "\"", BlessedCascadeBounds::ReasonName(r), "\":", s.boundsUnknown[r]);

      uint32_t known = 0u, ready = 0u, pending = 0u;
      BlessedCascadeBounds::Counts(&known, &ready, &pending);
      return out + str::format(",\"meshes\":", known, ",\"measured\":", ready, ",\"measuring\":", pending, "}");
    }


    void WriteLog() {
      if (!g_state.logTried) {
        g_state.logTried = true;
        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

        if (!dir.empty()) {
          std::error_code ec;
          std::filesystem::create_directories(dir, ec);
          g_state.log.open(dir + env::PlatformDirSlash + "cascade_cache.jsonl", std::ios::out | std::ios::app);
          g_state.projLog.open(dir + env::PlatformDirSlash + "cascade_cache_proj.jsonl", std::ios::out | std::ios::app);
        }
      }

      double t = std::chrono::duration<double>(dxvk::high_resolution_clock::now() - g_state.start).count();
      double n = double(std::max<uint64_t>(g_state.presentsInWindow, 1u));

      std::string line = str::format("{\"t\":", t, ",\"frames\":", g_state.presentsInWindow,
        ",\"learned\":", g_state.array ? 1 : 0, ",\"size\":", g_state.arrayExtent.width, ",\"slices\":[");

      for (uint32_t i = 0; i < MaxSlices; i++) {
        Slice& s = g_state.slices[i];

        if (i)
          line += ",";

        line += str::format("{\"layer\":", i,
          ",\"pass\":", s.frames[uint32_t(Mode::Pass)],
          ",\"build\":", s.frames[uint32_t(Mode::Build)],
          ",\"reuse\":", s.frames[uint32_t(Mode::Reuse)],
          ",\"verify\":", s.frames[uint32_t(Mode::Verify)],
          ",\"cached_draws\":", s.valid ? s.keyDraws : 0u,
          ",\"skipped_per_frame\":", double(s.skipped) / n,
          ",\"redrawn_moving_per_frame\":", double(s.redrawnMoving) / n,
          ",\"redrawn_new_per_frame\":", double(s.redrawnNew) / n,
          ",\"missing_keys\":", s.missingKeys,
          ",\"hot_meshes\":", s.hot.size(),
          ",\"draws_per_frame\":", double(s.draws) / n,
          ",\"proj_changes\":", s.projChanges,
          ",\"restores\":", s.restores,
          ",\"restores_skipped\":", s.restoresSkipped,
          ",\"hot_reheats\":", s.hotReheats,
          ",\"partial\":{\"restores\":", s.partialRestores,
          ",\"tiles_per_restore\":", s.partialRestores ? double(s.partialTiles) / double(s.partialRestores) : 0.0,
          ",\"cover\":", s.partialRestores ? double(s.partialTiles) / double(s.partialRestores) / double(Slice::Tiles * Slice::Tiles) : 0.0,
          ",\"rects_per_restore\":", s.partialRestores ? double(s.partialRects) / double(s.partialRestores) : 0.0,
          ",\"fallback_cover\":", s.partialFallbackCover,
          ",\"fallback_rects\":", s.partialFallbackRects,
          ",\"not_deferred\":", s.partialNotDeferred,
          ",\"unknown_bounds\":", UnknownBoundsJson(s), "}",
          ",\"tol_texels\":", GetConfig().tolTexels,
          ",\"tol_depth\":", Tolerant() ? DepthTolerance(s) : 0.0,
          ",\"cached_depth_bias\":", s.cacheBias,
          ",\"reuse_err_texels_max\":", s.errTexelsMax,
          ",\"reuse_err_depth_max\":", s.errDepthMax,
          ",\"rebuilds\":{");

        for (uint32_t r = 0; r < ReasonCount; r++)
          line += str::format(r ? "," : "", "\"", ReasonName(r), "\":", s.reasons[r]);

        // why draws were not cached (counted outside pass mode), per frame
        line += "},\"not_static_per_frame\":{";

        for (uint32_t r = 0; r < RejectCount; r++)
          line += str::format(r ? "," : "", "\"", RejectName(r), "\":", double(s.rejects[r]) / n);

        line += str::format("},\"proj_read_failed\":", s.rejects[RejectCount], "}");

        std::fill(std::begin(s.rejects), std::end(s.rejects), 0u);
        s.draws = 0u;
        s.projChanges = 0u;
        s.restores = 0u;
        s.restoresSkipped = 0u;
        s.hotReheats = 0u;
        s.partialRestores = s.partialTiles = s.partialRects = 0u;
        s.partialFallbackCover = s.partialFallbackRects = s.partialNotDeferred = 0u;
        std::fill(std::begin(s.boundsUnknown), std::end(s.boundsUnknown), 0u);
        s.errTexelsMax = 0.0;
        s.errDepthMax = 0.0;

        std::fill(std::begin(s.frames), std::end(s.frames), 0u);
        std::fill(std::begin(s.reasons), std::end(s.reasons), 0u);
        s.skipped = s.redrawnMoving = s.redrawnNew = s.missingKeys = 0u;
      }

      line += "]";

      if (!g_state.verifyLines.empty()) {
        line += ",\"verify\":[" + g_state.verifyLines + "]";
        g_state.verifyLines.clear();
      }

      line += "}\n";

      if (g_state.log.is_open()) {
        g_state.log << line;
        g_state.log.flush();
      }

      if (g_state.projLog.is_open() && !g_state.projLines.empty()) {
        g_state.projLog << g_state.projLines;
        g_state.projLog.flush();
      }

      g_state.projLines.clear();

      g_state.presentsInWindow = 0u;
    }


    void ReadVerifies() {
      // blessed: eight presents is well past dxvk's frame latency limit,
      // so the gpu has long written these host-coherent counters
      while (!g_state.verifies.empty() && g_state.frame - g_state.verifies.front().frame >= 8u) {
        PendingVerify pv = std::move(g_state.verifies.front());
        g_state.verifies.pop_front();

        BlessedCascadeVerifyCounters v = { };
        std::memcpy(&v, pv.counters->getSliceInfo().mapPtr, sizeof(v));

        std::string entry = str::format("{\"frame\":", pv.frame, ",\"layer\":", pv.slice,
          ",\"differ\":", v.differ, ",\"ghost\":", v.ghost, ",\"missing\":", v.missing,
          ",\"max_diff\":", v.maxDiff, ",\"missing_keys\":", pv.missingKeys,
          ",\"proj_err_texels\":", pv.errTexels, ",\"proj_err_depth\":", pv.errDepth,
          ",\"kind\":\"", pv.restoreCheck ? "restore" : "cache", "\"}");

        if (!g_state.verifyLines.empty())
          g_state.verifyLines += ",";
        g_state.verifyLines += entry;

        if (v.differ)
          Logger::warn(str::format("BlessedCascadeCache: verify layer ", pv.slice, ": ", v.differ,
            " texels differ (ghost ", v.ghost, ", missing ", v.missing, ", max ", v.maxDiff, ")"));
      }
    }

  }


  namespace blessed_cascade_cache_detail {
    extern const bool g_enabled = ComputeEnabled();
  }


  bool BlessedCascadeCache::OnDraw(
          D3D11ImmediateContext*  ctx,
    const D3D11ContextState&      state,
          BlessedCascadeDrawKind  kind,
          uint32_t                indexCount,
          uint32_t                startIndex,
          int32_t                 baseVertex) {
    // a draw with a colour target: never a cascade draw, maybe the mask
    // draw that names the cascade array
    if (state.om.rtvs[0].ptr()) {
      if (g_state.open)
        CloseOpen(ctx);

      TryLearn(state);
      NoteStrayWrite(state);
      return false;
    }

    Slice* sp = g_state.open;

    if (!sp) {
      NoteStrayWrite(state);
      return false;
    }

    Slice& s = *sp;

    if (state.om.dsv.ptr() != s.dsv) {
      uint32_t layer = 0u;

      if (SliceFor(state.om.dsv.ptr(), &layer) != sp || layer != g_state.openLayer) {
        CloseOpen(ctx);
        NoteStrayWrite(state);
        return false;
      }
    }

    s.draws++;

    if (s.mode == Mode::Pending)
      Decide(ctx, s, g_state.openLayer, state);

    if (s.mode == Mode::Pass)
      return false;

    // the game re-bound its targets since we bound ours: the layer (or
    // whatever it bound) is what dxvk has now
    if (s.boundToAlt && state.om.blessedOmGeneration != s.altGeneration)
      s.boundToAlt = false;

    const CascadeConfig& c = GetConfig();

    StaticDraw d;
    uint32_t reject = RejectCount;
    bool isStatic = StaticKey(state, s, kind, indexCount, startIndex, baseVertex, &d, &reject);

    if (!isStatic)
      s.rejects[reject]++;

    if (isStatic && !s.hot.empty()) {
      auto h = s.hot.find(d.identity);

      if (h != s.hot.end()) {
        Slice::Hot& e = h->second;

        if (e.serial != g_state.serial) {
          // first draw of this mesh in this cascade: did the last cascade's
          // draws differ from the one before? (key + world position, so a
          // cascade camera step no longer counts as the mesh moving)
          if (e.serial && e.sumCur != e.sumPrev) {
            e.until = g_state.frame + c.hotFrames;
            s.hotReheats++;

            if (s.reheatLogged < 48u && !e.geomPrev.empty())
              NoteReheat(s, g_state.openLayer, d, e);
          }

          e.sumPrev = e.sumCur;
          e.sumCur  = 0u;
          e.serial  = g_state.serial;

          if (s.reheatLogged < 48u && d.geom) {
            e.geomPrev = std::move(e.geomCur);
            std::memcpy(e.posPrev, e.posCur, sizeof(e.posPrev));
            e.geomCur.assign(d.geom, d.geom + std::min(d.geomSize, 128u));
            std::memcpy(e.posCur, d.pos, sizeof(e.posCur));
          }
        }

        e.sumCur += d.instance * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;

        if (g_state.frame < e.until)
          isStatic = false;
        else
          s.hot.erase(h);
      }
    }

    if (s.mode == Mode::Build) {
      if (isStatic) {
        BindAlt(ctx, s, s.cache, state);

        KeyInst inst;
        std::memcpy(inst.pos, d.pos, sizeof(inst.pos));
        s.building[d.key].insts.push_back(inst);
        s.buildingIds[d.identity]++;
        s.buildDraws++;

        D3D11RasterizerState* rs = state.rs.state.ptr();
        s.buildBiasMin = std::min(s.buildBiasMin, rs ? int32_t(std::abs(rs->Desc().DepthBias)) : 0);

        if (s.buildGeom.size() < 8192u && d.geom)
          s.buildGeom.emplace(d.identity, std::vector<uint8_t>(d.geom, d.geom + std::min(d.geomSize, 128u)));
      } else {
        BindLayer(ctx, s);

        if (c.restorePartial)
          TrackLayerDraw(s, state, kind, indexCount, startIndex, baseVertex);
      }

      return false;
    }

    // reuse and verify: a cached draw has the same key and stands within
    // BLESSED_CASCADE_CACHE_POS_TOL world units of where it was cached
    bool cached = false;

    if (isStatic) {
      auto it = s.keys.find(d.key);

      if (it != s.keys.end()) {
        double tol = double(c.posTol);

        for (KeyInst& inst : it->second.insts) {
          if (inst.serial == g_state.serial)
            continue;

          if (std::abs(inst.pos[0] - d.pos[0]) <= tol
           && std::abs(inst.pos[1] - d.pos[1]) <= tol
           && std::abs(inst.pos[2] - d.pos[2]) <= tol) {
            inst.serial = g_state.serial;
            s.seenDraws++;
            cached = true;
            break;
          }
        }
      }

      if (!cached) {
        if (s.identities.find(d.identity) != s.identities.end()) {
          if (s.newIdentityHits.size() < 64u)
            s.newIdentityHits.push_back({ d.identity, d.key });

          NoteMoved(s, g_state.openLayer, d);
        }

        auto age = s.uncachedAge.find(d.instance);
        s.uncachedNext[d.instance] = age != s.uncachedAge.end() ? age->second + 1u : 1u;
      }
    }

    if (s.mode == Mode::Verify) {
      // cached keys are redrawn into scratch (compared with the cache at
      // the end), everything else into the layer as usual
      if (cached)
        BindAlt(ctx, s, s.scratch, state);
      else
        BindLayer(ctx, s);

      if (cached) s.skipped++;
      else if (isStatic) s.redrawnNew++;
      else s.redrawnMoving++;
      return false;
    }

    if (cached) {
      s.skipped++;
      return true;
    }

    if (isStatic) s.redrawnNew++;
    else s.redrawnMoving++;

    if (c.restorePartial)
      TrackLayerDraw(s, state, kind, indexCount, startIndex, baseVertex);

    return c.debugMode == Debug::Cached;
  }


  bool BlessedCascadeCache::OnClearDsv(
          D3D11ImmediateContext*  ctx,
          D3D11DepthStencilView*  dsv,
          uint32_t                clearFlags,
          float                   depth) {
    if (g_state.open)
      CloseOpen(ctx);

    uint32_t layer = 0u;
    Slice* s = SliceFor(dsv, &layer);

    if (!s || !(clearFlags & D3D11_CLEAR_DEPTH))
      return false;

    const auto& extent = g_state.arrayRef->info().extent;

    s->dsv        = dsv;
    s->dsvView    = dsv->GetImageView();
    s->extent     = { extent.width, extent.height };
    s->clearDepth = depth;
    s->mode       = Mode::Pending;

    g_state.open           = s;
    g_state.openLayer      = layer;
    g_state.serial++;
    g_state.arrayLastClear = g_state.frame;

    // blessed: the partial restore needs last frame's layer, so the clear
    // waits for the mode; Decide (or CloseOpen) runs it if it is needed
    const CascadeConfig& c = GetConfig();
    s->clearDeferred = c.restorePartial && c.debugMode == Debug::Off
                    && s->valid && s->dirtyValid && s->keyDraws
                    && depth == s->clearDepthLast && clearFlags == D3D11_CLEAR_DEPTH;
    s->clearDepthLast = depth;
    return s->clearDeferred;
  }


  void BlessedCascadeCache::OnPresent(
          D3D11ImmediateContext*  ctx) {
    // blessed: called from the swap chain, not a context method
    D3D10DeviceLock lock = ctx->LockContext();

    if (g_state.open)
      CloseOpen(ctx);

    if (!g_state.startSet) {
      g_state.startSet = true;
      g_state.start = dxvk::high_resolution_clock::now();
    }

    g_state.frame++;

    ReadVerifies();

    // blessed: the partial restore's mesh bounds: read back what finished,
    // measure what this frame met for the first time
    if (GetConfig().restorePartial) {
      BlessedCascadeBounds::ReadBack(g_state.frame);

      BlessedCascadeBoundsArgs args;

      if (BlessedCascadeBounds::TakeRequests(BlessedCascadeCacheCtx::Device(ctx), &args, g_state.frame)) {
        BlessedCascadeCacheCtx::Emit(ctx, [cArgs = std::move(args)] (DxvkContext* dxvkCtx) {
          dxvkCtx->blessedRunCascadeBounds(cArgs);
        });
      }
    }

    if (++g_state.presentsInWindow >= 120u)
      WriteLog();
  }


  void BlessedCascadeCache::NotifySwapchainExtent(
          uint32_t                width,
          uint32_t                height) {
    if (width == g_state.swapW && height == g_state.swapH)
      return;

    g_state.swapW = width;
    g_state.swapH = height;

    ResetSlices();
    g_state.array    = nullptr;
    g_state.arrayRef = nullptr;
    g_state.lastSrv  = nullptr;
  }

}
