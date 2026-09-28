// blessed: point-light shadow mask draws, app-thread half -- see blessed_point_shadow.h
#include "blessed_point_shadow.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d11_context_imm.h"
#include "d3d11_texture.h"

#include "../dxvk/blessed/blessed_point_shadow.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // blessed: which vanilla utility mask shader a pixel shader is, by the
    // first 8 hex digits of its dxbc checksum (docs/research/point-lights.md,
    // "the vanilla hashes": filters 0, 1, 2 and 4/8 of each kind)
    enum class MaskKind : uint8_t {
      None,
      Sun,
      Spot,
      Paraboloid,     // single and dual: both are a rigid light-space transform
      Unknown448,     // BLESSED_POINT_DETECT=b2: not in the table, 448-byte b2
    };

    struct KnownHash {
      const char* prefix;
      MaskKind    kind;
      const char* label;
    };

    const KnownHash s_knownHashes[] = {
      { "0a42b955", MaskKind::Sun, "sun" },         { "38253679", MaskKind::Sun, "sun" },
      { "1dec460f", MaskKind::Sun, "sun" },         { "aaec0a89", MaskKind::Sun, "sun" },
      { "23dffdfd", MaskKind::Sun, "sun+focus" },   { "7c75eb3c", MaskKind::Sun, "sun+focus" },
      { "086a8d72", MaskKind::Sun, "sun+focus" },   { "b070feb5", MaskKind::Sun, "sun+focus" },
      { "43c25e60", MaskKind::Spot, "spot" },       { "0ba43c56", MaskKind::Spot, "spot" },
      { "b8aadb7a", MaskKind::Spot, "spot" },       { "a0d1560b", MaskKind::Spot, "spot" },
      { "870e3306", MaskKind::Spot, "spot+focus" }, { "7baf2259", MaskKind::Spot, "spot+focus" },
      { "11f4c3f3", MaskKind::Spot, "spot+focus" }, { "21346819", MaskKind::Spot, "spot+focus" },
      { "48b067d7", MaskKind::Paraboloid, "pb" },   { "60862f52", MaskKind::Paraboloid, "pb" },
      { "09869777", MaskKind::Paraboloid, "pb" },   { "d490fbff", MaskKind::Paraboloid, "pb" },
      { "66aa7d11", MaskKind::Paraboloid, "dpb" },  { "9fdaccef", MaskKind::Paraboloid, "dpb" },
      { "1f7234ed", MaskKind::Paraboloid, "dpb" },  { "190fc694", MaskKind::Paraboloid, "dpb" },
    };

    struct ShaderClass {
      MaskKind    kind  = MaskKind::None;
      const char* label = "none";
    };

    ShaderClass ClassifyName(const std::string& name) {
      if (name.size() < 11 || name.compare(0, 3, "fs.") != 0)
        return ShaderClass();

      for (const auto& h : s_knownHashes) {
        if (name.compare(3, 8, h.prefix) == 0)
          return ShaderClass { h.kind, h.label };
      }

      return ShaderClass();
    }

    // "<vs|ps>:<slot>:<byte offset>", as BLESSED_SHADOW_INVVP / _CAMPOS
    struct CbSlot {
      bool             valid  = false;
      D3D11ShaderType  stage  = D3D11ShaderType::ePixel;
      uint32_t         slot   = 0;
      uint32_t         offset = 0;
    };

    CbSlot ParseCbSlot(const std::string& s, uint32_t defSlot, uint32_t defOffset) {
      CbSlot out;
      out.valid  = true;
      out.slot   = defSlot;
      out.offset = defOffset;

      size_t a = s.find(':');
      size_t b = a == std::string::npos ? std::string::npos : s.find(':', a + 1);
      if (a == std::string::npos || b == std::string::npos)
        return out;

      out.stage  = s.substr(0, a) == "vs" ? D3D11ShaderType::eVertex : D3D11ShaderType::ePixel;
      out.slot   = std::strtoul(s.substr(a + 1, b - a - 1).c_str(), nullptr, 10);
      out.offset = std::strtoul(s.substr(b + 1).c_str(), nullptr, 10);
      return out;
    }

    struct PointConfig {
      bool     trace       = false;  // BLESSED_POINT_SHADOWS=1
      bool     sunGate     = false;  // rtshadow, unless BLESSED_SHADOW_SUN_GATE=0
      bool     skipRaster  = false;  // BLESSED_POINT_SKIP_RASTER=1
      bool     detectB2    = false;  // BLESSED_POINT_DETECT=b2
      bool     projCols    = false;  // BLESSED_POINT_PROJ=cols: ShadowMapProj rows are columns
      CbSlot   invVp;
      CbSlot   camPos;
      bool     depthFromDsv = false; // BLESSED_POINT_DEPTH=dsv
      uint32_t depthSrvSlot = 2;     // BLESSED_POINT_DEPTH=srv:<n>, default srv:2
      float    farValue     = 1.0f;
      float    proxyRadius  = 24.0f; // BLESSED_POINT_PROXY
      uint32_t debugMode    = 0;     // BLESSED_POINT_DEBUG
      uint64_t dumpAt       = 0;     // BLESSED_POINT_DUMP=<n>: the nth traced light
      bool     soft         = false; // BLESSED_POINT_SOFT=1
      float    lightRadius  = 4.0f;  // BLESSED_POINT_LIGHT_RADIUS
      uint32_t spp          = 1;     // BLESSED_POINT_SPP
    };

    // blessed: env-only, safe at static init (no Logger, no other statics)
    bool ComputePointTrace() {
      return env::getEnvVar("BLESSED_POINT_SHADOWS") == "1";
    }

    bool ComputeSunGate() {
      return env::getEnvVar("BLESSED_HOOK_MODE") == "rtshadow"
          && env::getEnvVar("BLESSED_SHADOW_SUN_GATE") != "0";
    }

    // blessed: lazy, first reached from a real draw once g_enabled is true
    const PointConfig& GetConfig() {
      static PointConfig s_config = [] {
        PointConfig c;
        c.trace      = ComputePointTrace();
        c.sunGate    = ComputeSunGate();
        c.skipRaster = c.trace && env::getEnvVar("BLESSED_POINT_SKIP_RASTER") == "1";
        c.detectB2   = env::getEnvVar("BLESSED_POINT_DETECT") == "b2";
        c.projCols   = env::getEnvVar("BLESSED_POINT_PROJ") == "cols";

        c.invVp  = ParseCbSlot(env::getEnvVar("BLESSED_SHADOW_INVVP"), 12, 512);
        c.camPos = ParseCbSlot(env::getEnvVar("BLESSED_SHADOW_CAMPOS"), 12, 640);

        std::string depth = env::getEnvVar("BLESSED_POINT_DEPTH");
        if (depth == "dsv")
          c.depthFromDsv = true;
        else if (depth.rfind("srv:", 0) == 0)
          c.depthSrvSlot = std::strtoul(depth.c_str() + 4, nullptr, 10);

        c.farValue = env::getEnvVar("BLESSED_SHADOW_FAR") == "0" ? 0.0f : 1.0f;

        std::string proxy = env::getEnvVar("BLESSED_POINT_PROXY");
        if (!proxy.empty())
          c.proxyRadius = std::max(0.0f, std::strtof(proxy.c_str(), nullptr));

        std::string dbg = env::getEnvVar("BLESSED_POINT_DEBUG");
        c.debugMode = dbg == "lit" ? 1u : dbg == "dark" ? 2u : dbg == "rings" ? 3u
                    : dbg == "radius" ? 4u : dbg == "t" ? 5u : dbg == "raster" ? 6u
                    : dbg == "hist" ? 7u : 0u;

        std::string dump = env::getEnvVar("BLESSED_POINT_DUMP");
        c.dumpAt = std::strtoull(dump.c_str(), nullptr, 10);
        if (c.dumpAt == 1u)
          c.dumpAt = 120u;

        c.soft = env::getEnvVar("BLESSED_POINT_SOFT") == "1";
        std::string lr = env::getEnvVar("BLESSED_POINT_LIGHT_RADIUS");
        if (!lr.empty())
          c.lightRadius = std::max(0.0f, std::strtof(lr.c_str(), nullptr));
        std::string spp = env::getEnvVar("BLESSED_POINT_SPP");
        if (!spp.empty())
          c.spp = std::min<uint32_t>(std::max<uint32_t>(std::strtoul(spp.c_str(), nullptr, 10), 1u), 4u);

        Logger::info(str::format("BlessedPointShadow: trace=", c.trace ? 1 : 0,
          " sun_gate=", c.sunGate ? 1 : 0, " skip_raster=", c.skipRaster ? 1 : 0,
          " detect_b2=", c.detectB2 ? 1 : 0, " proxy=", c.proxyRadius,
          " debug=", c.debugMode, " soft=", c.soft ? 1 : 0));
        return c;
      }();

      return s_config;
    }

    // ---- app-thread state (immediate context, under the device lock) ----

    std::unordered_map<const void*, ShaderClass> g_classCache;
    const void*  g_lastPs    = nullptr;
    ShaderClass  g_lastClass;

    uint32_t g_claimThisFrame = 0u;   // channel bits claimed by point mask draws
    uint32_t g_claimLastFrame = 0u;
    uint64_t g_frameId        = 1u;   // presents seen + 1
    uint64_t g_traced         = 0u;   // lights traced since enable

    // blessed: point shadow maps to drop, as (image, slice): learned from
    // each mask draw's t3/t4 and EndSplitDistances.x this frame, used the
    // next (the maps are drawn before their mask draw)
    struct MapKey {
      DxvkImage* image;
      uint32_t   slice;
      bool operator == (const MapKey& o) const { return image == o.image && slice == o.slice; }
    };
    std::vector<MapKey> g_mapsLearning;
    std::vector<MapKey> g_mapsToSkip;

    // what the last frame's lights looked like, for the log
    struct LightRecord {
      const char* label;
      uint32_t    channels;
      float       origin[3];
      float       radius;
      float       c3[4];
      float       det;
      int32_t     slice;
      bool        solved;
    };
    std::vector<LightRecord> g_lightsThisFrame;
    std::vector<LightRecord> g_lightsLastFrame;

    // ---- pointshadow.jsonl window (120 presents) ----
    struct Window {
      uint64_t frames         = 0;
      uint64_t sunDraws       = 0;
      uint64_t pointDraws     = 0;
      uint64_t traced         = 0;
      uint64_t skippedMasks   = 0;
      uint64_t skippedMaps    = 0;
      uint64_t rasterKept     = 0;   // SKIP_RASTER, but the trace could not go out
      uint64_t sunGateClosed  = 0;
      uint64_t unsolved       = 0;
      uint64_t noStorage      = 0;
      uint64_t noDepth        = 0;
      uint64_t noCamera       = 0;
    };
    Window        g_win;
    std::ofstream g_logFile;
    bool          g_logTried = false;
    dxvk::high_resolution_clock::time_point g_start;

    // sun gate: the last verdict, logged on change
    int32_t g_sunGateLast = -1;

    const ShaderClass& ClassifyPs(const D3D11ContextState& state) {
      static const ShaderClass s_none;
      D3D11PixelShader* ps = state.ps.ptr();
      if (!ps)
        return s_none;

      const void* key = ps;
      if (key == g_lastPs)
        return g_lastClass;

      auto it = g_classCache.find(key);
      if (it == g_classCache.end())
        it = g_classCache.emplace(key, ClassifyName(ps->GetCommonShader()->GetName())).first;

      g_lastPs    = key;
      g_lastClass = it->second;
      return g_lastClass;
    }

    // bytes the bound constant buffer exposes to the shader (0 = unbound)
    uint32_t BoundCbSize(const D3D11ContextState& state, D3D11ShaderType stage, uint32_t slot) {
      const auto& cbv = state.cbv[stage];
      if (slot >= cbv.maxCount)
        return 0u;

      const auto& cb = cbv.buffers[slot];
      D3D11Buffer* buffer = cb.buffer.ptr();
      if (!buffer)
        return 0u;

      uint32_t width  = buffer->Desc()->ByteWidth;
      uint32_t offset = cb.constantOffset * 16u;
      uint32_t size   = offset < width ? width - offset : 0u;
      if (cb.constantCount)
        size = std::min(size, cb.constantCount * 16u);
      return size;
    }

    bool ReadCb(const D3D11ContextState& state, const CbSlot& slot, float* out, uint32_t count) {
      if (!slot.valid)
        return false;

      const auto& cbv = state.cbv[slot.stage];
      if (slot.slot >= cbv.maxCount)
        return false;

      D3D11Buffer* buffer = cbv.buffers[slot.slot].buffer.ptr();
      if (!buffer)
        return false;

      void* mapPtr = buffer->GetMapPtr();
      if (!mapPtr)
        return false;
      buffer->GetBuffer()->blessedMarkCpuRead(); // blessed: perf-halfrate -- keep its ring chunks cached

      uint32_t byteOffset = cbv.buffers[slot.slot].constantOffset * 16u + slot.offset;
      if (byteOffset + count * sizeof(float) > buffer->Desc()->ByteWidth)
        return false;

      std::memcpy(out, reinterpret_cast<const uint8_t*>(mapPtr) + byteOffset, count * sizeof(float));
      return true;
    }

    uint32_t WriteMask0(const D3D11ContextState& state) {
      uint32_t mask = D3D11_COLOR_WRITE_ENABLE_ALL;
      if (auto* blend = state.om.cbState.ptr())
        mask = blend->Desc().RenderTarget[0].RenderTargetWriteMask;
      return mask & 0xFu;
    }

    // blessed: solves a * x = b for a 3x3 (row-major), Cramer's rule
    bool Solve3(const float a[9], const float b[3], float x[3], float* detOut) {
      float det = a[0] * (a[4] * a[8] - a[5] * a[7])
                - a[1] * (a[3] * a[8] - a[5] * a[6])
                + a[2] * (a[3] * a[7] - a[4] * a[6]);
      *detOut = det;

      if (!std::isfinite(det) || std::fabs(det) < 1.0e-12f)
        return false;

      float inv = 1.0f / det;
      x[0] = (b[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b[1] * a[8] - a[5] * b[2]) + a[2] * (b[1] * a[7] - a[4] * b[2])) * inv;
      x[1] = (a[0] * (b[1] * a[8] - a[5] * b[2]) - b[0] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * b[2] - b[1] * a[6])) * inv;
      x[2] = (a[0] * (a[4] * b[2] - b[1] * a[7]) - a[1] * (a[3] * b[2] - b[1] * a[6]) + b[0] * (a[3] * a[7] - a[4] * a[6])) * inv;
      return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
    }

    /**
     * blessed: the light's camera-relative origin from ShadowMapProj[0]
     * (ps b2 c16..c19, one output row each: posLS.k = dot(row_k.xyz, p) + row_k.w).
     *  - paraboloid: rows 0..2 are a rigid transform [R | t], origin = -R^-1 t
     *  - spot: a perspective projection, origin = where rows x, y and w all vanish
     * An unknown shader picks spot if row w is not (0, 0, 0, *).
     */
    bool SolveOrigin(MaskKind kind, const float m[16], bool cols, float origin[3], float* det, bool* isSpot) {
      float r[16];
      for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < 4; j++)
          r[i * 4 + j] = cols ? m[j * 4 + i] : m[i * 4 + j];
      }

      bool spot = kind == MaskKind::Spot;
      if (kind == MaskKind::Unknown448)
        spot = std::fabs(r[12]) + std::fabs(r[13]) + std::fabs(r[14]) > 1.0e-6f;
      *isSpot = spot;

      const uint32_t rows[3] = { 0u, 1u, spot ? 3u : 2u };
      float a[9], b[3];
      for (uint32_t k = 0; k < 3; k++) {
        a[k * 3 + 0] = r[rows[k] * 4 + 0];
        a[k * 3 + 1] = r[rows[k] * 4 + 1];
        a[k * 3 + 2] = r[rows[k] * 4 + 2];
        b[k]         = -r[rows[k] * 4 + 3];
      }

      return Solve3(a, b, origin, det);
    }

    void LearnMaps(const D3D11ContextState& state, int32_t slice) {
      if (slice < 0)
        return;

      const auto& srvs = state.srv[D3D11ShaderType::ePixel];
      for (uint32_t slot = 3; slot <= 4; slot++) {
        if (slot >= srvs.maxCount)
          continue;

        D3D11ShaderResourceView* srv = srvs.views[slot].ptr();
        if (!srv)
          continue;

        Rc<DxvkImageView> view = srv->GetImageView();
        if (!view || !view->image())
          continue;

        MapKey key = { view->image(), uint32_t(slice) };
        if (std::find(g_mapsLearning.begin(), g_mapsLearning.end(), key) == g_mapsLearning.end())
          g_mapsLearning.push_back(key);
      }
    }

    bool IsLearnedMap(const D3D11_VK_VIEW_INFO& vi, const Rc<DxvkImageView>& view) {
      if (vi.Image.NumLayers != 1u || !view)
        return false;

      MapKey key = { view->image(), vi.Image.MinLayer };
      return std::find(g_mapsToSkip.begin(), g_mapsToSkip.end(), key) != g_mapsToSkip.end();
    }

    // blessed: everything that happens at one point mask draw, whether it
    // was drawn (post-draw) or is about to be dropped (pre-draw,
    // BLESSED_POINT_SKIP_RASTER). True: a trace was dispatched for it.
    bool HandlePointMaskDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state, const ShaderClass& cls) {
      const PointConfig& cfg = GetConfig();

      uint32_t channels = WriteMask0(state);
      g_claimThisFrame |= channels;
      g_win.pointDraws++;

      if (!cfg.trace || !channels)
        return false;

      LightRecord rec = { };
      rec.label    = cls.label;
      rec.channels = channels;
      rec.slice    = -1;

      float endSplit[4];
      CbSlot b0c2 = { true, D3D11ShaderType::ePixel, 0u, 32u };
      if (ReadCb(state, b0c2, endSplit, 4))
        rec.slice = int32_t(std::lround(endSplit[0]));

      if (cfg.skipRaster)
        LearnMaps(state, rec.slice);

      float proj[16] = { };
      CbSlot b2c3  = { true, D3D11ShaderType::ePixel, 2u, 48u };
      CbSlot b2c16 = { true, D3D11ShaderType::ePixel, 2u, 256u };
      bool haveParam = ReadCb(state, b2c3, rec.c3, 4);
      bool haveProj  = ReadCb(state, b2c16, proj, 16);

      bool spot = cls.kind == MaskKind::Spot;
      rec.solved = haveProj && SolveOrigin(cls.kind, proj, cfg.projCols, rec.origin, &rec.det, &spot);

      // blessed: c3.x is the radius for the paraboloid kinds; for spot it is
      // a cone falloff, so no cull there (a wrong cull would unshadow pixels)
      rec.radius = (haveParam && !spot && rec.c3[0] > 0.0f) ? rec.c3[0] : 0.0f;

      if (g_lightsThisFrame.size() < 8u)
        g_lightsThisFrame.push_back(rec);

      if (!rec.solved) {
        g_win.unsolved++;
        return false;
      }

      D3D11RenderTargetView* rtv0 = state.om.maxRtv > 0 ? state.om.rtvs[0].ptr() : nullptr;
      if (!rtv0)
        return false;

      Rc<DxvkImageView> rtvView = rtv0->GetImageView();
      DxvkImage* outputImage = rtvView->image();
      if (!(outputImage->info().usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        g_win.noStorage++;
        return false;
      }

      Rc<DxvkImageView> depthSrc;
      if (cfg.depthFromDsv) {
        if (D3D11DepthStencilView* dsv = state.om.dsv.ptr())
          depthSrc = dsv->GetImageView();
      } else {
        const auto& srvs = state.srv[D3D11ShaderType::ePixel];
        if (cfg.depthSrvSlot < srvs.maxCount) {
          if (D3D11ShaderResourceView* srv = srvs.views[cfg.depthSrvSlot].ptr())
            depthSrc = srv->GetImageView();
        }
      }

      if (!depthSrc || !(depthSrc->image()->formatInfo()->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        g_win.noDepth++;
        return false;
      }

      BlessedPointShadowDispatchArgs args;

      if (!ReadCb(state, cfg.invVp, args.invViewProj, 16)) {
        g_win.noCamera++;
        return false;
      }
      ReadCb(state, cfg.camPos, args.camPosNow, 3); // ok to fail: stays 0

      DxvkImage* depthImage = depthSrc->image();

      DxvkImageViewKey depthKey = { };
      depthKey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
      depthKey.usage      = VK_IMAGE_USAGE_SAMPLED_BIT;
      depthKey.format     = depthImage->info().format;
      depthKey.layout     = VK_IMAGE_LAYOUT_GENERAL;
      depthKey.aspects    = VK_IMAGE_ASPECT_DEPTH_BIT;
      depthKey.mipIndex   = 0u;
      depthKey.mipCount   = 1u;
      depthKey.layerIndex = 0u;
      depthKey.layerCount = 1u;

      DxvkImageViewKey outKey = { };
      outKey.viewType   = VK_IMAGE_VIEW_TYPE_2D;
      outKey.usage      = VK_IMAGE_USAGE_STORAGE_BIT;
      outKey.format     = outputImage->info().format;
      outKey.layout     = VK_IMAGE_LAYOUT_GENERAL;
      outKey.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
      outKey.mipIndex   = 0u;
      outKey.mipCount   = 1u;
      outKey.layerIndex = 0u;
      outKey.layerCount = 1u;

      VkExtent3D extent = outputImage->info().extent;

      args.depthView     = depthImage->createView(depthKey);
      args.outputView    = outputImage->createView(outKey);
      args.width         = extent.width;
      args.height        = extent.height;
      args.lightPos[0]   = rec.origin[0];
      args.lightPos[1]   = rec.origin[1];
      args.lightPos[2]   = rec.origin[2];
      args.radius        = rec.radius;
      args.proxyRadius   = cfg.proxyRadius;
      args.farDepthValue = cfg.farValue;
      args.channelMask   = channels;
      args.debugMode     = cfg.debugMode;
      args.softEnabled   = cfg.soft;
      args.lightRadius   = cfg.lightRadius;
      args.spp           = cfg.spp;
      args.frameId       = g_frameId;

      g_traced++;
      g_win.traced++;

      if (cfg.dumpAt && g_traced == cfg.dumpAt) {
        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
        if (!dir.empty()) {
          args.dump     = true;
          args.dumpPath = dir + env::PlatformDirSlash + str::format("pointshadow-", cfg.dumpAt, ".pgm");
        }
      }

      BlessedPointShadow::EmitDispatch(ctx, std::move(args));
      return true;
    }

    void WriteWindow() {
      if (!g_logTried) {
        g_logTried = true;
        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
        if (!dir.empty()) {
          std::error_code ec;
          std::filesystem::create_directories(dir, ec);
          g_logFile.open(dir + env::PlatformDirSlash + "pointshadow.jsonl", std::ios::out | std::ios::app);
        }
      }

      if (!g_logFile.is_open())
        return;

      double f = double(std::max<uint64_t>(g_win.frames, 1u));
      double t = std::chrono::duration<double>(dxvk::high_resolution_clock::now() - g_start).count();

      std::string line = str::format("{\"t\":", t,
        ",\"frames\":", g_win.frames,
        ",\"sun_mask_draws\":", double(g_win.sunDraws) / f,
        ",\"point_mask_draws\":", double(g_win.pointDraws) / f,
        ",\"traced\":", double(g_win.traced) / f,
        ",\"skipped_mask_draws\":", double(g_win.skippedMasks) / f,
        ",\"skipped_map_draws\":", double(g_win.skippedMaps) / f,
        ",\"raster_kept\":", g_win.rasterKept,
        ",\"sun_gate_closed\":", double(g_win.sunGateClosed) / f,
        ",\"unsolved\":", g_win.unsolved,
        ",\"no_storage\":", g_win.noStorage,
        ",\"no_depth\":", g_win.noDepth,
        ",\"no_camera\":", g_win.noCamera,
        ",\"maps_learned\":", g_mapsToSkip.size(),
        ",\"claimed\":", g_claimLastFrame,
        ",\"lights\":[");

      for (size_t i = 0; i < g_lightsLastFrame.size(); i++) {
        const LightRecord& r = g_lightsLastFrame[i];
        line += str::format(i ? "," : "", "{\"kind\":\"", r.label,
          "\",\"channels\":", r.channels,
          ",\"slice\":", r.slice,
          ",\"solved\":", r.solved ? 1 : 0,
          ",\"origin\":[", r.origin[0], ",", r.origin[1], ",", r.origin[2],
          "],\"dist\":", std::sqrt(r.origin[0] * r.origin[0] + r.origin[1] * r.origin[1] + r.origin[2] * r.origin[2]),
          ",\"radius\":", r.radius,
          ",\"c3\":[", r.c3[0], ",", r.c3[1], ",", r.c3[2], ",", r.c3[3],
          "],\"det\":", r.det, "}");
      }

      line += "]}\n";
      g_logFile << line;
      g_logFile.flush();
    }

  }


  namespace blessed_point_detail {
    extern const bool g_enabled = ComputePointTrace() || ComputeSunGate();
    extern const bool g_skip    = ComputePointTrace() && env::getEnvVar("BLESSED_POINT_SKIP_RASTER") == "1";
  }


  void BlessedPointShadow::EmitDispatch(D3D11ImmediateContext* ctx, BlessedPointShadowDispatchArgs&& args) {
    ctx->EmitCs([cArgs = std::move(args)] (DxvkContext* dxvkCtx) {
      dxvkCtx->blessedRunPointShadowPass(cArgs);
    });
  }


  bool BlessedPointShadow::ShouldSkipDrawSlow(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    const ShaderClass& cls = ClassifyPs(state);

    // the mask draw itself: trace in its place. If the trace could not go
    // out (unsolved light, no depth, ...) the raster draw stays.
    if (cls.kind == MaskKind::Spot || cls.kind == MaskKind::Paraboloid) {
      ShaderClass copy = cls;
      if (!HandlePointMaskDraw(ctx, state, copy)) {
        g_win.rasterKept++;
        return false;
      }
      g_win.skippedMasks++;
      return true;
    }

    if (g_mapsToSkip.empty())
      return false;

    // a point shadow-map draw: its depth target, or its colour target 0 (a
    // paraboloid map may store distance as colour), is a single-slice view
    // of a learned (image, slice)
    bool isMap = false;

    if (D3D11DepthStencilView* dsv = state.om.dsv.ptr())
      isMap = IsLearnedMap(dsv->GetViewInfo(), dsv->GetImageView());

    if (!isMap && state.om.maxRtv > 0) {
      if (D3D11RenderTargetView* rtv = state.om.rtvs[0].ptr())
        isMap = IsLearnedMap(rtv->GetViewInfo(), rtv->GetImageView());
    }

    if (!isMap)
      return false;

    g_win.skippedMaps++;
    return true;
  }


  void BlessedPointShadow::OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    if (!IsEnabled())
      return;

    const ShaderClass& cls = ClassifyPs(state);

    if (cls.kind == MaskKind::Sun) {
      g_win.sunDraws++;
      return;
    }

    if (cls.kind == MaskKind::None) {
      // blessed: BLESSED_POINT_DETECT=b2 -- a mask draw the hash table
      // doesn't know: a 448-byte ps b2, colour target 0 only, no depth target
      if (!GetConfig().detectB2 || state.om.dsv.ptr() || !state.ps.ptr())
        return;
      if (BoundCbSize(state, D3D11ShaderType::ePixel, 2) != 448u)
        return;
      D3D11RenderTargetView* rtv0 = state.om.maxRtv > 0 ? state.om.rtvs[0].ptr() : nullptr;
      if (!rtv0)
        return;
      D3D11_RENDER_TARGET_VIEW_DESC desc = { };
      rtv0->GetDesc(&desc);
      if (desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && desc.Format != DXGI_FORMAT_R8G8B8A8_TYPELESS)
        return;

      ShaderClass unknown = { MaskKind::Unknown448, "b2-448" };
      HandlePointMaskDraw(ctx, state, unknown);
      return;
    }

    ShaderClass copy = cls;
    HandlePointMaskDraw(ctx, state, copy);
  }


  void BlessedPointShadow::OnPresent() {
    if (!IsEnabled())
      return;

    if (!g_win.frames && !g_logTried)
      g_start = dxvk::high_resolution_clock::now();

    g_claimLastFrame = g_claimThisFrame;
    g_claimThisFrame = 0u;
    g_frameId++;

    g_mapsToSkip.swap(g_mapsLearning);
    g_mapsLearning.clear();

    g_lightsLastFrame.swap(g_lightsThisFrame);
    g_lightsThisFrame.clear();

    if (++g_win.frames >= 120u) {
      WriteWindow();
      g_win = Window();
    }
  }


  bool BlessedPointShadow::SunMayWriteR(const D3D11ContextState& state) {
    if (!IsEnabled() || !GetConfig().sunGate)
      return true;

    const char* reason = nullptr;
    uint32_t b2 = BoundCbSize(state, D3D11ShaderType::ePixel, 2);

    if (b2 != 400u)
      reason = "ps b2 is not the sun's 400 bytes";
    else if (!(WriteMask0(state) & 1u))
      reason = "the draw does not write channel r";
    else if ((g_claimThisFrame | g_claimLastFrame) & 1u)
      reason = "a point light owns channel r";

    int32_t verdict = reason ? 0 : 1;
    if (verdict != g_sunGateLast) {
      g_sunGateLast = verdict;
      if (reason)
        Logger::info(str::format("BlessedPointShadow: sun gate closed (", reason, ", b2=", b2, ")"));
      else
        Logger::info("BlessedPointShadow: sun gate open");
    }

    if (reason)
      g_win.sunGateClosed++;

    return reason == nullptr;
  }

}
