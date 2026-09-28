// blessed: BLESSED_VOLUMETRICS -- app-thread half: config, shader matching, cbuffer reads, dispatch handoff. See blessed_volumetrics.h.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "blessed_volumetrics.h"

#include "d3d11_context_imm.h"
#include "d3d11_texture.h"

#include "../dxvk/blessed/blessed_async.h" // blessed: async-compute
#include "../dxvk/blessed/blessed_volumetrics.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    enum class VolMode : uint8_t { Off, Vanilla, Fill, Rt };

    VolMode ParseMode(const std::string& s) {
      if (s == "vanilla") return VolMode::Vanilla;
      if (s == "fill")    return VolMode::Fill;
      if (s == "rt")      return VolMode::Rt;
      return VolMode::Off;
    }

    // "<vs|ps>:<slot>:<byte offset>", as BLESSED_SHADOW_* slots
    struct VolSlot {
      bool            valid  = false;
      D3D11ShaderType stage  = D3D11ShaderType::ePixel;
      uint32_t        slot   = 0;
      uint32_t        offset = 0;
    };

    VolSlot MakeSlot(uint32_t slot, uint32_t offset) {
      VolSlot s;
      s.valid  = true;
      s.slot   = slot;
      s.offset = offset;
      return s;
    }

    VolSlot ParseSlot(const std::string& s, const VolSlot& fallback) {
      if (s.empty())
        return fallback;
      if (s == "off")
        return VolSlot();

      size_t c1 = s.find(':');
      size_t c2 = c1 == std::string::npos ? std::string::npos : s.find(':', c1 + 1);
      if (c1 == std::string::npos || c2 == std::string::npos) {
        Logger::warn(str::format("blessed: volumetrics: bad cbuffer slot '", s, "'"));
        return fallback;
      }

      VolSlot out;
      out.valid  = true;
      out.stage  = s.substr(0, c1) == "vs" ? D3D11ShaderType::eVertex : D3D11ShaderType::ePixel;
      out.slot   = std::strtoul(s.substr(c1 + 1, c2 - c1 - 1).c_str(), nullptr, 10);
      out.offset = std::strtoul(s.substr(c2 + 1).c_str(), nullptr, 10);
      return out;
    }

    float EnvFloat(const char* name, float fallback) {
      std::string s = env::getEnvVar(name);
      return s.empty() ? fallback : std::strtof(s.c_str(), nullptr);
    }

    uint32_t EnvUint(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
      std::string s = env::getEnvVar(name);
      uint32_t v = s.empty() ? fallback : uint32_t(std::strtoul(s.c_str(), nullptr, 10));
      return std::min(std::max(v, lo), hi);
    }

    std::vector<std::string> SplitCsv(const std::string& csv) {
      std::vector<std::string> out;
      size_t pos = 0;
      while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        std::string tok = csv.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? csv.size() : comma + 1;
        if (!tok.empty())
          out.push_back(tok);
      }
      return out;
    }

    // a full "fs.<hash>" name, or a hex prefix (8+ chars) of what follows "fs."
    bool MatchesAny(const std::string& name, const std::vector<std::string>& tokens) {
      for (const auto& tok : tokens) {
        if (tok.rfind("fs.", 0) == 0) {
          if (name == tok)
            return true;
        } else if (tok.size() >= 8 && name.rfind("fs.", 0) == 0 && name.compare(3, tok.size(), tok) == 0) {
          return true;
        }
      }
      return false;
    }

    struct VolConfig {
      VolMode                  mode = VolMode::Off;
      std::vector<std::string> applyPs;
      std::vector<std::string> sunPs;
      VolSlot                  sunSlot;
      bool                     sunFromProj  = true;
      bool                     camFromMask  = true;
      VolSlot                  invVp;
      VolSlot                  camPos;
      VolSlot                  intensity;
      uint32_t                 depthSrvSlot = 0;
      float                    farValue     = 1.0f;
      float                    fill[4]      = { 0.0f, 0.0f, 0.0f, 0.0f };
      bool                     skipCascades = false;
      uint64_t                 dumpFrame    = 0;   // 0 = no dump
      float                    dumpScale    = 128.0f;
      std::string              debugName;
      BlessedVolDispatchArgs   tuning;       // tuning fields only
      // blessed: async-compute -- the kick draw and its depth srv
      std::vector<std::string> kickPs;
      bool                     anyDsv       = false; // blessed: vol-2
      uint32_t                 kickDepthSrv = 2u;

      static const VolConfig& Get() {
        static VolConfig s_config = [] {
          VolConfig c;
          std::string modeStr = env::getEnvVar("BLESSED_VOLUMETRICS");
          c.mode = ParseMode(modeStr);
          if (c.mode == VolMode::Off)
            return c;

          std::string applyStr = env::getEnvVar("BLESSED_VOL_PS");
          c.applyPs = SplitCsv(applyStr.empty() ? std::string("c480e36e") : applyStr);
          std::string sunPsStr = env::getEnvVar("BLESSED_VOL_SUN_PS");
          c.sunPs = SplitCsv(sunPsStr.empty() ? std::string("b070feb5") : sunPsStr);

          // blessed: the sun is the shadow projection's depth axis at the
          // mask draw (ps b2 c18), negated -- PROJECT.md's lesson; ps b2 c0
          // there is a per-frame noise rotation, not the sun
          std::string sunStr = env::getEnvVar("BLESSED_VOL_SUNDIR");
          if (sunStr.empty())
            sunStr = "proj:ps:2:288";
          c.sunFromProj = sunStr.rfind("proj:", 0) == 0;
          c.sunSlot = ParseSlot(c.sunFromProj ? sunStr.substr(5) : sunStr, VolSlot());

          c.camFromMask = env::getEnvVar("BLESSED_VOL_CAM") != "self";
          c.invVp  = ParseSlot(env::getEnvVar("BLESSED_VOL_INVVP"),  MakeSlot(12, 512));
          c.camPos = ParseSlot(env::getEnvVar("BLESSED_VOL_CAMPOS"), MakeSlot(12, 640));
          // pass 138's g_IntensityX_TemporalY.x (ps b2 c0.x)
          c.intensity = ParseSlot(env::getEnvVar("BLESSED_VOL_INTENSITY"), MakeSlot(2, 0));

          std::string depthStr = env::getEnvVar("BLESSED_VOL_DEPTH");
          if (depthStr.rfind("srv:", 0) == 0)
            c.depthSrvSlot = std::strtoul(depthStr.c_str() + 4, nullptr, 10);

          c.farValue = env::getEnvVar("BLESSED_VOL_FAR") == "0" ? 0.0f : 1.0f;

          float fill = EnvFloat("BLESSED_VOL_FILL", 0.0f);
          for (float& f : c.fill)
            f = fill;

          c.skipCascades = env::getEnvVar("BLESSED_SKIP_CASCADES") == "1";

          std::string dumpStr = env::getEnvVar("BLESSED_VOL_DUMP");
          if (!dumpStr.empty() && dumpStr != "0") {
            uint64_t n = std::strtoull(dumpStr.c_str(), nullptr, 10);
            c.dumpFrame = n > 1 ? n : 300;
          }
          c.dumpScale = EnvFloat("BLESSED_VOL_DUMP_SCALE", 128.0f);

          BlessedVolDispatchArgs& t = c.tuning;
          // blessed: vol-2 -- BLESSED_VOL_GAIN=auto matches our mean to
          // vanilla's pass 138 output (see blessed_volumetrics.h)
          std::string gainStr = env::getEnvVar("BLESSED_VOL_GAIN");
          t.autoGain      = gainStr == "auto";
          t.gain          = t.autoGain ? t.gain : EnvFloat("BLESSED_VOL_GAIN", t.gain);
          t.maxOut        = EnvFloat("BLESSED_VOL_MAX", t.maxOut);
          t.statsEvery    = EnvUint("BLESSED_VOL_STATS_EVERY", t.autoGain ? 4u : 30u, 0u, 100000u);
          t.phaseG        = std::clamp(EnvFloat("BLESSED_VOL_G", t.phaseG), -0.95f, 0.95f);
          t.density       = EnvFloat("BLESSED_VOL_DENSITY", t.density);
          t.heightFalloff = EnvFloat("BLESSED_VOL_HEIGHT_FALLOFF", t.heightFalloff);
          t.heightBase    = EnvFloat("BLESSED_VOL_HEIGHT_BASE", t.heightBase);
          t.heightFloor   = std::clamp(EnvFloat("BLESSED_VOL_HEIGHT_FLOOR", t.heightFloor), 0.0f, 1.0f);
          t.range         = EnvFloat("BLESSED_VOL_RANGE", t.range);
          t.sunRayLength  = EnvFloat("BLESSED_VOL_SUN_RAY", t.sunRayLength);
          t.historyWeight = std::clamp(EnvFloat("BLESSED_VOL_HISTORY", t.historyWeight), 0.0f, 0.98f);
          t.rejectRel     = EnvFloat("BLESSED_VOL_REJECT", t.rejectRel);
          t.steps         = EnvUint("BLESSED_VOL_STEPS", t.steps, 1u, 64u);
          t.resDivisor    = EnvUint("BLESSED_VOL_RES", t.resDivisor, 1u, 8u);

          // blessed: async-compute -- the sao composite (pass 127) reads the
          // final depth at ps srv 2, the same draw and slot rtao uses
          std::string kickStr = env::getEnvVar("BLESSED_ASYNC_VOL_KICK_PS");
          c.kickPs = SplitCsv(kickStr.empty() ? std::string("ddcce9bc") : kickStr);
          c.kickDepthSrv = EnvUint("BLESSED_ASYNC_VOL_DEPTH_SRV", c.kickDepthSrv, 0u, 127u);

          // blessed: vol-2 -- the three draws we watch bind no depth target
          // (whiterun dump: 57 of 3,858 draws); BLESSED_VOL_ANY_DSV=1 looks
          // at every draw instead
          c.anyDsv = env::getEnvVar("BLESSED_VOL_ANY_DSV") == "1";

          c.debugName = env::getEnvVar("BLESSED_VOL_DEBUG");
          const std::string& d = c.debugName;
          t.debugMode = d == "dist" ? 1u : d == "vis" ? 2u : d == "raw" ? 3u : d == "hist" ? 4u : d == "density" ? 5u : 0u;

          Logger::info(str::format("blessed: volumetrics: mode=", modeStr,
            " debug=", d.empty() ? std::string("off") : d,
            " gain=", t.autoGain ? std::string("auto") : str::format(t.gain), " max=", t.maxOut, " g=", t.phaseG, " density=", t.density,
            " falloff=", t.heightFalloff, " steps=", t.steps, " res=1/", t.resDivisor,
            " cam=", c.camFromMask ? "mask" : "self",
            " dump=", c.dumpFrame));
          return c;
        }();

        return s_config;
      }
    };

    bool ReadCbufferFloats(const D3D11ContextState& state, const VolSlot& slot, float* out, uint32_t count) {
      if (!slot.valid)
        return false;

      const auto& cbvStage = state.cbv[slot.stage];
      if (slot.slot >= cbvStage.maxCount)
        return false;

      D3D11Buffer* buffer = cbvStage.buffers[slot.slot].buffer.ptr();
      if (!buffer)
        return false;

      void* mapPtr = buffer->GetMapPtr();
      if (!mapPtr)
        return false;

      UINT byteWidth  = buffer->Desc()->ByteWidth;
      UINT byteOffset = cbvStage.buffers[slot.slot].constantOffset * 16u + slot.offset;
      if (byteOffset + count * sizeof(float) > byteWidth)
        return false;

      std::memcpy(out, reinterpret_cast<const uint8_t*>(mapPtr) + byteOffset, count * sizeof(float));
      return true;
    }

    enum class Match : uint8_t { None, Apply, Sun, Kick }; // blessed: async-compute, Kick

    // what the mask draw saw, consumed by the next pass-138 match
    struct MaskCapture {
      bool  sunValid = false;
      bool  camValid = false;
      float sunDir[3] = { };
      float invViewProj[16] = { };
      float camPos[3] = { };
    };

    struct VolState {
      std::unordered_map<const void*, Match> matchCache;
      MaskCapture pending;
      bool        pendingFresh = false;
      bool        haveLastSun  = false;
      float       lastSun[3]   = { };
      uint64_t    applyCount   = 0;
      bool        warnedStorage = false;
      bool        warnedCamera  = false;
      bool        warnedSun     = false;
      // blessed: vol-2 -- true while OnDraw runs for a pass-138 draw the
      // skip-replaced switch dropped: nothing wrote the target this frame
      bool        vanillaSkipped = false;

      // blessed: async-compute. Pass 138's depth image and target extent,
      // remembered for the next frame's kick (identity by cookie, never
      // owned). A kick is valid for pass 138 only in the frame it was made
      // in and only if no draw wrote that depth in between.
      uint64_t    lastDepthCookie = 0;
      uint32_t    lastWidth       = 0;
      uint32_t    lastHeight      = 0;
      uint64_t    frame           = 0;
      bool        kickPending     = false;
      bool        kickInvalid     = false;
      uint64_t    kickFrame       = 0;
      uint64_t    kickDepthCookie = 0;
      uint32_t    kickWidth       = 0;
      uint32_t    kickHeight      = 0;
      uint64_t    kicks           = 0;
      uint64_t    kicksUsed       = 0;
    };

    VolState g_vol;

    bool ComputeEnabled() {
      return ParseMode(env::getEnvVar("BLESSED_VOLUMETRICS")) != VolMode::Off;
    }

    Rc<DxvkImageView> MakeView(DxvkImage* image, VkImageUsageFlagBits usage, VkImageAspectFlags aspects) {
      DxvkImageViewKey key = { };
      key.viewType   = VK_IMAGE_VIEW_TYPE_2D;
      key.usage      = usage;
      key.format     = image->info().format;
      key.layout     = VK_IMAGE_LAYOUT_GENERAL;
      key.aspects    = aspects;
      key.mipIndex   = 0u;
      key.mipCount   = 1u;
      key.layerIndex = 0u;
      key.layerCount = 1u;
      return image->createView(key);
    }

    void OnSunDraw(const VolConfig& cfg, const D3D11ContextState& state) {
      // first mask draw after the last pass-138 match wins (as the shadow
      // hook's BLESSED_HOOK_ONCE)
      if (g_vol.pendingFresh)
        return;

      MaskCapture cap;
      float live[3];
      if (ReadCbufferFloats(state, cfg.sunSlot, live, 3)) {
        float len = std::sqrt(live[0]*live[0] + live[1]*live[1] + live[2]*live[2]);
        if (len > 1.0e-6f) {
          float sign = cfg.sunFromProj ? -1.0f : 1.0f;
          for (uint32_t i = 0; i < 3; i++)
            cap.sunDir[i] = sign * live[i] / len;
          cap.sunValid = true;
        }
      }

      if (cfg.camFromMask) {
        cap.camValid = ReadCbufferFloats(state, cfg.invVp, cap.invViewProj, 16)
                    && ReadCbufferFloats(state, cfg.camPos, cap.camPos, 3);
      }

      g_vol.pending = cap;
      g_vol.pendingFresh = true;
    }

    void OnApplyDraw(D3D11ImmediateContext* ctx, const VolConfig& cfg, const D3D11ContextState& state) {
      auto appT0 = std::chrono::steady_clock::now(); // blessed: vol-2, app_us
      MaskCapture cap = g_vol.pendingFresh ? g_vol.pending : MaskCapture();
      g_vol.pendingFresh = false;

      if (cap.sunValid) {
        std::memcpy(g_vol.lastSun, cap.sunDir, sizeof(g_vol.lastSun));
        g_vol.haveLastSun = true;
      }

      g_vol.applyCount++;
      bool dumpNow = cfg.dumpFrame != 0 && g_vol.applyCount == cfg.dumpFrame;

      D3D11RenderTargetView* rtv0 = state.om.maxRtv > 0 ? state.om.rtvs[0].ptr() : nullptr;
      if (!rtv0)
        return;

      Rc<DxvkImageView> rtvView = rtv0->GetImageView();
      DxvkImage* outImage = rtvView->image();

      BlessedVolDispatchArgs args = cfg.tuning;
      args.width  = outImage->info().extent.width;
      args.height = outImage->info().extent.height;
      args.farDepthValue = cfg.farValue;
      args.vanillaSkipped = g_vol.vanillaSkipped;

      // blessed: vol-2 -- vanilla's draw was skipped and we cannot trace:
      // clear the target so the blur and pass 167 never see last frame's
      static const FLOAT s_zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
      auto bail = [&] {
        if (g_vol.vanillaSkipped)
          ctx->ClearRenderTargetView(rtv0, s_zero);
      };

      if (cfg.mode == VolMode::Fill)
        ctx->ClearRenderTargetView(rtv0, cfg.fill);

      if (dumpNow) {
        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
        if (dir.empty()) {
          Logger::warn("blessed: volumetrics: BLESSED_VOL_DUMP needs BLESSED_PROBE_DIR");
        } else {
          const char* modeName = cfg.mode == VolMode::Vanilla ? "vanilla" : cfg.mode == VolMode::Fill ? "fill" : "rt";
          args.dumpTag = str::format(modeName,
            cfg.debugName.empty() ? std::string() : std::string("-") + cfg.debugName,
            cfg.skipCascades ? "-skip" : "", "-", g_vol.applyCount);
          args.dumpPath  = dir + env::PlatformDirSlash + "vol-" + args.dumpTag + ".pgm";
          args.dumpScale = cfg.dumpScale;
          args.dump      = true;
          args.outputSampledView = MakeView(outImage, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        }
      }

      float intensity[1];
      if (ReadCbufferFloats(state, cfg.intensity, intensity, 1) && std::isfinite(intensity[0]))
        args.gameIntensity = intensity[0];

      if (cfg.mode != VolMode::Rt) {
        if (args.dump) {
          args.op = BlessedVolDispatchArgs::Op::DumpOnly;
          BlessedVolumetrics::Emit(ctx, std::move(args));
        }
        return;
      }

      // rt: storage on the target, depth, camera, sun
      if (!(outImage->info().usage & VK_IMAGE_USAGE_STORAGE_BIT) || outImage->info().format != VK_FORMAT_R16_SFLOAT) {
        if (!g_vol.warnedStorage) {
          g_vol.warnedStorage = true;
          Logger::warn(str::format("blessed: volumetrics: pass 138's target is not a storage r16f image (format ",
            uint32_t(outImage->info().format), "); leaving vanilla"));
        }
        bail();
        return;
      }

      const auto& srvStage = state.srv[D3D11ShaderType::ePixel];
      D3D11ShaderResourceView* depthSrv = cfg.depthSrvSlot < srvStage.maxCount
        ? srvStage.views[cfg.depthSrvSlot].ptr() : nullptr;
      if (!depthSrv) {
        bail();
        return;
      }

      DxvkImage* depthImage = depthSrv->GetImageView()->image();
      bool isDepth = lookupFormatInfo(depthImage->info().format)->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT;
      args.depthView = MakeView(depthImage, VK_IMAGE_USAGE_SAMPLED_BIT,
        isDepth ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT) : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT));
      args.outputStorageView = MakeView(outImage, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

      bool camOk = false;
      if (cap.camValid) {
        std::memcpy(args.invViewProj, cap.invViewProj, sizeof(args.invViewProj));
        std::memcpy(args.camPosNow, cap.camPos, sizeof(args.camPosNow));
        camOk = true;
      } else {
        // BLESSED_VOL_CAM=self, or no mask draw this frame: read pass 138's
        // own b12 (the same per-frame buffer, remapped just before it)
        camOk = ReadCbufferFloats(state, cfg.invVp, args.invViewProj, 16)
             && ReadCbufferFloats(state, cfg.camPos, args.camPosNow, 3);
      }

      if (!camOk) {
        if (!g_vol.warnedCamera) {
          g_vol.warnedCamera = true;
          Logger::warn("blessed: volumetrics: no camera matrix this frame; leaving vanilla");
        }
        bail();
        return;
      }

      if (!g_vol.haveLastSun) {
        if (!g_vol.warnedSun) {
          g_vol.warnedSun = true;
          Logger::warn("blessed: volumetrics: no sun direction yet (mask draw not seen); leaving vanilla");
        }
        bail();
        return;
      }
      std::memcpy(args.sunDir, g_vol.lastSun, sizeof(args.sunDir));

      args.op = BlessedVolDispatchArgs::Op::Trace;

      // blessed: async-compute -- this frame's kick already traced; only the
      // upsample is left. The cs side falls back to Trace on its own if the
      // kick never reached the async queue.
      if (unlikely(BlessedVolumetrics::IsAsyncKickEnabled())) {
        bool useKick = g_vol.kickPending && !g_vol.kickInvalid
          && g_vol.kickFrame == g_vol.frame
          && g_vol.kickDepthCookie == depthImage->cookie()
          && g_vol.kickWidth == args.width && g_vol.kickHeight == args.height;

        if (useKick) {
          args.op = BlessedVolDispatchArgs::Op::AsyncUpsample;
          g_vol.kicksUsed++;
        }

        g_vol.kickPending     = false;
        g_vol.lastDepthCookie = isDepth ? depthImage->cookie() : 0;
        g_vol.lastWidth       = args.width;
        g_vol.lastHeight      = args.height;
      }

      args.appNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - appT0).count());
      BlessedVolumetrics::Emit(ctx, std::move(args));
    }

    // blessed: async-compute -- the kick draw: depth is final here (117 is
    // the frame's last depth write), pass 138 is ~ten passes away
    void OnKickDraw(D3D11ImmediateContext* ctx, const VolConfig& cfg, const D3D11ContextState& state) {
      g_vol.kickPending = false;

      if (cfg.mode != VolMode::Rt || !g_vol.lastDepthCookie)
        return;

      const auto& srvStage = state.srv[D3D11ShaderType::ePixel];
      D3D11ShaderResourceView* depthSrv = cfg.kickDepthSrv < srvStage.maxCount
        ? srvStage.views[cfg.kickDepthSrv].ptr() : nullptr;
      if (!depthSrv)
        return;

      // the same image pass 138 read last frame, and a depth one
      DxvkImage* depthImage = depthSrv->GetImageView()->image();
      if (depthImage->cookie() != g_vol.lastDepthCookie)
        return;

      MaskCapture cap = g_vol.pendingFresh ? g_vol.pending : MaskCapture();

      BlessedVolDispatchArgs args = cfg.tuning;
      args.width  = g_vol.lastWidth;
      args.height = g_vol.lastHeight;
      args.farDepthValue = cfg.farValue;

      if (cap.camValid) {
        std::memcpy(args.invViewProj, cap.invViewProj, sizeof(args.invViewProj));
        std::memcpy(args.camPosNow, cap.camPos, sizeof(args.camPosNow));
      } else if (!ReadCbufferFloats(state, cfg.invVp, args.invViewProj, 16)
              || !ReadCbufferFloats(state, cfg.camPos, args.camPosNow, 3)) {
        return;
      }

      if (cap.sunValid)
        std::memcpy(args.sunDir, cap.sunDir, sizeof(args.sunDir));
      else if (g_vol.haveLastSun)
        std::memcpy(args.sunDir, g_vol.lastSun, sizeof(args.sunDir));
      else
        return;

      args.depthView = MakeView(depthImage, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      args.op = BlessedVolDispatchArgs::Op::AsyncTrace;

      g_vol.kickPending     = true;
      g_vol.kickInvalid     = false;
      g_vol.kickFrame       = g_vol.frame;
      g_vol.kickDepthCookie = g_vol.lastDepthCookie;
      g_vol.kickWidth       = args.width;
      g_vol.kickHeight      = args.height;
      g_vol.kicks++;

      BlessedVolumetrics::Emit(ctx, std::move(args));
    }

    // blessed: async-compute -- a draw that writes the kicked depth between
    // the kick and pass 138 makes the kick's trace stale
    void CheckKickDepthWrite(const D3D11ContextState& state) {
      D3D11DepthStencilView* dsv = state.om.dsv.ptr();
      if (!dsv || !(dsv->GetWritableAspectMask() & VK_IMAGE_ASPECT_DEPTH_BIT))
        return;

      D3D11DepthStencilState* ds = state.om.dsState.ptr();
      if (ds && (!ds->Desc().DepthEnable || ds->Desc().DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ALL))
        return;

      if (dsv->GetImageView()->image()->cookie() == g_vol.kickDepthCookie)
        g_vol.kickInvalid = true;
    }

  }


  namespace blessed_vol_detail {
    extern const bool g_enabled = ComputeEnabled();

    // blessed: async-compute
    extern const bool g_asyncKick = ParseMode(env::getEnvVar("BLESSED_VOLUMETRICS")) == VolMode::Rt
      && BlessedAsync::IsRequested() && BlessedAsync::PassRequested(BlessedAsyncPass::Vol);
  }


  void BlessedVolumetrics::Emit(D3D11ImmediateContext* ctx, BlessedVolDispatchArgs&& args) {
    ctx->EmitCs([args = std::move(args)] (DxvkContext* dxvkCtx) {
      dxvkCtx->blessedRunVolumetricsPass(args);
    });
  }


  // blessed: async-compute -- one cache for the post-draw and the kick hook
  static Match ClassifyPs(D3D11PixelShader* ps, const VolConfig& cfg) {
    auto it = g_vol.matchCache.find(ps);
    if (it != g_vol.matchCache.end())
      return it->second;

    const std::string& name = ps->GetCommonShader()->GetName();
    Match match = MatchesAny(name, cfg.applyPs) ? Match::Apply
                : MatchesAny(name, cfg.sunPs)   ? Match::Sun
                : (BlessedVolumetrics::IsAsyncKickEnabled() && MatchesAny(name, cfg.kickPs)) ? Match::Kick
                : Match::None;
    g_vol.matchCache.emplace(ps, match);
    return match;
  }


  void BlessedVolumetrics::OnDrawPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps)
      return;

    const VolConfig& cfg = VolConfig::Get();

    if (state.om.dsv != nullptr && !cfg.anyDsv) // blessed: vol-2
      return;

    if (ClassifyPs(ps, cfg) == Match::Kick)
      OnKickDraw(ctx, cfg, state);
  }


  void BlessedVolumetrics::OnPresent() {
    g_vol.frame++;
    g_vol.kickPending = false;

    // one line per ~10 s at 120 fps: how many kicks pass 138 actually used
    if ((g_vol.frame % 1200u) == 0u) {
      Logger::info(str::format("blessed: async: vol kicks=", g_vol.kicks,
        " used=", g_vol.kicksUsed, " (of ", g_vol.frame, " frames)"));
    }
  }


  void BlessedVolumetrics::OnSkippedDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    if (!IsEnabled())
      return;

    g_vol.vanillaSkipped = true;
    OnDraw(ctx, state);
    g_vol.vanillaSkipped = false;
  }


  void BlessedVolumetrics::OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    if (!IsEnabled())
      return;

    // blessed: async-compute -- only between a kick and pass 138
    if (unlikely(g_vol.kickPending) && !g_vol.kickInvalid)
      CheckKickDepthWrite(state);

    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps)
      return;

    const VolConfig& cfg = VolConfig::Get();

    // blessed: vol-2 -- skip the match-cache lookup for depth-target draws
    if (state.om.dsv != nullptr && !cfg.anyDsv)
      return;

    Match match = ClassifyPs(ps, cfg);

    if (match == Match::Sun)
      OnSunDraw(cfg, state);
    else if (match == Match::Apply)
      OnApplyDraw(ctx, cfg, state);
  }

}
