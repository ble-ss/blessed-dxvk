// blessed: app-thread half of the ray-traced sun shadow pass -- see blessed_shadow.h
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "blessed_shadow.h"
#include "blessed_cascades.h"
#include "blessed_point_shadow.h" // blessed: point-lights, the interior sun gate
#include "blessed_scene_capture.h"

#include "d3d11_context_imm.h"
#include "d3d11_texture.h"

#include "../dxvk/dxvk_memory.h"
#include "../dxvk/blessed/blessed_shadow.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // "<vs|ps>:<slot>:<byte offset>", e.g. "ps:12:512"
    struct BlessedShadowSlot {
      bool             valid  = false;
      D3D11ShaderType  stage  = D3D11ShaderType::ePixel;
      uint32_t         slot   = 0;
      uint32_t         offset = 0;
    };

    BlessedShadowSlot ParseSlot(const std::string& s, const BlessedShadowSlot& fallback) {
      if (s.empty())
        return fallback;

      size_t firstColon = s.find(':');
      size_t secondColon = firstColon == std::string::npos ? std::string::npos : s.find(':', firstColon + 1);
      if (firstColon == std::string::npos || secondColon == std::string::npos)
        return fallback;

      std::string stageStr = s.substr(0, firstColon);
      BlessedShadowSlot out;
      out.stage  = stageStr == "vs" ? D3D11ShaderType::eVertex : D3D11ShaderType::ePixel;
      out.slot   = std::strtoul(s.substr(firstColon + 1, secondColon - firstColon - 1).c_str(), nullptr, 10);
      out.offset = std::strtoul(s.substr(secondColon + 1).c_str(), nullptr, 10);
      out.valid  = true;
      return out;
    }

    bool ParseFloats(const std::string& s, float* out, uint32_t count) {
      size_t pos = 0;
      for (uint32_t i = 0; i < count; i++) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (tok.empty())
          return false;
        out[i] = std::strtof(tok.c_str(), nullptr);
        if (comma == std::string::npos) {
          if (i + 1 != count)
            return false;
        } else {
          pos = comma + 1;
        }
      }
      return true;
    }

    struct BlessedShadowConfig {
      bool               enabled       = false;
      BlessedShadowSlot  invVp;
      BlessedShadowSlot  camPos;
      float              sunDir[3]     = { 0.3f, 0.4f, 0.866f };
      BlessedShadowSlot  sunSlot;       // blessed: live sun from a cbuffer
      bool               sunFromProj   = false; // blessed: slot points at the shadow projection's depth axis
      float              farValue      = 1.0f;
      bool               depthFromSrv  = false;
      uint32_t           depthSrvSlot  = 0;
      bool               dumpRequested = false;
      uint64_t           dumpFrame     = 120;   // blessed: BLESSED_SHADOW_DUMP=<frame> (1 = 120)

      // blessed: BLESSED_SHADOW_SOFT=1 -- soft, temporally-accumulated
      // shadows (see blessed_soft_shadow.h on the dxvk side)
      bool               softEnabled     = false;
      float              sunHalfAngleRad = 0.53f * (3.14159265358979f / 180.0f); // BLESSED_SHADOW_SUN_DEG
      uint32_t           spp             = 1;    // BLESSED_SHADOW_SPP, clamped [1,4]
      BlessedShadowSlot  prevVp;                 // CameraPreviousViewProjUnjittered

      static const BlessedShadowConfig& Get() {
        static BlessedShadowConfig s_config = [] {
          BlessedShadowConfig c;
          c.enabled = env::getEnvVar("BLESSED_HOOK_MODE") == "rtshadow";
          if (!c.enabled)
            return c;

          BlessedShadowSlot invVpDefault;
          invVpDefault.valid  = true;
          invVpDefault.stage  = D3D11ShaderType::ePixel;
          invVpDefault.slot   = 12;
          invVpDefault.offset = 512;
          c.invVp = ParseSlot(env::getEnvVar("BLESSED_SHADOW_INVVP"), invVpDefault);

          BlessedShadowSlot camPosDefault;
          camPosDefault.valid  = true;
          camPosDefault.stage  = D3D11ShaderType::ePixel;
          camPosDefault.slot   = 12;
          camPosDefault.offset = 640;
          c.camPos = ParseSlot(env::getEnvVar("BLESSED_SHADOW_CAMPOS"), camPosDefault);

          std::string sunDirStr = env::getEnvVar("BLESSED_SHADOW_SUNDIR");
          if (sunDirStr.rfind("proj:", 0) == 0) {
            // the mask shader's ShadowMapProj[0] (ps b2 c16-c18, column-major
            // float4x3): c18 is the light-space depth axis, so the direction
            // toward the sun is -normalize(c18.xyz). (ps b2 c0 is not the sun:
            // it changes every frame, likely the filter's noise rotation.)
            c.sunSlot = ParseSlot(sunDirStr.substr(5), BlessedShadowSlot());
            c.sunFromProj = true;
          } else if (sunDirStr.rfind("ps:", 0) == 0 || sunDirStr.rfind("vs:", 0) == 0) {
            // the shadow-mask pass's ps b2 c0 holds skyrim's own sun
            // direction (docs/research/whiterun-frame.md)
            c.sunSlot = ParseSlot(sunDirStr, BlessedShadowSlot());
          } else if (!sunDirStr.empty()) {
            float parsed[3];
            if (ParseFloats(sunDirStr, parsed, 3)) {
              float len = std::sqrt(parsed[0]*parsed[0] + parsed[1]*parsed[1] + parsed[2]*parsed[2]);
              if (len > 1.0e-6f) {
                c.sunDir[0] = parsed[0] / len;
                c.sunDir[1] = parsed[1] / len;
                c.sunDir[2] = parsed[2] / len;
              }
            } else {
              Logger::warn(str::format("blessed: shadow pass: could not parse BLESSED_SHADOW_SUNDIR '", sunDirStr, "'"));
            }
          }

          c.farValue = env::getEnvVar("BLESSED_SHADOW_FAR") == "0" ? 0.0f : 1.0f;

          std::string depthStr = env::getEnvVar("BLESSED_SHADOW_DEPTH");
          if (depthStr.rfind("srv:", 0) == 0) {
            c.depthFromSrv = true;
            c.depthSrvSlot = std::strtoul(depthStr.c_str() + 4, nullptr, 10);
          }

          std::string dumpStr = env::getEnvVar("BLESSED_SHADOW_DUMP");
          c.dumpRequested = !dumpStr.empty() && dumpStr != "0";
          uint64_t n = std::strtoull(dumpStr.c_str(), nullptr, 10);
          c.dumpFrame = n > 1 ? n : 120;

          c.softEnabled = env::getEnvVar("BLESSED_SHADOW_SOFT") == "1";
          if (c.softEnabled) {
            std::string sunDegStr = env::getEnvVar("BLESSED_SHADOW_SUN_DEG");
            float sunDeg = sunDegStr.empty() ? 0.53f : std::strtof(sunDegStr.c_str(), nullptr);
            c.sunHalfAngleRad = sunDeg * (3.14159265358979f / 180.0f);

            std::string sppStr = env::getEnvVar("BLESSED_SHADOW_SPP");
            uint32_t spp = sppStr.empty() ? 1u : std::strtoul(sppStr.c_str(), nullptr, 10);
            c.spp = std::min<uint32_t>(std::max<uint32_t>(spp, 1u), 4u);

            BlessedShadowSlot prevVpDefault;
            prevVpDefault.valid  = true;
            prevVpDefault.stage  = D3D11ShaderType::ePixel;
            prevVpDefault.slot   = 12;
            prevVpDefault.offset = 256;
            c.prevVp = ParseSlot(env::getEnvVar("BLESSED_SHADOW_PREVVP"), prevVpDefault);

            Logger::info(str::format("BlessedShadow: soft shadows enabled (spp=", c.spp, " sun_deg=", sunDeg, ")"));
          }

          Logger::info("BlessedShadow: enabled (BLESSED_HOOK_MODE=rtshadow)");
          return c;
        }();

        return s_config;
      }
    };

    // blessed: reads `count` floats out of a bound cbuffer at `slot.offset`
    // bytes past its bound start (constantOffset, in 16-byte units, matches
    // BlessedDump's convention). Returns false if the cbuffer isn't bound,
    // isn't mapped (not a dynamic/CPU-writable buffer), or is too small.
    bool ReadCbufferFloats(
      const D3D11ContextState&  state,
            const BlessedShadowSlot& slot,
            float*              out,
            uint32_t            count) {
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

    // blessed: BLESSED_SHADOW_DUMP fires once, on the 120th matched draw
    // after the hook goes active -- see the shadow-pass brief.
    uint64_t g_frameCounter = 0;
    bool     g_dumpDone     = false;

    // blessed: the callback is installed by BlessedHook itself (mode rtshadow)

  }


  bool BlessedShadow::IsEnabled() {
    return BlessedShadowConfig::Get().enabled;
  }


  void BlessedShadow::OnDraw(D3D11ImmediateContext* ctx, const BlessedHookTargets& targets) {
    if (!IsEnabled())
      return;

    // blessed: BLESSED_SKIP_CASCADES -- this draw IS the mask draw (the
    // hook only calls us on a ps match), so this is exactly the moment to
    // (re)learn which images the cascades rendered into this frame. Runs
    // before the early-outs below: cheap no-op unless BLESSED_SKIP_CASCADES
    // is set, and learning shouldn't depend on this pass's own output being
    // storage-capable. See blessed_cascades.h.
    if (targets.state)
      BlessedCascadeSkip::LearnFromMaskDraw(*targets.state);

    const BlessedShadowConfig& cfg = BlessedShadowConfig::Get();

    if (!targets.hasRtv0 || !targets.rtv0View || !targets.state)
      return;

    // blessed: point-lights -- indoors channel r may be a point light's; only
    // trace the sun when this really is the sun's mask draw (blessed_point_shadow.h)
    if (!BlessedPointShadow::SunMayWriteR(*targets.state))
      return;

    DxvkImage* outputImage = targets.rtv0View->image();
    bool outputStorageCapable = (outputImage->info().usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;

    if (!outputStorageCapable) {
      // blessed: d3d11_texture.cpp only adds STORAGE_BIT when this same env
      // var is set, so this should be rare -- most likely a format that
      // doesn't support storage images at all. Logged once via the hook's
      // own hook.jsonl match counter; nothing more to do here.
      return;
    }

    // blessed: resolve the depth source -- bound dsv by default, or a ps
    // srv slot when BLESSED_SHADOW_DEPTH=srv:<slot> overrides it
    Rc<DxvkImageView> depthSrc;
    if (cfg.depthFromSrv) {
      const auto& srvStage = targets.state->srv[D3D11ShaderType::ePixel];
      if (cfg.depthSrvSlot < srvStage.maxCount) {
        if (D3D11ShaderResourceView* srv = srvStage.views[cfg.depthSrvSlot].ptr())
          depthSrc = srv->GetImageView();
      }
    } else {
      depthSrc = targets.dsvView;
    }

    if (!depthSrc)
      return;

    DxvkImage* depthImage = depthSrc->image();

    // blessed: fresh sampled/storage views every call -- DxvkImage::createView
    // is expected to memoize by key the way every other blessed meta-pass
    // relies on (see dxvk_meta_mipgen.cpp's own per-call createView use)
    DxvkImageViewKey depthViewInfo = { };
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.usage    = VK_IMAGE_USAGE_SAMPLED_BIT;
    depthViewInfo.format   = depthImage->info().format;
    depthViewInfo.layout   = VK_IMAGE_LAYOUT_GENERAL;
    depthViewInfo.aspects  = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthViewInfo.mipIndex = 0u;
    depthViewInfo.mipCount = 1u;
    depthViewInfo.layerIndex = 0u;
    depthViewInfo.layerCount = 1u;

    Rc<DxvkImageView> depthView = depthImage->createView(depthViewInfo);

    DxvkImageViewKey outputViewInfo = { };
    outputViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    outputViewInfo.usage    = VK_IMAGE_USAGE_STORAGE_BIT;
    outputViewInfo.format   = outputImage->info().format;
    outputViewInfo.layout   = VK_IMAGE_LAYOUT_GENERAL;
    outputViewInfo.aspects  = VK_IMAGE_ASPECT_COLOR_BIT;
    outputViewInfo.mipIndex = 0u;
    outputViewInfo.mipCount = 1u;
    outputViewInfo.layerIndex = 0u;
    outputViewInfo.layerCount = 1u;

    Rc<DxvkImageView> outputView = outputImage->createView(outputViewInfo);

    BlessedShadowDispatchArgs args;
    args.depthView            = depthView;
    args.outputView           = outputView;
    args.outputStorageCapable = true;
    args.width                = targets.rtv0.w;
    args.height               = targets.rtv0.h;
    args.sunDir[0] = cfg.sunDir[0];
    args.sunDir[1] = cfg.sunDir[1];
    args.sunDir[2] = cfg.sunDir[2];
    args.farDepthValue = cfg.farValue;

    if (cfg.sunSlot.valid) {
      float live[3];
      if (ReadCbufferFloats(*targets.state, cfg.sunSlot, live, 3)) {
        float len = std::sqrt(live[0]*live[0] + live[1]*live[1] + live[2]*live[2]);
        if (len > 1.0e-6f) {
          float sign = cfg.sunFromProj ? -1.0f : 1.0f;
          args.sunDir[0] = sign * live[0] / len;
          args.sunDir[1] = sign * live[1] / len;
          args.sunDir[2] = sign * live[2] / len;
        }
        static uint32_t s_sunLogs = 0;
        if (s_sunLogs < 6 && g_frameCounter > 300) {
          s_sunLogs++;
          Logger::info(str::format("blessed: rtshadow sun[", g_frameCounter, "] = (", args.sunDir[0], ",", args.sunDir[1], ",", args.sunDir[2], ")"));
        }
      }
    }

    if (!ReadCbufferFloats(*targets.state, cfg.invVp, args.invViewProj, 16)) {
      // blessed: no matrix, no correct trace -- skip this frame rather than
      // fire rays from garbage. Static shaders' cbuffers are usually bound
      // well before the shadow-mask draw, so this is expected to be rare.
      return;
    }

    // blessed: raw camera position only -- NOT a delta, and NOT compared
    // against the tlas's recorded camPos here. That comparison has to
    // happen on the cs thread, at dispatch time, against whichever tlas
    // frame is actually current then; see BlessedShadowDispatchArgs and
    // DxvkContext::blessedRunShadowPass. Grabbing BlessedScene::currentFrame()
    // eagerly on this (app) thread, as this used to, could snapshot a
    // ping-pong tlas slot that endFrame() rebuilt out from under it before
    // the cs thread's turn to run this dispatch came up -- the fork-
    // cam-cascades.md seat traced that race to the sun mask flickering
    // frame to frame in a completely static scene.
    ReadCbufferFloats(*targets.state, cfg.camPos, args.camPosNow, 3); // ok to fail: stays 0

    args.softEnabled = cfg.softEnabled;
    if (cfg.softEnabled) {
      args.sunHalfAngleRad = cfg.sunHalfAngleRad;
      args.spp             = cfg.spp;

      if (!ReadCbufferFloats(*targets.state, cfg.prevVp, args.prevViewProj, 16)) {
        // blessed: no previous-frame matrix this frame -- zero it and let
        // the dxvk-side history tracking (BlessedSoftShadowState) reject
        // every pixel instead of reprojecting off garbage
        std::memset(args.prevViewProj, 0, sizeof(args.prevViewProj));
      }

      // blessed: this frame's camPosNow minus *last* frame's, one
      // app-thread OnDraw call apart -- see the comment on
      // BlessedShadowDispatchArgs::camReprojDelta. Static local is safe
      // here (single-threaded, in draw order) unlike the tlas camDelta
      // above, which really does need cs-thread-side resolution.
      static float s_lastCamPosNow[3] = { 0.0f, 0.0f, 0.0f };
      static bool  s_haveLastCamPosNow = false;
      if (s_haveLastCamPosNow) {
        args.camReprojDelta[0] = args.camPosNow[0] - s_lastCamPosNow[0];
        args.camReprojDelta[1] = args.camPosNow[1] - s_lastCamPosNow[1];
        args.camReprojDelta[2] = args.camPosNow[2] - s_lastCamPosNow[2];
      }
      s_lastCamPosNow[0] = args.camPosNow[0];
      s_lastCamPosNow[1] = args.camPosNow[1];
      s_lastCamPosNow[2] = args.camPosNow[2];
      s_haveLastCamPosNow = true;
    }

    g_frameCounter++;
    if (g_frameCounter == cfg.dumpFrame) {
      Logger::info(str::format("blessed: rtshadow cam now=(", args.camPosNow[0], ",", args.camPosNow[1], ",", args.camPosNow[2],
        ") sun=(", args.sunDir[0], ",", args.sunDir[1], ",", args.sunDir[2], ")"));
    }
    if (cfg.dumpRequested && !g_dumpDone && g_frameCounter == cfg.dumpFrame) {
      g_dumpDone = true;
      args.dump = true;

      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
      if (!dir.empty())
        args.dumpPath = dir + env::PlatformDirSlash + "rtshadow-120.pgm";
      else
        args.dump = false; // nowhere to write it
    }

    ctx->EmitCs([args = std::move(args)] (DxvkContext* dxvkCtx) {
      dxvkCtx->blessedRunShadowPass(args);
    });
  }

}
