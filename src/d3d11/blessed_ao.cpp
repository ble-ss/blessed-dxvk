// blessed: app-thread half of BLESSED_AO=rt -- see blessed_ao.h
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "blessed_ao.h"

#include "d3d11_context_imm.h"
#include "d3d11_view_srv.h"

#include "../dxvk/blessed/blessed_ao.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // "<vs|ps>:<slot>:<byte offset>", as the shadow pass's BLESSED_SHADOW_INVVP
    struct AoSlot {
      bool             valid  = false;
      D3D11ShaderType  stage  = D3D11ShaderType::ePixel;
      uint32_t         slot   = 0;
      uint32_t         offset = 0;
    };

    AoSlot ParseSlot(const std::string& s, uint32_t slot, uint32_t offset) {
      AoSlot out;
      out.valid  = true;
      out.slot   = slot;
      out.offset = offset;

      size_t a = s.find(':');
      size_t b = a == std::string::npos ? std::string::npos : s.find(':', a + 1);
      if (b == std::string::npos)
        return out;

      out.stage  = s.substr(0, a) == "vs" ? D3D11ShaderType::eVertex : D3D11ShaderType::ePixel;
      out.slot   = std::strtoul(s.substr(a + 1, b - a - 1).c_str(), nullptr, 10);
      out.offset = std::strtoul(s.substr(b + 1).c_str(), nullptr, 10);
      return out;
    }

    float EnvFloat(const char* name, float fallback, float lo, float hi) {
      std::string s = env::getEnvVar(name);
      if (s.empty())
        return fallback;
      return std::clamp(std::strtof(s.c_str(), nullptr), lo, hi);
    }

    struct AoConfig {
      std::vector<std::string> psTokens;   // 8+ hex chars after "fs."
      uint32_t       outputSrv  = 1u;
      uint32_t       depthSrv   = 2u;
      AoSlot         invVp;
      AoSlot         camPos;
      float          farValue   = 1.0f;
      float          radius     = 64.0f;
      float          strength   = 1.0f;
      uint32_t       rays       = 2u;
      bool           halfRes    = true;
      BlessedAoDebug debug      = BlessedAoDebug::None;

      // blessed: parsed on first use, which only happens with BLESSED_AO=rt
      static const AoConfig& Get() {
        static const AoConfig s_config = [] {
          AoConfig c;

          std::string ps = env::getEnvVar("BLESSED_AO_PS");
          if (ps.empty())
            ps = "ddcce9bc";

          size_t pos = 0;
          while (pos < ps.size()) {
            size_t comma = ps.find(',', pos);
            std::string tok = ps.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            pos = comma == std::string::npos ? ps.size() : comma + 1;
            if (tok.rfind("fs.", 0) == 0)
              tok = tok.substr(3);
            if (tok.size() >= 8)
              c.psTokens.push_back(tok);
          }

          std::string srv = env::getEnvVar("BLESSED_AO_SRV");
          if (!srv.empty())
            c.outputSrv = std::strtoul(srv.c_str(), nullptr, 10);
          std::string depthSrv = env::getEnvVar("BLESSED_AO_DEPTH_SRV");
          if (!depthSrv.empty())
            c.depthSrv = std::strtoul(depthSrv.c_str(), nullptr, 10);

          c.invVp  = ParseSlot(env::getEnvVar("BLESSED_AO_INVVP"),  12u, 512u);
          c.camPos = ParseSlot(env::getEnvVar("BLESSED_AO_CAMPOS"), 12u, 640u);

          c.farValue = env::getEnvVar("BLESSED_AO_FAR") == "0" ? 0.0f : 1.0f;
          c.radius   = EnvFloat("BLESSED_AO_RADIUS", 64.0f, 1.0f, 4096.0f);
          c.strength = EnvFloat("BLESSED_AO_STRENGTH", 1.0f, 0.0f, 4.0f);
          c.rays     = uint32_t(EnvFloat("BLESSED_AO_RAYS", 2.0f, 1.0f, 16.0f));
          c.halfRes  = env::getEnvVar("BLESSED_AO_HALF") != "0";

          std::string dbg = env::getEnvVar("BLESSED_AO_DEBUG");
          c.debug = dbg == "black"  ? BlessedAoDebug::Black
                  : dbg == "white"  ? BlessedAoDebug::White
                  : dbg == "raw"    ? BlessedAoDebug::Raw
                  : dbg == "noblur" ? BlessedAoDebug::NoBlur
                  : dbg == "hist"   ? BlessedAoDebug::Hist
                  : dbg == "normal" ? BlessedAoDebug::Normal
                  : dbg == "reproj" ? BlessedAoDebug::Reproj
                  : BlessedAoDebug::None;

          Logger::info(str::format("BlessedAo: enabled (BLESSED_AO=rt), ps=", ps,
            " srv=", c.outputSrv, " depth_srv=", c.depthSrv,
            " rays=", c.rays, " radius=", c.radius, " strength=", c.strength,
            " half=", c.halfRes ? 1 : 0, " debug=", dbg.empty() ? "none" : dbg));
          return c;
        }();

        return s_config;
      }
    };

    // blessed: per-pixel-shader-pointer match memo, as BlessedHook's
    std::unordered_map<const void*, bool> g_matchCache;

    bool Matches(D3D11PixelShader* ps) {
      auto it = g_matchCache.find(ps);
      if (it != g_matchCache.end())
        return it->second;

      const std::string& name = ps->GetCommonShader()->GetName();
      bool match = false;
      if (name.rfind("fs.", 0) == 0) {
        for (const auto& tok : AoConfig::Get().psTokens)
          match |= name.compare(3, tok.size(), tok) == 0;
      }

      g_matchCache.emplace(ps, match);
      return match;
    }

    // blessed: same as the shadow pass's ReadCbufferFloats
    bool ReadCbufferFloats(const D3D11ContextState& state, const AoSlot& slot, float* out, uint32_t count) {
      const auto& stage = state.cbv[slot.stage];
      if (!slot.valid || slot.slot >= stage.maxCount)
        return false;

      D3D11Buffer* buffer = stage.buffers[slot.slot].buffer.ptr();
      if (!buffer)
        return false;

      void* mapPtr = buffer->GetMapPtr();
      if (!mapPtr)
        return false;

      UINT offset = stage.buffers[slot.slot].constantOffset * 16u + slot.offset;
      if (offset + count * sizeof(float) > buffer->Desc()->ByteWidth)
        return false;

      std::memcpy(out, reinterpret_cast<const uint8_t*>(mapPtr) + offset, count * sizeof(float));
      return true;
    }

    DxvkImage* SrvImage(const D3D11ContextState& state, uint32_t slot) {
      const auto& srvs = state.srv[D3D11ShaderType::ePixel];
      if (slot >= srvs.maxCount)
        return nullptr;

      D3D11ShaderResourceView* srv = srvs.views[slot].ptr();
      if (!srv)
        return nullptr;

      Rc<DxvkImageView> view = srv->GetImageView();
      return view != nullptr ? view->image() : nullptr;
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

    // blessed: one-time notes about why a matched draw was skipped
    void WarnOnce(bool& flag, const std::string& msg) {
      if (!flag) {
        flag = true;
        Logger::warn(msg);
      }
    }

    bool g_warnNoOutput  = false;
    bool g_warnNoStorage = false;
    bool g_warnFormat    = false;
    bool g_warnNoDepth   = false;
    bool g_warnNoMatrix  = false;
    bool g_loggedFirst   = false;

  }


  namespace blessed_ao_detail {
    extern const bool g_enabled = env::getEnvVar("BLESSED_AO") == "rt";
  }


  void BlessedAo::OnDrawPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    D3D11PixelShader* ps = state.ps.ptr();
    if (!ps || !Matches(ps))
      return;

    const AoConfig& cfg = AoConfig::Get();

    DxvkImage* output = SrvImage(state, cfg.outputSrv);
    if (!output) {
      WarnOnce(g_warnNoOutput, str::format("BlessedAo: matched draw has no image at ps srv ", cfg.outputSrv));
      return;
    }

    if (!(output->info().usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
      WarnOnce(g_warnNoStorage, "BlessedAo: sao texture was not created storage-capable; skipping");
      return;
    }

    if (output->info().format != VK_FORMAT_R8G8B8A8_UNORM) {
      WarnOnce(g_warnFormat, str::format("BlessedAo: sao texture format ", uint32_t(output->info().format),
        " is not rgba8 unorm; skipping"));
      return;
    }

    DxvkImage* depth = SrvImage(state, cfg.depthSrv);
    if (!depth || !(depth->formatInfo()->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)) {
      WarnOnce(g_warnNoDepth, str::format("BlessedAo: no depth image at ps srv ", cfg.depthSrv));
      return;
    }

    BlessedAoDispatchArgs args;
    if (!ReadCbufferFloats(state, cfg.invVp, args.invViewProj, 16)) {
      WarnOnce(g_warnNoMatrix, "BlessedAo: could not read CameraViewProjInverse; skipping");
      return;
    }
    ReadCbufferFloats(state, cfg.camPos, args.camPosNow, 3); // ok to fail: stays 0

    VkExtent3D extent = output->info().extent;
    args.depthView     = MakeView(depth, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    args.outputView    = MakeView(output, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    args.width         = extent.width;
    args.height        = extent.height;
    args.farDepthValue = cfg.farValue;
    args.radius        = cfg.radius;
    args.strength      = cfg.strength;
    args.rays          = cfg.rays;
    args.halfRes       = cfg.halfRes;
    args.debug         = cfg.debug;

    if (!g_loggedFirst) {
      g_loggedFirst = true;
      Logger::info(str::format("BlessedAo: first match: sao ", extent.width, "x", extent.height,
        ", depth ", depth->info().extent.width, "x", depth->info().extent.height,
        ", cam (", args.camPosNow[0], ",", args.camPosNow[1], ",", args.camPosNow[2], ")"));
    }

    ctx->EmitCs([args = std::move(args)] (DxvkContext* dxvkCtx) {
      dxvkCtx->blessedRunAoPass(args);
    });
  }

}
