// blessed: sun shadow cascade learner + skip -- see blessed_cascades.h
#include "blessed_cascades.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "d3d11_shader.h"
#include "d3d11_view_dsv.h"

#include "../dxvk/dxvk_image.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    std::vector<uint32_t> ParseSrvSlots(const std::string& csv) {
      std::vector<uint32_t> out;
      size_t pos = 0;

      while (pos <= csv.size()) {
        size_t comma = csv.find(',', pos);
        std::string tok = csv.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);

        if (!tok.empty())
          out.push_back(std::strtoul(tok.c_str(), nullptr, 10));

        if (comma == std::string::npos)
          break;
        pos = comma + 1;
      }

      return out;
    }

    struct CascadeConfig {
      bool                   enabled = false;
      std::vector<uint32_t>  srvSlots;
    };

    // blessed: pure function of env vars alone -- no Logger call, no other
    // static touched -- safe to run at static-init time (dll load) for
    // blessed_cascades_detail::g_enabled below. c11's own comment on
    // ComputeHookEnabled explains why this split matters: this module's
    // *full* config (below) still needs to stay lazy, because its
    // Logger::info call would otherwise run during dxvk's own static init,
    // before Logger's statics (its log file name/handle) are ready --
    // exactly the crash a first deploy of this seat's earlier draft hit
    // (game exits before the main menu, a 0-byte log with the wrong name).
    bool ComputeCascadeEnabled() {
      return env::getEnvVar("BLESSED_SKIP_CASCADES") == "1"
          && env::getEnvVar("BLESSED_HOOK_MODE") == "rtshadow";
    }

    // blessed: function-local magic static -- lazy on purpose. Only ever
    // reached from LearnFromMaskDraw, itself only reached once
    // blessed_cascades_detail::g_enabled is already known true (see
    // IsEnabled() in blessed_cascades.h), i.e. well after static init, from
    // a real Draw call -- so the thread-safe-init-guard cost of a magic
    // static is off the disabled path this seat exists to make free, and
    // Logger::info here runs at a point where Logger is fully alive.
    const CascadeConfig& GetConfig() {
      static CascadeConfig s_config = [] {
        CascadeConfig c;
        c.enabled = ComputeCascadeEnabled();

        if (!c.enabled)
          return c;

        std::string srvStr = env::getEnvVar("BLESSED_SKIP_CASCADES_SRV");
        c.srvSlots = ParseSrvSlots(srvStr.empty() ? "4,6" : srvStr);

        Logger::info(str::format("BlessedCascadeSkip: enabled, ",
          c.srvSlots.size(), " ps srv slot(s)"));

        return c;
      }();

      return s_config;
    }

    // blessed: app-thread only (immediate context, under D3D10DeviceLock) --
    // see the "single-threaded by construction" note in blessed_cascades.h.
    // Replaced wholesale on every mask-draw match, so a stale entry never
    // outlives more than one frame even if the image behind it was
    // destroyed and its address reused.
    std::unordered_set<DxvkImage*> g_cascadeImages;

    uint32_t g_swapchainWidth  = 0;
    uint32_t g_swapchainHeight = 0;

    // ---- BLESSED_PROBE_DIR/cascades.jsonl, window of 120 presents ----
    uint64_t              g_presentsInWindow = 0;
    uint64_t              g_skippedDraws     = 0;
    uint64_t              g_skippedClears    = 0;
    std::ofstream         g_logFile;
    bool                  g_logFileTried     = false;
    dxvk::high_resolution_clock::time_point g_processStart;
    bool                  g_processStartSet  = false;

  }


  // blessed: definition for the header's inline IsEnabled()/ShouldSkipDraw()/
  // ShouldSkipClear() -- see blessed_cascades.h. Independent of GetConfig()
  // above on purpose (see ComputeCascadeEnabled()'s comment).
  namespace blessed_cascades_detail {
    extern const bool g_enabled = ComputeCascadeEnabled();
  }


  void BlessedCascadeSkip::LearnFromMaskDraw(const D3D11ContextState& state) {
    if (!IsEnabled())
      return;

    g_cascadeImages.clear();

    const auto& srvStage = state.srv[D3D11ShaderType::ePixel];

    for (uint32_t slot : GetConfig().srvSlots) {
      if (slot >= srvStage.maxCount)
        continue;

      D3D11ShaderResourceView* srv = srvStage.views[slot].ptr();
      if (!srv)
        continue;

      Rc<DxvkImageView> view = srv->GetImageView();
      if (view && view->image())
        g_cascadeImages.insert(view->image());
    }
  }


  bool BlessedCascadeSkip::ShouldSkipDrawSlow(const D3D11ContextState& state) {
    if (g_cascadeImages.empty())
      return false;

    // no rtv bound
    for (uint32_t i = 0; i < state.om.maxRtv; i++) {
      if (state.om.rtvs[i].ptr())
        return false;
    }

    D3D11DepthStencilView* dsv = state.om.dsv.ptr();
    if (!dsv)
      return false;

    Rc<DxvkImageView> view = dsv->GetImageView();
    return view && g_cascadeImages.count(view->image()) != 0;
  }


  bool BlessedCascadeSkip::ShouldSkipClearSlow(D3D11DepthStencilView* dsv) {
    if (g_cascadeImages.empty() || !dsv)
      return false;

    Rc<DxvkImageView> view = dsv->GetImageView();
    return view && g_cascadeImages.count(view->image()) != 0;
  }


  void BlessedCascadeSkip::RecordSkippedDraw() {
    g_skippedDraws++;
  }


  void BlessedCascadeSkip::RecordSkippedClear() {
    g_skippedClears++;
  }


  void BlessedCascadeSkip::NotifySwapchainExtent(uint32_t width, uint32_t height) {
    if (!IsEnabled())
      return;

    if (width != g_swapchainWidth || height != g_swapchainHeight) {
      g_swapchainWidth  = width;
      g_swapchainHeight = height;
      // blessed: a resolution change invalidates whatever we learned --
      // the old cascade images may already be gone. The next matching
      // mask draw re-learns fresh ones before any skip check runs again.
      g_cascadeImages.clear();
    }
  }


  void BlessedCascadeSkip::OnPresent() {
    if (!IsEnabled())
      return;

    if (!g_processStartSet) {
      g_processStartSet = true;
      g_processStart = dxvk::high_resolution_clock::now();
    }

    if (++g_presentsInWindow < 120)
      return;

    if (!g_logFileTried) {
      g_logFileTried = true;
      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

      if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        g_logFile.open(dir + env::PlatformDirSlash + "cascades.jsonl", std::ios::out | std::ios::app);
      }
    }

    double t = std::chrono::duration<double>(
      dxvk::high_resolution_clock::now() - g_processStart).count();

    double drawsPerFrame  = double(g_skippedDraws)  / double(g_presentsInWindow);
    double clearsPerFrame = double(g_skippedClears) / double(g_presentsInWindow);

    if (g_logFile.is_open()) {
      g_logFile << str::format("{\"t\":", t,
        ",\"frames\":", g_presentsInWindow,
        ",\"skipped_draws\":", drawsPerFrame,
        ",\"skipped_clears\":", clearsPerFrame,
        ",\"cascade_images\":", g_cascadeImages.size(), "}\n");
      g_logFile.flush();
    }

    g_presentsInWindow = 0;
    g_skippedDraws     = 0;
    g_skippedClears    = 0;
  }

}
