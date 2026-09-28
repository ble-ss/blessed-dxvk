// blessed: refl-harden -- the water reflection cube's faces at half rate, gated on camera motion and face age
#include "blessed_reflect_halfrate.h"
#include "blessed_vanilla_halfrate.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "d3d11_shader.h"
#include "d3d11_view_dsv.h"
#include "d3d11_view_rtv.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // blessed: the plugin's live switch (F8), shared with BLESSED_HALFRATE.
    // Keep in step with skse/skybench/src/halfrate_switch.cpp in the main repo.
    struct SharedSwitch {
      uint32_t      magic;
      uint32_t      version;
      volatile LONG enabled;
      volatile LONG toggles;
    };

    constexpr uint32_t       SwitchMagic   = 0x46524842u; // "BHRF"
    constexpr uint32_t       SwitchVersion = 1u;
    constexpr const wchar_t* SwitchName    = L"Local\\BlessedHalfRate";

    constexpr uint32_t MaxFaces    = 8u;   // the cube has 6; room for a surprise
    constexpr uint32_t ArmFrames   = 3u;   // frames the learned cube must be seen before a skip
    constexpr uint64_t LostFrames  = 120u; // frames without the learned cube before another may be learned
    constexpr double   LogSeconds  = 5.0;

    std::vector<std::string> SplitCsv(const std::string& csv) {
      std::vector<std::string> out;
      size_t pos = 0;

      while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        std::string tok = csv.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? csv.size() : comma + 1;

        if (tok.rfind("fs.", 0) == 0 || tok.rfind("ps.", 0) == 0)
          tok = tok.substr(3);
        if (!tok.empty())
          out.push_back(std::move(tok));
      }

      return out;
    }

    bool HasToken(const std::string& csv, const char* token) {
      for (const auto& t : SplitCsv(csv)) {
        if (t == token)
          return true;
      }
      return false;
    }

    float EnvFloat(const char* name, float fallback) {
      std::string s = env::getEnvVar(name);
      return s.empty() ? fallback : std::strtof(s.c_str(), nullptr);
    }

    // BLESSED_REFLECT_HALFRATE=1 turns it on. The retired reflect part of
    // BLESSED_HALFRATE (BLESSED_HALFRATE_PARTS listing "reflect", with
    // BLESSED_HALFRATE not 0) is routed here too, so old plan rows run
    // this code. BLESSED_REFLECT_HALFRATE=0 wins over both.
    bool LegacyRequested() {
      return env::getEnvVar("BLESSED_HALFRATE") != "0"
          && HasToken(env::getEnvVar("BLESSED_HALFRATE_PARTS"), "reflect");
    }

    bool ComputeEnabled() {
      std::string v = env::getEnvVar("BLESSED_REFLECT_HALFRATE");
      if (v == "0")
        return false;
      return v == "1" || LegacyRequested();
    }

    struct ReflectConfig {
      bool                     gate       = true;
      float                    maxDeg     = 1.0f;
      float                    maxMove    = 50.0f;
      uint32_t                 maxAge     = 6u;
      bool                     blind      = false;
      bool                     useSwitch  = true;
      uint32_t                 maxSize    = 2048u;
      std::vector<std::string> psTokens   = { "79c17b07" };

      static const ReflectConfig& Get() {
        static const ReflectConfig s_config = [] {
          ReflectConfig c;
          c.gate      = env::getEnvVar("BLESSED_REFLECT_HALFRATE_GATE") != "0";
          // the volumetric half rate's thresholds unless ours are set
          c.maxDeg    = EnvFloat("BLESSED_REFLECT_HALFRATE_MAXDEG",
                          EnvFloat("BLESSED_VOL_HALFRATE_MAXDEG", c.maxDeg));
          c.maxMove   = EnvFloat("BLESSED_REFLECT_HALFRATE_MAXMOVE",
                          EnvFloat("BLESSED_VOL_HALFRATE_MAXMOVE", c.maxMove));
          c.blind     = env::getEnvVar("BLESSED_REFLECT_HALFRATE_BLIND") == "1";
          c.useSwitch = env::getEnvVar("BLESSED_REFLECT_HALFRATE_SWITCH") != "0";

          std::string age = env::getEnvVar("BLESSED_REFLECT_HALFRATE_MAXAGE");
          if (!age.empty())
            c.maxAge = std::clamp<uint32_t>(uint32_t(std::strtoul(age.c_str(), nullptr, 10)), 2u, 1000u);

          std::string size = env::getEnvVar("BLESSED_REFLECT_HALFRATE_MAX_SIZE");
          if (!size.empty())
            c.maxSize = std::max<uint32_t>(uint32_t(std::strtoul(size.c_str(), nullptr, 10)), 16u);

          std::string ps = env::getEnvVar("BLESSED_REFLECT_HALFRATE_PS");
          if (!ps.empty())
            c.psTokens = SplitCsv(ps);

          std::string psList;
          for (const auto& t : c.psTokens)
            psList += (psList.empty() ? "" : ",") + t;

          Logger::info(str::format("BlessedReflectHalfRate: on",
            env::getEnvVar("BLESSED_REFLECT_HALFRATE") == "1" ? "" : " (via the retired BLESSED_HALFRATE_PARTS=reflect)",
            ", gate=", c.gate ? 1 : 0,
            " maxdeg=", c.maxDeg, " maxmove=", c.maxMove,
            " maxage=", c.maxAge, " blind=", c.blind ? 1 : 0,
            " switch=", c.useSwitch ? 1 : 0, " ps=", psList));
          return c;
        }();
        return s_config;
      }
    };

    // why a face pass was drawn instead of skipped
    enum class Reason : uint8_t { None, Learning, Cadence, Motion, NoCamera, Age };

    enum class FaceState : uint8_t { Unknown, Render, Skip };

    enum class OmClass : uint8_t { Other, Face, Candidate };

    struct Counters {
      uint64_t frames      = 0;
      uint64_t rendered    = 0;   // face passes drawn, forced or not
      uint64_t skipped     = 0;
      uint64_t forcedCadence  = 0;
      uint64_t forcedMotion   = 0;
      uint64_t forcedNoCamera = 0;
      uint64_t forcedAge      = 0;
      uint64_t forcedLearning = 0;
      uint64_t draws       = 0;   // draws dropped
      uint64_t clears      = 0;   // colour clears dropped
      float    maxDeg      = 0.0f;
      float    maxMove     = 0.0f;
    };

    struct State {
      // the live switch
      HANDLE              mapping   = nullptr;
      const SharedSwitch* shared    = nullptr;
      high_resolution_clock::time_point lastOpenTry;
      bool                openTried = false;
      bool                live      = true;
      bool                liveKnown = false;

      // the learned cube
      uint64_t cubeCookie  = 0;     // 0: not learned
      uint64_t depthCookie = 0;
      uint32_t cubeSize    = 0;
      uint32_t armFrames   = 0;     // frames the learned cube was drawn
      bool     armed       = false;
      bool     cubeSeenNow = false;
      uint64_t lastCubeFrame = 0;
      std::unordered_set<std::string> companions;  // ps names seen while learning

      // om classification, per OM generation
      uint64_t omGeneration = ~0ull;
      OmClass  omClass      = OmClass::Other;
      uint32_t omFace       = 0;

      // shader memos (object pointers, cleared now and then like BlessedHalfRate's)
      std::unordered_map<const void*, bool> psMemo;
      std::unordered_map<const void*, bool> csMemo;

      // the frame
      uint64_t  frame            = 1;
      Reason    frameReason      = Reason::Learning;   // Reason::None: skipping allowed
      bool      skippedNow       = false;
      bool      lastFrameSkipped = false;
      FaceState faces[MaxFaces]  = { };
      uint64_t  lastDrawn[MaxFaces] = { };             // 0: never

      // the camera (the volumetric generate's)
      BlessedVolCameraPose poseNow;
      BlessedVolCameraPose posePrev;
      bool      motionKnown = false;
      float     motionDeg   = 0.0f;
      float     motionMove  = 0.0f;

      Counters  n;
      high_resolution_clock::time_point lastLog = high_resolution_clock::now();
    };

    State g;

    std::string ShortName(const std::string& name) {
      // "fs.<32 hex>" -> the first 8 hex digits
      return name.size() >= 11 ? name.substr(3, 8) : name;
    }

    bool NameMatches(const std::string& name, const std::vector<std::string>& tokens) {
      if (name.size() < 4)
        return false;
      for (const auto& tok : tokens) {
        if (name.compare(3, tok.size(), tok) == 0)
          return true;
      }
      return false;
    }

    bool IsOpenerPs(const D3D11ContextState& state) {
      auto ps = state.ps.ptr();
      if (!ps)
        return false;

      auto it = g.psMemo.find(ps);
      if (it != g.psMemo.end())
        return it->second;

      bool match = NameMatches(ps->GetCommonShader()->GetName(), ReflectConfig::Get().psTokens);
      g.psMemo.emplace(ps, match);
      return match;
    }

    void UpdateWatch() {
      blessed_reflect_halfrate_detail::g_watch = g.live;
    }

    void PollSwitch() {
      const ReflectConfig& c = ReflectConfig::Get();

      if (c.useSwitch && !g.shared) {
        auto now = high_resolution_clock::now();

        if (!g.openTried || now - g.lastOpenTry >= std::chrono::seconds(1)) {
          g.openTried   = true;
          g.lastOpenTry = now;

          HANDLE mapping = ::OpenFileMappingW(FILE_MAP_READ, FALSE, SwitchName);

          if (mapping) {
            void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, sizeof(SharedSwitch));

            if (view) {
              g.mapping = mapping;
              g.shared  = reinterpret_cast<const SharedSwitch*>(view);
            } else {
              ::CloseHandle(mapping);
            }
          }
        }
      }

      bool live = true;

      if (g.shared && g.shared->magic == SwitchMagic && g.shared->version == SwitchVersion)
        live = g.shared->enabled != 0;

      if (!g.liveKnown || live != g.live) {
        Logger::info(str::format("BlessedReflectHalfRate: ", live ? "live" : "paused",
          g.shared ? " (plugin switch)" : " (no plugin switch)"));
        g.liveKnown = true;
        g.live = live;
      }
    }

    // the image behind a view, or null
    DxvkImage* ImageOf(const Rc<DxvkImageView>& view) {
      return view != nullptr ? view->image() : nullptr;
    }

    // the cube's shape: one layer of a 6-layer cube-compatible square
    // rgba16f image, with a single-layer depth image of the same size
    bool IsCubeShape(const Rc<DxvkImageView>& colour, const Rc<DxvkImageView>& depth) {
      DxvkImage* ci = ImageOf(colour);
      DxvkImage* di = ImageOf(depth);
      if (!ci || !di)
        return false;

      const auto& cInfo = ci->info();
      const auto& dInfo = di->info();
      auto        cView = colour->info();

      return cInfo.type == VK_IMAGE_TYPE_2D
          && (cInfo.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
          && cInfo.numLayers == 6u
          && cInfo.extent.width == cInfo.extent.height
          && cInfo.extent.width >= 16u
          && cInfo.extent.width <= ReflectConfig::Get().maxSize
          && cView.format == VK_FORMAT_R16G16B16A16_SFLOAT
          && cView.layerCount == 1u
          && dInfo.numLayers == 1u
          && dInfo.extent.width  == cInfo.extent.width
          && dInfo.extent.height == cInfo.extent.height;
    }

    void Learn(const Rc<DxvkImageView>& colour, const Rc<DxvkImageView>& depth, const D3D11ContextState& state) {
      bool relearn = g.cubeCookie != 0;

      g.cubeCookie  = ImageOf(colour)->cookie();
      g.depthCookie = ImageOf(depth)->cookie();
      g.cubeSize    = ImageOf(colour)->info().extent.width;
      g.armFrames   = 0;
      g.armed       = false;
      g.companions.clear();

      for (uint32_t i = 0; i < MaxFaces; i++) {
        g.faces[i]     = FaceState::Unknown;
        g.lastDrawn[i] = 0;
      }

      Logger::info(str::format("BlessedReflectHalfRate: ", relearn ? "relearned" : "learned",
        " the water cube: ", g.cubeSize, "x", g.cubeSize, " rgba16f, 6 layers, depth format ",
        uint32_t(ImageOf(depth)->info().format), ", taught by ps ",
        ShortName(state.ps->GetCommonShader()->GetName()),
        "; skipping starts after ", ArmFrames, " frames"));
    }

    OmClass Classify(const D3D11ContextState& state, uint32_t* face) {
      if (state.om.maxRtv != 1u || !state.om.dsv.ptr() || !state.om.rtvs[0].ptr())
        return OmClass::Other;

      Rc<DxvkImageView> colour = state.om.rtvs[0]->GetImageView();
      Rc<DxvkImageView> depth  = state.om.dsv->GetImageView();
      DxvkImage* ci = ImageOf(colour);
      DxvkImage* di = ImageOf(depth);

      if (!ci || !di)
        return OmClass::Other;

      if (g.cubeCookie && ci->cookie() == g.cubeCookie && di->cookie() == g.depthCookie) {
        auto view = colour->info();
        if (view.layerCount != 1u || view.layerIndex >= MaxFaces)
          return OmClass::Other;
        *face = view.layerIndex;
        return OmClass::Face;
      }

      return IsCubeShape(colour, depth) ? OmClass::Candidate : OmClass::Other;
    }

    // once per face pass: draw it or skip it
    FaceState Decide(uint32_t face) {
      FaceState& fs = g.faces[face];
      if (fs != FaceState::Unknown)
        return fs;

      const ReflectConfig& c = ReflectConfig::Get();
      Reason reason = g.armed ? g.frameReason : Reason::Learning;

      if (reason == Reason::None) {
        uint64_t age = g.lastDrawn[face] ? g.frame - g.lastDrawn[face] : ~0ull;
        if (age >= uint64_t(c.maxAge))
          reason = Reason::Age;
      }

      switch (reason) {
        case Reason::None:     break;
        case Reason::Learning: g.n.forcedLearning++; break;
        case Reason::Cadence:  g.n.forcedCadence++;  break;
        case Reason::Motion:   g.n.forcedMotion++;   break;
        case Reason::NoCamera: g.n.forcedNoCamera++; break;
        case Reason::Age:      g.n.forcedAge++;      break;
      }

      if (reason == Reason::None) {
        fs = FaceState::Skip;
        g.skippedNow = true;
        g.n.skipped++;
      } else {
        fs = FaceState::Render;
        g.lastDrawn[face] = g.frame;
        g.n.rendered++;
      }

      return fs;
    }

    void WriteLog() {
      auto now = high_resolution_clock::now();
      double t = std::chrono::duration<double>(now - g.lastLog).count();

      if (t < LogSeconds)
        return;

      Counters& n = g.n;

      if (n.rendered + n.skipped) {
        Logger::info(str::format("BlessedReflectHalfRate: ", uint32_t(t + 0.5), " s, ", n.frames, " frames: faces rendered ",
          n.rendered, ", skipped ", n.skipped,
          "; forced by motion ", n.forcedMotion, ", by age ", n.forcedAge,
          ", after a skip ", n.forcedCadence, ", no camera ", n.forcedNoCamera,
          ", learning ", n.forcedLearning,
          "; dropped ", n.draws, " draws ", n.clears, " clears; max turn ", n.maxDeg,
          " deg, max move ", n.maxMove));
      }

      n = Counters();
      g.lastLog = now;
    }

  }


  namespace blessed_reflect_halfrate_detail {
    extern const bool g_enabled = ComputeEnabled();
    bool g_watch = false;
  }


  bool BlessedReflectHalfRate::OnDraw(const D3D11ContextState& state) {
    if (state.om.blessedOmGeneration != g.omGeneration) {
      g.omGeneration = state.om.blessedOmGeneration;
      g.omClass = Classify(state, &g.omFace);
    }

    if (likely(g.omClass == OmClass::Other))
      return false;

    if (g.omClass == OmClass::Candidate) {
      // a second cube-shaped target never displaces a cube still in use
      if (g.cubeCookie && g.frame - g.lastCubeFrame < LostFrames)
        return false;

      if (!IsOpenerPs(state))
        return false;

      Learn(state.om.rtvs[0]->GetImageView(), state.om.dsv->GetImageView(), state);
      g.omClass = Classify(state, &g.omFace);

      if (g.omClass != OmClass::Face)
        return false;
    }

    g.cubeSeenNow   = true;
    g.lastCubeFrame = g.frame;

    if (!g.armed) {
      if (auto ps = state.ps.ptr())
        g.companions.insert(ShortName(ps->GetCommonShader()->GetName()));
    }

    if (Decide(g.omFace) == FaceState::Skip) {
      g.n.draws++;
      return true;
    }

    return false;
  }


  bool BlessedReflectHalfRate::OnClearRtv(D3D11RenderTargetView* rtv) {
    if (!g.armed || !rtv)
      return false;

    Rc<DxvkImageView> view = rtv->GetImageView();
    DxvkImage* image = ImageOf(view);

    if (!image || image->cookie() != g.cubeCookie)
      return false;

    auto info = view->info();
    if (info.layerCount != 1u || info.layerIndex >= MaxFaces)
      return false;

    if (Decide(info.layerIndex) == FaceState::Skip) {
      g.n.clears++;
      return true;
    }

    return false;
  }


  void BlessedReflectHalfRate::OnDispatch(const D3D11ContextState& state) {
    if (g.poseNow.valid)
      return;

    auto cs = state.cs.ptr();
    if (!cs)
      return;

    auto it = g.csMemo.find(cs);
    bool generate;

    if (it != g.csMemo.end()) {
      generate = it->second;
    } else {
      generate = BlessedIsVolGenerateName(cs->GetCommonShader()->GetName());
      g.csMemo.emplace(cs, generate);
    }

    if (generate)
      g.poseNow = BlessedReadVolCamera(state);
  }


  void BlessedReflectHalfRate::OnPresent() {
    const ReflectConfig& c = ReflectConfig::Get();

    PollSwitch();

    // the frame that just ended
    g.n.frames++;
    g.lastFrameSkipped = g.skippedNow;
    g.skippedNow = false;

    for (uint32_t i = 0; i < MaxFaces; i++)
      g.faces[i] = FaceState::Unknown;

    if (g.cubeSeenNow && g.cubeCookie && !g.armed && ++g.armFrames >= ArmFrames) {
      g.armed = true;

      std::string list;
      for (const auto& name : g.companions)
        list += (list.empty() ? "" : ",") + name;

      Logger::info(str::format("BlessedReflectHalfRate: armed; the cube passes' pixel shaders: ", list));
    }

    g.cubeSeenNow = false;

    // its motion: this frame's generate camera against the last frame's
    g.motionKnown = g.poseNow.valid && g.posePrev.valid;

    if (g.motionKnown) {
      BlessedVolCameraMotion(g.poseNow, g.posePrev, &g.motionDeg, &g.motionMove);
      g.n.maxDeg  = std::max(g.n.maxDeg,  g.motionDeg);
      g.n.maxMove = std::max(g.n.maxMove, g.motionMove);
    }

    g.posePrev = g.poseNow;
    g.poseNow  = BlessedVolCameraPose();

    g.frame++;

    // the next frame's rule; the cube is its first pass, so this frame's
    // motion is the freshest there is
    if (!g.live || !g.armed)
      g.frameReason = Reason::Learning;
    else if (g.lastFrameSkipped)
      g.frameReason = Reason::Cadence;
    else if (!c.gate)
      g.frameReason = Reason::None;
    else if (!g.motionKnown)
      g.frameReason = c.blind ? Reason::None : Reason::NoCamera;
    else if (g.motionDeg > c.maxDeg || g.motionMove > c.maxMove)
      g.frameReason = Reason::Motion;
    else
      g.frameReason = Reason::None;

    // shader objects die with their owners; relearn the memos now and then
    if ((g.frame % 3600u) == 0u) {
      g.psMemo.clear();
      g.csMemo.clear();
    }

    UpdateWatch();
    WriteLog();
  }

}
