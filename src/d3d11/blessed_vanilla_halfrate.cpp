// blessed: perf-halfrate -- see blessed_vanilla_halfrate.h
#include "blessed_vanilla_halfrate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d11_buffer.h"
#include "d3d11_shader.h"
#include "d3d11_view_uav.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // ---- env helpers (pure, safe at static init) ----

    float EnvFloat(const char* name, float fallback) {
      std::string s = env::getEnvVar(name);
      return s.empty() ? fallback : std::strtof(s.c_str(), nullptr);
    }

    uint32_t VolLevel() {
      std::string s = env::getEnvVar("BLESSED_VOL_HALFRATE");
      if (s == "1") return 1u;
      if (s == "2") return 2u;
      return 0u;
    }

    // blessed: BLESSED_VOL_PERIOD -- run the chain at least once every n
    // eligible frames (default 2 = today's alternate-frame behaviour
    // exactly). n=1 disables the skip (never skips).
    uint32_t VolPeriod() {
      std::string s = env::getEnvVar("BLESSED_VOL_PERIOD");
      if (s.empty())
        return 2u;
      long v = std::strtol(s.c_str(), nullptr, 10);
      return v >= 1 ? uint32_t(v) : 2u;
    }

    bool ComputeVolScreen() {
      // blessed: level 2 skips pass 138's draw, which the traced volumetrics
      // (BLESSED_VOLUMETRICS) run from: they stay on level 1
      std::string vol = env::getEnvVar("BLESSED_VOLUMETRICS");
      return VolLevel() == 2u && (vol.empty() || vol == "off");
    }

    // a csv of bare 8+ char hex prefixes ("cs."/"fs." allowed)
    std::vector<std::string> SplitHexTokens(const std::string& csv, const char* fallback) {
      std::vector<std::string> out;
      const std::string s = csv.empty() ? std::string(fallback) : csv;

      size_t pos = 0;
      while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? s.size() : comma + 1;

        if (tok.rfind("fs.", 0) == 0 || tok.rfind("cs.", 0) == 0)
          tok = tok.substr(3);
        if (tok.size() >= 8)
          out.push_back(std::move(tok));
      }

      return out;
    }

    // name is "fs.<hash>..." or "cs.<hash>...": the hash starts at index 3
    bool NameMatchesTokens(const std::string& name, const std::vector<std::string>& tokens) {
      if (name.size() < 11)
        return false;
      for (const auto& tok : tokens) {
        if (name.compare(3, tok.size(), tok) == 0)
          return true;
      }
      return false;
    }

    // ---- cadence: BLESSED_VOL_PERIOD (default 2) allows up to period-1
    // skips in a row before forcing a run; DecideGenerate's motion/target/
    // camera gates can always force a run earlier still. Every 8 periods
    // run fully on schedule, one extra run is forced out of turn (two runs
    // back to back, no skip between), so the sampling walks by one frame
    // against the engine's 8-step jitter index: whatever period is chosen,
    // every jitter phase still gets a real generate run regularly, not
    // just the same subset every time. See NOTES-vol-period.md for the
    // phase table this was checked against (period=2 reproduces today's
    // behaviour bit for bit; period=3 was the requested case). ----

    struct Cadence {
      explicit Cadence(uint32_t p)
        : period(std::max(p, 1u))
        , skipStreak(period - 1)  // the first decision always runs
      { }

      uint32_t period;
      uint32_t skipStreak;
      uint32_t lapsSinceShift = 0;

      static constexpr uint32_t kShiftLaps = 8u;

      // True: the cadence would let this frame skip.
      bool AllowsSkip() const {
        return skipStreak < period - 1 && lapsSinceShift < kShiftLaps;
      }

      void Record(bool skipped) {
        if (skipped) {
          skipStreak++;
          return;
        }

        // on schedule: this run used up every skip the period allowed, so
        // it counts toward the next shift. forced early (motion/target/
        // camera, or the shift run itself) restarts the lap count: an
        // early run already moves the phase, as the shift would.
        bool onSchedule = skipStreak == period - 1;
        skipStreak = 0;
        lapsSinceShift = onSchedule ? lapsSinceShift + 1u : 0u;
      }
    };

    // ---- per-feature periodic jsonl file ----

    struct JsonlFile {
      std::ofstream file;
      bool          tried = false;
      dxvk::high_resolution_clock::time_point start;

      bool Open(const char* fileName) {
        if (!tried) {
          tried = true;
          start = dxvk::high_resolution_clock::now();
          std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
          if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            file.open(dir + env::PlatformDirSlash + fileName, std::ios::out | std::ios::app);
          }
        }
        return file.is_open();
      }

      double Seconds() const {
        return std::chrono::duration<double>(dxvk::high_resolution_clock::now() - start).count();
      }
    };

    // ============================ volumetrics ============================

    enum class VolRole : uint8_t { None, Generate, March, Blur, Apply };

    struct VolConfig {
      std::vector<std::string> generateCs;
      std::vector<std::string> marchCs;
      std::vector<std::string> blurCs;
      std::vector<std::string> applyPs;
      float    maxDeg        = 1.0f;
      float    maxMove       = 50.0f;
      float    screenMaxDeg  = 0.05f;
      float    screenMaxMove = 1.0f;
      bool     blind         = true;   // camera unreadable: still skip

      static const VolConfig& Get() {
        static const VolConfig s_config = [] {
          VolConfig c;
          c.generateCs    = SplitHexTokens(env::getEnvVar("BLESSED_VOL_HALFRATE_GEN_CS"), "ab674eb1");
          c.marchCs       = SplitHexTokens(env::getEnvVar("BLESSED_VOL_HALFRATE_MARCH_CS"), "1c4ebb62");
          c.blurCs        = SplitHexTokens(env::getEnvVar("BLESSED_VOL_HALFRATE_BLUR_CS"), "cf1e1a21,dbb99d2e");
          c.applyPs       = SplitHexTokens(env::getEnvVar("BLESSED_VOL_HALFRATE_APPLY_PS"), "c480e36e");
          c.maxDeg        = EnvFloat("BLESSED_VOL_HALFRATE_MAXDEG", c.maxDeg);
          c.maxMove       = EnvFloat("BLESSED_VOL_HALFRATE_MAXMOVE", c.maxMove);
          c.screenMaxDeg  = EnvFloat("BLESSED_VOL_HALFRATE_SCREEN_MAXDEG", c.screenMaxDeg);
          c.screenMaxMove = EnvFloat("BLESSED_VOL_HALFRATE_SCREEN_MAXMOVE", c.screenMaxMove);
          c.blind         = env::getEnvVar("BLESSED_VOL_HALFRATE_BLIND") != "0";

          std::string vol = env::getEnvVar("BLESSED_VOLUMETRICS");
          bool traced = !vol.empty() && vol != "off";
          Logger::info(str::format("blessed: vol-halfrate: level ", VolLevel(),
            (VolLevel() == 2u && traced) ? " (level 2 needs pass 138's draw, BLESSED_VOLUMETRICS is set: running level 1)" : "",
            ", period=", VolPeriod(),
            ", maxdeg=", c.maxDeg, " maxmove=", c.maxMove,
            " screen maxdeg=", c.screenMaxDeg, " maxmove=", c.screenMaxMove,
            " blind=", c.blind ? 1 : 0));
          return c;
        }();
        return s_config;
      }
    };

    // the generate's b0 as ISVolumetricLightingGenerateCS.hlsl lays it out
    constexpr uint32_t kCbViewProj     = 0u;      // c0..c3, row major
    constexpr uint32_t kCbViewProjInv  = 64u;     // c4..c7, row major
    constexpr uint32_t kCbPosAdjust    = 352u;    // c22.xyz
    constexpr uint32_t kCbIteration    = 364u;    // c22.w
    constexpr uint32_t kCbMinSize      = 368u;

    struct VolCamera {
      bool  valid     = false;
      float center[3] = { };  // unit view direction through the screen centre
      float corner[3] = { };  // unit direction through the top-right corner
      float nearPt[3] = { };  // a point just past the near plane, world space
      float iteration = -1.0f;
    };

    struct VolState {
      std::unordered_map<const void*, VolRole> roleCache;
      const void* lastCs     = nullptr;
      VolRole     lastCsRole = VolRole::None;
      const void* lastPs     = nullptr;
      bool        lastPsApply = false;

      Cadence   cadence{VolPeriod()};
      bool      skipActive  = false;   // this frame's latest generate was skipped
      uint64_t  lastCookie  = 0;       // the volume the last run wrote
      VolCamera lastCam;               // the camera the last run used
      VolCamera lastScreenCam;         // the camera pass 138 + its blur last ran with (level 2)

      // window counters
      uint64_t presents = 0, runs = 0, skips = 0, screenSkips = 0;
      uint64_t forcedCadence = 0, forcedTarget = 0, forcedMotion = 0, forcedCamera = 0;
      uint64_t skippedDispatches = 0, skippedDraws = 0, camOk = 0, camBad = 0;
      double   sumDeg = 0.0, maxDegSeen = 0.0, sumMove = 0.0;
      uint32_t iterMask = 0;
      uint32_t runIterMask = 0;        // jitter phases a real run landed on
      bool     warnedCam = false;
      JsonlFile log;
    };

    VolState g_volState;

    bool ReadCb(const D3D11ContextState& state, D3D11ShaderType stage, uint32_t slot,
                uint32_t minSize, const uint8_t** out) {
      const auto& cbvStage = state.cbv[stage];
      if (slot >= cbvStage.maxCount)
        return false;

      D3D11Buffer* buffer = cbvStage.buffers[slot].buffer.ptr();
      if (!buffer)
        return false;

      const uint8_t* mapPtr = reinterpret_cast<const uint8_t*>(buffer->GetMapPtr());
      if (!mapPtr)
        return false;

      UINT byteOffset = cbvStage.buffers[slot].constantOffset * 16u;
      if (byteOffset + minSize > buffer->Desc()->ByteWidth)
        return false;

      *out = mapPtr + byteOffset;
      return true;
    }

    // row-major m (HLSL row_major + mul(M, v)): r_i = dot(row i, v)
    void MulPoint(const float* m, const float* v, float* r) {
      for (uint32_t i = 0; i < 4; i++)
        r[i] = m[4*i+0]*v[0] + m[4*i+1]*v[1] + m[4*i+2]*v[2] + m[4*i+3]*v[3];
    }

    bool Unproject(const float* inv, float x, float y, float z, float* out) {
      float c[4] = { x, y, z, 1.0f };
      float r[4];
      MulPoint(inv, c, r);
      if (!std::isfinite(r[3]) || std::abs(r[3]) < 1.0e-12f)
        return false;
      for (uint32_t i = 0; i < 3; i++)
        out[i] = r[i] / r[3];
      return std::isfinite(out[0]) && std::isfinite(out[1]) && std::isfinite(out[2]);
    }

    bool Normalize(float* v) {
      float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
      if (!(len > 1.0e-9f))
        return false;
      for (uint32_t i = 0; i < 3; i++)
        v[i] /= len;
      return true;
    }

    bool Direction(const float* inv, float x, float y, float* dir) {
      float a[3], b[3];
      if (!Unproject(inv, x, y, 0.25f, a) || !Unproject(inv, x, y, 0.75f, b))
        return false;
      for (uint32_t i = 0; i < 3; i++)
        dir[i] = b[i] - a[i];
      return Normalize(dir);
    }

    VolCamera ReadCamera(const D3D11ContextState& state) {
      VolCamera cam;
      const uint8_t* cb = nullptr;
      if (!ReadCb(state, D3D11ShaderType::eCompute, 0u, kCbMinSize, &cb))
        return cam;

      float vp[16], inv[16], pos[3], iter;
      std::memcpy(vp,   cb + kCbViewProj,    sizeof(vp));
      std::memcpy(inv,  cb + kCbViewProjInv, sizeof(inv));
      std::memcpy(pos,  cb + kCbPosAdjust,   sizeof(pos));
      std::memcpy(&iter, cb + kCbIteration,  sizeof(iter));

      // layout check: vp * (inv * p) must give p back
      float p[4] = { 0.3f, -0.2f, 0.5f, 1.0f };
      float w[4], back[4];
      MulPoint(inv, p, w);
      MulPoint(vp, w, back);
      if (!std::isfinite(back[3]) || std::abs(back[3]) < 1.0e-12f)
        return cam;
      for (uint32_t i = 0; i < 3; i++) {
        if (!(std::abs(back[i] / back[3] - p[i]) < 1.0e-2f))
          return cam;
      }

      float nearA[3], nearB[3];
      if (!Direction(inv, 0.0f, 0.0f, cam.center)
       || !Direction(inv, 1.0f, 1.0f, cam.corner)
       || !Unproject(inv, 0.0f, 0.0f, 0.25f, nearA)
       || !Unproject(inv, 0.0f, 0.0f, 0.75f, nearB))
        return cam;

      // the nearer of the two points (depth may be reversed); PosAdjust is
      // added so a camera-relative matrix still shows the camera moving
      float* nearPt = (nearA[0]*nearA[0] + nearA[1]*nearA[1] + nearA[2]*nearA[2])
                    < (nearB[0]*nearB[0] + nearB[1]*nearB[1] + nearB[2]*nearB[2]) ? nearA : nearB;
      for (uint32_t i = 0; i < 3; i++)
        cam.nearPt[i] = nearPt[i] + (std::isfinite(pos[i]) ? pos[i] : 0.0f);

      cam.iteration = iter;
      cam.valid = true;
      return cam;
    }

    float AngleDeg(const float* a, const float* b) {
      float d = std::clamp(a[0]*b[0] + a[1]*b[1] + a[2]*b[2], -1.0f, 1.0f);
      return std::acos(d) * (180.0f / 3.14159265f);
    }

    float Distance(const float* a, const float* b) {
      float d[3] = { a[0]-b[0], a[1]-b[1], a[2]-b[2] };
      return std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    }

    VolRole ClassifyCs(const std::string& name) {
      const VolConfig& cfg = VolConfig::Get();
      if (NameMatchesTokens(name, cfg.generateCs)) return VolRole::Generate;
      if (NameMatchesTokens(name, cfg.marchCs))    return VolRole::March;
      if (NameMatchesTokens(name, cfg.blurCs))     return VolRole::Blur;
      return VolRole::None;
    }

    // one decision per generate dispatch
    bool DecideGenerate(const D3D11ContextState& state) {
      const VolConfig& cfg = VolConfig::Get();
      VolState& s = g_volState;

      uint64_t cookie = 0;
      if (state.uav.maxCount > 0u) {
        D3D11UnorderedAccessView* uav = state.uav.views[0].ptr();
        if (uav) {
          Rc<DxvkImageView> view = uav->GetImageView();
          if (view != nullptr && view->image() != nullptr)
            cookie = view->image()->cookie();
        }
      }

      VolCamera cam = ReadCamera(state);
      if (cam.valid) {
        s.camOk++;
        if (cam.iteration >= 0.0f && cam.iteration < 32.0f)
          s.iterMask |= 1u << uint32_t(cam.iteration);
      } else {
        s.camBad++;
        if (!s.warnedCam) {
          s.warnedCam = true;
          Logger::warn(str::format("blessed: vol-halfrate: the generate's b0 does not read as a camera; ",
            cfg.blind ? "skipping without the motion gate" : "running at full rate"));
        }
      }

      float deg = 0.0f, move = 0.0f;
      bool haveMotion = cam.valid && s.lastCam.valid;
      if (haveMotion) {
        deg  = std::max(AngleDeg(cam.center, s.lastCam.center), AngleDeg(cam.corner, s.lastCam.corner));
        move = Distance(cam.nearPt, s.lastCam.nearPt);
        s.sumDeg  += deg;
        s.sumMove += move;
        s.maxDegSeen = std::max(s.maxDegSeen, double(deg));
      }

      bool skip = true;
      if (!s.cadence.AllowsSkip()) {
        skip = false;
        s.forcedCadence++;
      } else if (cookie == 0 || cookie != s.lastCookie) {
        skip = false;
        s.forcedTarget++;
      } else if (haveMotion && (deg > cfg.maxDeg || move > cfg.maxMove)) {
        skip = false;
        s.forcedMotion++;
      } else if (!haveMotion && !cfg.blind) {
        skip = false;
        s.forcedCamera++;
      }

      s.cadence.Record(skip);
      s.skipActive = skip;

      // blessed: vol-period -- level 2 shows the blurred result of the last
      // frame pass 138 ran, so gate against that frame's camera. With
      // period 2 it is always the last run (== lastCam); with period 3+ a
      // skip frame can run the screen chain after the last run.
      bool screenOff = false;
      if (skip && blessed_vol_halfrate_detail::g_volScreen
       && cam.valid && s.lastScreenCam.valid) {
        float sdeg  = std::max(AngleDeg(cam.center, s.lastScreenCam.center),
                               AngleDeg(cam.corner, s.lastScreenCam.corner));
        float smove = Distance(cam.nearPt, s.lastScreenCam.nearPt);
        screenOff = sdeg <= cfg.screenMaxDeg && smove <= cfg.screenMaxMove;
      }
      blessed_vol_halfrate_detail::g_volScreenOff = screenOff;
      if (!screenOff)
        s.lastScreenCam = cam;

      if (skip) {
        s.skips++;
        if (blessed_vol_halfrate_detail::g_volScreenOff)
          s.screenSkips++;
      } else {
        s.runs++;
        if (cam.valid && cam.iteration >= 0.0f && cam.iteration < 32.0f)
          s.runIterMask |= 1u << uint32_t(cam.iteration);
        s.lastCookie = cookie;
        s.lastCam    = cam;
      }

      return skip;
    }

  }


  namespace blessed_vol_halfrate_detail {
    extern const bool g_volEnabled     = VolLevel() != 0u;
    extern const bool g_volScreen      = ComputeVolScreen();
    bool              g_volScreenOff    = false;
  }


  // ============================ volumetrics ============================

  bool BlessedVolHalfRate::ShouldSkipDispatchSlow(const D3D11ContextState& state) {
    VolState& s = g_volState;
    const void* cs = state.cs.ptr();
    if (!cs)
      return false;

    VolRole role;
    if (cs == s.lastCs) {
      role = s.lastCsRole;
    } else {
      auto it = s.roleCache.find(cs);
      if (it != s.roleCache.end()) {
        role = it->second;
      } else {
        role = ClassifyCs(state.cs->GetCommonShader()->GetName());
        s.roleCache.emplace(cs, role);
      }
      s.lastCs     = cs;
      s.lastCsRole = role;
    }

    bool skip = false;
    switch (role) {
      case VolRole::Generate: skip = DecideGenerate(state); break;
      case VolRole::March:    skip = s.skipActive; break;
      case VolRole::Blur:     skip = blessed_vol_halfrate_detail::g_volScreenOff; break;
      default: break;
    }

    if (skip)
      s.skippedDispatches++;
    return skip;
  }


  bool BlessedVolHalfRate::ShouldSkipDrawSlow(const D3D11ContextState& state) {
    VolState& s = g_volState;
    const void* ps = state.ps.ptr();
    if (!ps)
      return false;

    if (ps != s.lastPs) {
      auto it = s.roleCache.find(ps);
      bool apply;
      if (it != s.roleCache.end()) {
        apply = it->second == VolRole::Apply;
      } else {
        apply = NameMatchesTokens(state.ps->GetCommonShader()->GetName(), VolConfig::Get().applyPs);
        s.roleCache.emplace(ps, apply ? VolRole::Apply : VolRole::None);
      }
      s.lastPs      = ps;
      s.lastPsApply = apply;
    }

    if (s.lastPsApply)
      s.skippedDraws++;
    return s.lastPsApply;
  }


  void BlessedVolHalfRate::OnPresent() {
    VolState& s = g_volState;
    s.skipActive = false;
    blessed_vol_halfrate_detail::g_volScreenOff = false;

    if (++s.presents < 120u)
      return;

    if (s.log.Open("vol_halfrate.jsonl")) {
      double decisions = double(s.runs + s.skips);
      double moves = double(std::max<uint64_t>(s.camOk, 1u));
      s.log.file << str::format("{\"t\":", s.log.Seconds(),
        ",\"period\":", s.cadence.period,
        ",\"frames\":", s.presents,
        ",\"runs\":", s.runs,
        ",\"skips\":", s.skips,
        ",\"screen_skips\":", s.screenSkips,
        ",\"skip_share\":", decisions > 0.0 ? double(s.skips) / decisions : 0.0,
        ",\"forced_cadence\":", s.forcedCadence,
        ",\"forced_target\":", s.forcedTarget,
        ",\"forced_motion\":", s.forcedMotion,
        ",\"forced_camera\":", s.forcedCamera,
        ",\"skipped_dispatches\":", s.skippedDispatches,
        ",\"skipped_draws\":", s.skippedDraws,
        ",\"cam_ok\":", s.camOk,
        ",\"cam_bad\":", s.camBad,
        ",\"mean_deg\":", s.sumDeg / moves,
        ",\"max_deg\":", s.maxDegSeen,
        ",\"mean_move\":", s.sumMove / moves,
        ",\"iter_mask\":", s.iterMask,
        ",\"run_iter_mask\":", s.runIterMask,
        "}\n");
      s.log.file.flush();
    }

    s.presents = s.runs = s.skips = s.screenSkips = 0;
    s.forcedCadence = s.forcedTarget = s.forcedMotion = s.forcedCamera = 0;
    s.skippedDispatches = s.skippedDraws = s.camOk = s.camBad = 0;
    s.sumDeg = s.maxDegSeen = s.sumMove = 0.0;
    s.iterMask = s.runIterMask = 0;
  }


  // ======================= the shared camera signal =======================

  bool BlessedIsVolGenerateName(const std::string& name) {
    static const std::vector<std::string> s_tokens =
      SplitHexTokens(env::getEnvVar("BLESSED_VOL_HALFRATE_GEN_CS"), "ab674eb1");
    return NameMatchesTokens(name, s_tokens);
  }


  BlessedVolCameraPose BlessedReadVolCamera(const D3D11ContextState& state) {
    VolCamera cam = ReadCamera(state);
    BlessedVolCameraPose pose;
    pose.valid = cam.valid;
    std::memcpy(pose.center, cam.center, sizeof(pose.center));
    std::memcpy(pose.corner, cam.corner, sizeof(pose.corner));
    std::memcpy(pose.nearPt, cam.nearPt, sizeof(pose.nearPt));
    return pose;
  }


  void BlessedVolCameraMotion(const BlessedVolCameraPose& a, const BlessedVolCameraPose& b, float* deg, float* move) {
    // the same measure DecideGenerate uses
    *deg  = std::max(AngleDeg(a.center, b.center), AngleDeg(a.corner, b.corner));
    *move = Distance(a.nearPt, b.nearPt);
  }

}
