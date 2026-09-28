// blessed: perf-halfrate -- vanilla volumetric light at half rate
#pragma once

#include <cstdint>
#include <string>

#include "d3d11_context_state.h"

namespace dxvk {

  // blessed: backing flags, split out so the hot-path gates are one inline
  // read each (same pattern as blessed_skip_replaced_detail). The const
  // ones are set once at static init from env vars alone; the plain
  // bool flips at most twice per frame.
  namespace blessed_vol_halfrate_detail {
    extern const bool g_volEnabled;      // BLESSED_VOL_HALFRATE=1 or 2
    extern const bool g_volScreen;       // BLESSED_VOL_HALFRATE=2: pass 138 + its blur too
    extern bool       g_volScreenOff;    // this frame skips pass 138 and its blur (level 2)
  }

  /**
   * \brief Vanilla volumetric lighting at half rate (BLESSED_VOL_HALFRATE)
   *
   * The whiterun dump (bench/runs/whiterun-dump-1, frame 1920) shows the
   * chain: one generate dispatch (cs `ab674eb1`, 10x6x90 groups) fills two
   * 320x192x90 r16f volumes, 90 raymarch dispatches (cs `1c4ebb62`)
   * ping-pong between them and leave the result in the first one. Pass
   * 138's draw (ps `c480e36e`) reads that volume plus its own previous
   * blurred output, the two blur dispatches (`cf1e1a21`, `dbb99d2e`)
   * write the blurred result, and a lens-flare draw (`69c92902`) also
   * reads the volume. Nothing else writes the volumes, so on a frame that
   * skips the 91 dispatches the volume still holds the previous frame's
   * result: pass 138 and the lens flare read valid data.
   *
   * Level 1 (=1) skips the generate and raymarch dispatches on frames the
   * cadence allows. Level 2 (=2) also skips pass 138's draw and the two
   * blur dispatches on those frames, under a tighter motion gate, so the
   * composite shows the last blurred result; that gate is measured from
   * the camera of the last frame pass 138 ran, not the last generate run.
   *
   * BLESSED_VOL_PERIOD (default 2) sets the cadence: the chain runs at
   * least once every `period` eligible frames (period=2 is today's
   * alternate-frame default; period=1 disables the skip). A frame runs
   * out of turn when: the volume image changed, or the camera moved more
   * than BLESSED_VOL_HALFRATE_MAXDEG degrees or MAXMOVE units since the
   * last run, or every 8th on-schedule period (so the engine's 8-step
   * jitter index still gets a real run at every phase, whatever period is
   * chosen -- see NOTES-vol-period.md for the phase table). The camera
   * comes from the generate's own cbuffer (b0: CameraViewProj c0, its
   * inverse c4, PosAdjust c22), the layout vendor/community-shaders'
   * ISVolumetricLightingGenerateCS.hlsl gives.
   */
  class BlessedVolHalfRate {
  public:
    static bool IsEnabled() {
      return blessed_vol_halfrate_detail::g_volEnabled;
    }

    // Pre-dispatch, immediate context only. True: drop this dispatch.
    static bool ShouldSkipDispatch(const D3D11ContextState& state) {
      return IsEnabled() && ShouldSkipDispatchSlow(state);
    }

    // Pre-draw, immediate context only. True: drop pass 138's draw (level 2).
    static bool ShouldSkipDraw(const D3D11ContextState& state) {
      return blessed_vol_halfrate_detail::g_volScreenOff
          && ShouldSkipDrawSlow(state);
    }

    // Frame boundary. Writes <BLESSED_PROBE_DIR>/vol_halfrate.jsonl every 120 presents.
    static void OnPresent();

  private:
    static bool ShouldSkipDispatchSlow(const D3D11ContextState& state);
    static bool ShouldSkipDrawSlow(const D3D11ContextState& state);
  };

  /**
   * \brief The volumetric generate's camera, as the gate above reads it
   *
   * Shared with blessed_reflect_halfrate.cpp so both half rates gate on
   * one motion signal. Works whether or not BLESSED_VOL_HALFRATE is set.
   */
  struct BlessedVolCameraPose {
    bool  valid     = false;
    float center[3] = { };  // unit view direction through the screen centre
    float corner[3] = { };  // unit direction through the top-right corner
    float nearPt[3] = { };  // a point just past the near plane, world space
  };

  /// True for the generate's cs name (BLESSED_VOL_HALFRATE_GEN_CS, default ab674eb1)
  bool BlessedIsVolGenerateName(const std::string& name);

  /// The camera from the bound generate dispatch's b0; invalid if the layout check fails
  BlessedVolCameraPose BlessedReadVolCamera(const D3D11ContextState& state);

  /// Turn in degrees (the larger of the centre and corner rays) and move in units
  void BlessedVolCameraMotion(const BlessedVolCameraPose& a, const BlessedVolCameraPose& b, float* deg, float* move);

}
