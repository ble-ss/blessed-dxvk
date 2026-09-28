// blessed: refl-harden -- the water reflection cube's faces at half rate, gated on camera motion and face age
#pragma once

#include <cstdint>

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11RenderTargetView;
  class D3D11DepthStencilView;

  // blessed: backing flags for the inline gates below. g_enabled is fixed
  // at static init from the env; g_watch is true while the feature is on
  // and live (the plugin's switch), written only at present.
  namespace blessed_reflect_halfrate_detail {
    extern const bool g_enabled;
    extern bool       g_watch;
  }

  /**
   * \brief The water reflection cube at half rate (BLESSED_REFLECT_HALFRATE=1)
   *
   * Skyrim renders the exterior water reflection into one 512x512 rgba16f
   * cube (6 layers, a d24s8 depth shared by all faces), two faces a frame,
   * round-robin: each face is redrawn every third frame. Every face pass
   * is ClearRenderTargetView(face), ClearDepthStencilView(depth), then the
   * draws; the first draw is the sky (ps 79c17b07).
   *
   * Recognition: the cube is learned, by image identity (dxvk cookie),
   * from a draw with ps 79c17b07 (BLESSED_REFLECT_HALFRATE_PS) whose only
   * target is one layer of a 6-layer cube-compatible rgba16f square image
   * and whose depth is a single-layer image of the same size. Skipping
   * starts once the same cube and depth have been seen in 3 frames. After
   * that, only draws and colour clears whose target is a layer of that
   * image, with that depth bound (draws), are ever dropped. Depth clears
   * are never dropped.
   *
   * Decision, once per face pass (the face's first clear or draw of the
   * frame). The face renders when: the previous frame skipped a face
   * (never two skipping frames in a row), the camera moved or turned more
   * than the thresholds in the last measured frame, the camera could not
   * be read, or the face is BLESSED_REFLECT_HALFRATE_MAXAGE frames old or
   * older. The default age cap of 6 means a face skipped on one visit is
   * always drawn on the next, so no face ever shows more than one missed
   * update.
   *
   * The motion signal is the volumetric half rate's (the generate's
   * camera, see BlessedVolCameraPose). The cube is the first pass of a
   * frame, so the gate uses the motion of the frame before.
   *
   * All entry points: immediate context only.
   */
  class BlessedReflectHalfRate {
  public:

    static bool IsEnabled() {
      return blessed_reflect_halfrate_detail::g_enabled;
    }

    /// One predictable branch per draw: false whenever there is nothing to do
    static bool IsWatching() {
      return blessed_reflect_halfrate_detail::g_watch;
    }

    /// Before a draw. True: drop this draw.
    static bool OnDraw(const D3D11ContextState& state);

    /// Before a colour clear (ClearRenderTargetView or ClearView). True: drop it.
    static bool OnClearRtv(D3D11RenderTargetView* rtv);

    /// Before a dispatch: samples the camera from the volumetric generate
    static void OnDispatch(const D3D11ContextState& state);

    /// Frame boundary: reads the switch, decides the next frame, logs every ~5 s
    static void OnPresent();

  };

}
