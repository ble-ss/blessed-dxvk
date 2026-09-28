// blessed: ray-traced sun shadow pass -- dispatch arguments and the dxvk-side compute pipeline
#pragma once

#include <cstdint>
#include <string>

#include "blessed_rt.h"

#include "../dxvk_image.h"

namespace dxvk {

  class DxvkContext;
  class DxvkCommandList;

  /**
   * \brief Everything DxvkContext::blessedRunShadowPass needs for one dispatch
   *
   * Most fields are filled on the d3d11 app thread, inside the BlessedHook
   * callback, where the bound views and cbuffer maps are still valid --
   * see src/d3d11/blessed_shadow.cpp. Captured by value into the EmitCs
   * lambda that carries it to the cs thread, so every field here must be
   * safe to copy and to read a frame later than it was filled. The tlas
   * fields and camDelta are the exception -- see the comment on them below.
   */
  struct BlessedShadowDispatchArgs {
    // depth input: a depth-aspect SAMPLED view of whatever the d3d11 side
    // resolved (bound dsv, or the BLESSED_SHADOW_DEPTH=srv:<slot> override)
    Rc<DxvkImageView> depthView;

    // output: a STORAGE view of rtv 0, same format, full extent. Only valid
    // (and only ever set) when the underlying image was actually created
    // with VK_IMAGE_USAGE_STORAGE_BIT -- see outputStorageCapable.
    Rc<DxvkImageView> outputView;
    bool              outputStorageCapable = false;

    uint32_t width  = 0;
    uint32_t height = 0;

    // CameraViewProjInverse, read raw off the bound cbuffer: 16 floats,
    // row-major as skyrim's shaders store it. Loaded directly into a GLSL
    // mat4 (column-major read, so the shader sees M's transpose) and used
    // as `v * mat`, which reproduces skyrim's `mul(M, v)`.
    float invViewProj[16] = { };

    // blessed: this frame's raw camera position (BLESSED_SHADOW_CAMPOS,
    // default ps:12:640), read on the app thread while the cbuffer map is
    // still valid -- NOT a delta yet. \ref camDelta below is resolved
    // cs-side, against whichever tlas frame is actually current when this
    // dispatch runs; see the comment there.
    float camPosNow[3] = { 0.0f, 0.0f, 0.0f };

    // unit vector toward the sun, world space (BLESSED_SHADOW_SUNDIR)
    float sunDir[3] = { 0.3f, 0.4f, 0.866f };

    // depth value that means "sky, skip" -- 0.0 or 1.0 depending on
    // BLESSED_SHADOW_FAR (reversed-z vs. regular z)
    float farDepthValue = 1.0f;

    // blessed: BLESSED_SHADOW_SOFT=1 -- soft, temporally-accumulated shadows
    // (see blessed_soft_shadow.h). Every field below this point is only
    // ever read when softEnabled is true; the hard-shadow path (default)
    // never touches them.
    bool softEnabled = false;

    // CameraPreviousViewProjUnjittered, read raw off the bound cbuffer the
    // same way as invViewProj above (BLESSED_SHADOW_PREVVP, default
    // ps:12:256). Only used with BLESSED_SHADOW_REPROJ=game (a/b and the
    // reproj debug log); the default reprojects with the matrix of the
    // dispatch that wrote the history instead (fork-shadow-slide.md).
    float prevViewProj[16] = { };

    // this frame's camPosNow minus *last frame's* camPosNow (both raw
    // BLESSED_SHADOW_CAMPOS reads, one app-thread frame apart) -- shifts a
    // position relative to this frame's camera into last frame's camera's
    // frame of reference, the way camDelta above does for the tlas. Zero
    // on the first frame after enable; the dxvk-side object forces a full
    // history reset that frame regardless (BlessedSoftShadowState).
    // Pairs with prevViewProj: BLESSED_SHADOW_REPROJ=game only.
    float camReprojDelta[3] = { 0.0f, 0.0f, 0.0f };

    // BLESSED_SHADOW_SUN_DEG (default 0.53, degrees), converted to radians
    float sunHalfAngleRad = 0.0f;

    // BLESSED_SHADOW_SPP, clamped to [1,4]
    uint32_t spp = 1;

    // blessed: everything below this point is resolved by
    // DxvkContext::blessedRunShadowPass itself, on the cs thread, at the
    // moment this dispatch actually runs -- BlessedShadow::OnDraw (the app
    // thread) leaves them at their defaults. They used to be filled on the
    // app thread, snapshotting BlessedScene::currentFrame() at draw-record
    // time; since the cs thread can lag the app thread by a frame or more,
    // that snapshot could name a ping-pong tlas slot endFrame() was about
    // to rebuild before this dispatch's cs-thread turn came up, which is
    // what caused the sun mask to flicker frame to frame in a static scene
    // (fork-cam-cascades.md). camDelta is camPosAdjust(now) -
    // camPosAdjust(tlas build): shifts this frame's camera-relative
    // position into the frame the tlas was actually built from.
    uint64_t tlasAddress = 0;
    bool     tlasValid   = false;
    float    camDelta[3] = { 0.0f, 0.0f, 0.0f };

    // blessed: the tlas object \ref tlasAddress was read from, so the
    // dxvk-side dispatch can track it (see DxvkContext::blessedRunShadowPass)
    // and keep it -- and every blas it references -- alive until this
    // dispatch's GPU work is done, independent of the scene cache's own
    // eviction or the ping-pong slot getting rebuilt out from under it.
    Rc<BlessedAccelStruct> tlasRef;

    // BLESSED_SHADOW_DUMP: request a readback of this dispatch's visibility
    // buffer to a .pgm at dumpPath. Set for exactly one frame (the 120th
    // after enable); the dxvk side does the (blocking, debug-only) readback.
    bool        dump = false;
    std::string dumpPath;
  };

}
