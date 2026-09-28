// blessed: ray-traced point-light shadow pass (tier a) -- dispatch arguments, one per point mask draw
#pragma once

#include <cstdint>
#include <string>

#include "blessed_rt.h"

#include "../dxvk_image.h"

namespace dxvk {

  /**
   * \brief Everything DxvkContext::blessedRunPointShadowPass needs for one light
   *
   * Filled on the d3d11 app thread at a point mask draw (see
   * src/d3d11/blessed_point_shadow.cpp) and carried by value to the cs
   * thread, the same way as BlessedShadowDispatchArgs. The tlas fields at
   * the end are resolved on the cs thread at dispatch time, for the reason
   * BlessedShadowDispatchArgs gives.
   */
  struct BlessedPointShadowDispatchArgs {
    // depth: a depth-aspect sampled view (ps srv 2 by default, the t2 the
    // vanilla mask shader rebuilds positions from)
    Rc<DxvkImageView> depthView;

    // the shadow mask: a storage view of rtv 0 (rgba8)
    Rc<DxvkImageView> outputView;

    uint32_t width  = 0;
    uint32_t height = 0;

    // CameraViewProjInverse (ps b12 byte 512 by default), row-major as stored
    float invViewProj[16] = { };

    // raw camera position (ps b12 byte 640), resolved to camDelta cs-side
    float camPosNow[3] = { 0.0f, 0.0f, 0.0f };

    // the light's origin, camera-relative (solved from ShadowMapProj[0])
    float lightPos[3] = { 0.0f, 0.0f, 0.0f };

    // pixels farther from the light than this are written 1 (the lit
    // shader zeroes the light there anyway); 0 = no cull (spot lights,
    // whose c3.x is a cone falloff, not a radius)
    float radius = 0.0f;

    // tMax = distance - proxyRadius: the light's own lantern or sconce must
    // not shadow it (BLESSED_POINT_PROXY)
    float proxyRadius = 24.0f;

    float farDepthValue = 1.0f;

    // which mask channels this light owns: the draw's colour write mask
    // (bit 0 r .. bit 3 a)
    uint32_t channelMask = 0u;

    // BLESSED_POINT_DEBUG, see blessed_point_shadow.comp
    uint32_t debugMode = 0u;

    // blessed: soft point shadows (BLESSED_POINT_SOFT=1) -- a sphere light
    // of BLESSED_POINT_LIGHT_RADIUS units, the sun's temporal filter
    bool     softEnabled = false;
    float    lightRadius = 4.0f;
    uint32_t spp         = 1u;

    // app-thread present count, rolls the gpu timing window cs-side
    uint64_t frameId = 0;

    // BLESSED_POINT_DUMP: read this dispatch's channel back to a .pgm
    bool        dump = false;
    std::string dumpPath;

    // resolved on the cs thread (see DxvkContext::blessedRunPointShadowPass)
    uint64_t tlasAddress = 0;
    bool     tlasValid   = false;
    float    camDelta[3] = { 0.0f, 0.0f, 0.0f };
    Rc<BlessedAccelStruct> tlasRef;
  };

}
