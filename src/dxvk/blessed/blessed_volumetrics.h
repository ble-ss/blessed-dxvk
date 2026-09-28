// blessed: ray-traced volumetric sun light (BLESSED_VOLUMETRICS) -- dispatch arguments for the dxvk side
#pragma once

#include <cstdint>
#include <string>

#include "blessed_rt.h"

#include "../dxvk_image.h"

namespace dxvk {

  /**
   * \brief What one volumetrics hook firing asks the cs thread to do
   *
   * Filled on the d3d11 app thread right after skyrim's
   * ISApplyVolumetricLighting draw (pass 138, ps c480e36e), captured by
   * value into an EmitCs lambda, run by DxvkContext::blessedRunVolumetricsPass.
   * Every field must be safe to copy and to read a frame later.
   *
   * Pass 138 writes a scalar r16f "vl power" target. Vanilla then blurs it
   * (two compute passes, h then v) and pass 167 adds
   * VolumetricLightingColor * vl into the hdr scene. Writing our own
   * in-scatter into that same target keeps the blur and the composite, and
   * so the game's own colour, untouched.
   */
  struct BlessedVolDispatchArgs {
    enum class Op : uint32_t {
      Trace = 0,  // BLESSED_VOLUMETRICS=rt: trace + temporal + upsample into the target
      DumpOnly,   // vanilla / fill: leave the target alone, only dump it this frame
      // blessed: async-compute (BLESSED_ASYNC=1, pass "vol"). AsyncTrace is
      // emitted at the kick draw (the sao composite, pass 127): copy depth
      // on graphics, trace + temporal on the async queue. AsyncUpsample is
      // emitted at pass 138 instead of Trace when that kick is still valid:
      // wait, then upsample into the target on graphics. Either one falls
      // back to the plain path when the device has no async queue.
      AsyncTrace,
      AsyncUpsample,
      // blessed: vol-2 -- pass 138's draw was skipped and no trace can run
      // (no tlas): zero the target instead of leaving last frame's image
      Clear,
    };

    Op op = Op::Trace;

    // pass 138's own depth srv (BLESSED_VOL_DEPTH, default srv:0), a
    // depth-aspect SAMPLED view
    Rc<DxvkImageView> depthView;

    // pass 138's rtv 0 (r16f): a STORAGE view for the trace op, and a
    // SAMPLED view for the dump
    Rc<DxvkImageView> outputStorageView;
    Rc<DxvkImageView> outputSampledView;

    uint32_t width  = 0;
    uint32_t height = 0;

    // CameraViewProjInverse, raw row-major cbuffer bytes (same convention as
    // BlessedShadowDispatchArgs::invViewProj: the shader uses `v * mat`)
    float invViewProj[16] = { };

    // CameraPosAdjust at this draw. Used for the tlas shift (resolved on the
    // cs thread against the current tlas, like the shadow pass) and for our
    // own reprojection (history camera vs this camera).
    float camPosNow[3] = { 0.0f, 0.0f, 0.0f };

    // unit vector toward the sun, world space (the shadow projection's
    // depth axis at the mask draw, BLESSED_VOL_SUNDIR)
    float sunDir[3] = { 0.3f, 0.4f, 0.866f };

    float farDepthValue = 1.0f;

    // the game's own g_IntensityX_TemporalY.x (pass 138 ps b2 c0.x), so the
    // weather and time of day keep driving the strength; times our gain
    float gameIntensity = 1.0f;

    // tuning (env, read once on the app thread)
    // blessed: vol-2 -- defaults recalibrated after vol-rt-2 (gain 40 blew
    // the sky toward the sun out to white): a lower gain plus a soft
    // shoulder, out = x * max / (max + x), so the forward-scatter peak rolls
    // off instead of clipping. BLESSED_VOL_GAIN=auto steers the gain so our
    // mean matches vanilla's pass 138 output (needs the vanilla draw).
    float    gain          = 24.0f;   // BLESSED_VOL_GAIN (a number, or "auto")
    bool     autoGain      = false;
    float    maxOut        = 1.0f;    // BLESSED_VOL_MAX, 0 = no shoulder
    uint32_t statsEvery    = 30u;     // BLESSED_VOL_STATS_EVERY, 0 = never; auto: 4
    float    phaseG        = 0.6f;    // BLESSED_VOL_G
    float    density       = 8.6e-5f; // BLESSED_VOL_DENSITY, per skyrim unit
    float    heightFalloff = 1.0e-3f; // BLESSED_VOL_HEIGHT_FALLOFF, per unit above the base
    float    heightBase    = -420.0f; // BLESSED_VOL_HEIGHT_BASE, camera-relative
    float    heightFloor   = 0.35f;   // BLESSED_VOL_HEIGHT_FLOOR, density kept high up
    float    range         = 20000.0f;// BLESSED_VOL_RANGE, march length for sky pixels
    float    sunRayLength  = 30000.0f;// BLESSED_VOL_SUN_RAY
    float    historyWeight = 0.9f;    // BLESSED_VOL_HISTORY
    float    rejectRel     = 0.05f;   // BLESSED_VOL_REJECT
    uint32_t steps         = 4u;      // BLESSED_VOL_STEPS
    uint32_t resDivisor    = 2u;      // BLESSED_VOL_RES
    uint32_t debugMode     = 0u;      // BLESSED_VOL_DEBUG (see blessed_vol_trace.comp)

    // blessed: vol-2 -- pass 138's draw was skipped: the target holds last
    // frame's image, so vanilla stats are meaningless and a pass that cannot
    // trace must clear it
    bool     vanillaSkipped = false;

    // blessed: vol-2 -- app-thread time spent in the pass-138 hook, logged
    uint64_t appNs = 0;

    // BLESSED_VOL_DUMP: blocking readback of the target after this op, to a
    // .pgm at dumpPath, plus one stats line in the log. One frame only.
    bool        dump      = false;
    float       dumpScale = 128.0f;   // BLESSED_VOL_DUMP_SCALE: byte = value * scale
    std::string dumpPath;
    std::string dumpTag;

    // resolved on the cs thread (DxvkContext::blessedRunVolumetricsPass),
    // same reasoning as BlessedShadowDispatchArgs's tlas fields
    uint64_t               tlasAddress = 0;
    bool                   tlasValid   = false;
    float                  camDelta[3] = { 0.0f, 0.0f, 0.0f };
    Rc<BlessedAccelStruct> tlasRef;
  };

}
