// blessed: ray-traced ambient occlusion written into skyrim's own sao texture -- dispatch arguments
#pragma once

#include <cstdint>

#include "blessed_rt.h"

#include "../dxvk_image.h"

namespace dxvk {

  /**
   * \brief BLESSED_AO_DEBUG views, shared by the d3d11 and dxvk halves
   */
  enum class BlessedAoDebug : uint32_t {
    None    = 0,  // the real thing
    Black   = 1,  // force the sao texture's x to 0 (proof: frame goes black where sao applies)
    White   = 2,  // force it to 1 (proof: vanilla's ao disappears)
    Raw     = 3,  // this frame's rays only: no history, no blur
    NoBlur  = 4,  // rays + history, no blur
    Hist    = 5,  // history length, 0 = just reset, 1 = full (16 frames)
    Normal  = 6,  // reconstructed normal facing the camera (flat = even grey, edges darker)
    Reproj  = 7,  // reprojection error, 16 per pixel of slide (see blessed_ao_trace.comp)
  };

  /**
   * \brief Everything DxvkContext::blessedRunAoPass needs for one dispatch
   *
   * Filled on the d3d11 app thread by BlessedAo::OnDraw, right before the
   * sao composite draw (skyrim's ISSAOComposite, ps ddcce9bc), while that
   * draw's srvs and cbuffers are bound. Captured by value into the EmitCs
   * lambda. The tlas fields are resolved on the cs thread, at dispatch
   * time, the same way BlessedShadowDispatchArgs resolves them.
   */
  struct BlessedAoDispatchArgs {
    // depth (sampled, depth aspect) -- the composite's own depth srv (t2)
    Rc<DxvkImageView> depthView;

    // the sao texture the composite is about to read (its srv t1), as a
    // storage view. Only the x channel is written; y/z/w stay vanilla's.
    Rc<DxvkImageView> outputView;

    uint32_t width  = 0;
    uint32_t height = 0;

    // CameraViewProjInverse (ps b12 byte 512), raw row-major bytes, same
    // convention as the shadow pass
    float invViewProj[16] = { };

    // CameraPosAdjust (ps b12 byte 640)
    float camPosNow[3] = { 0.0f, 0.0f, 0.0f };

    float    farDepthValue = 1.0f;
    float    radius        = 64.0f;  // BLESSED_AO_RADIUS, game units
    float    strength      = 1.0f;   // BLESSED_AO_STRENGTH
    uint32_t rays          = 2u;     // BLESSED_AO_RAYS, [1,16]
    bool     halfRes       = true;   // BLESSED_AO_HALF (default 1)
    BlessedAoDebug debug   = BlessedAoDebug::None;

    // blessed: resolved on the cs thread (DxvkContext::blessedRunAoPass)
    uint64_t tlasAddress = 0;
    bool     tlasValid   = false;
    float    camDelta[3] = { 0.0f, 0.0f, 0.0f };
    Rc<BlessedAccelStruct> tlasRef;
  };

}
