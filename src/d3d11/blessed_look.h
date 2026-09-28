// blessed: BLESSED_LOOK -- d3d11 side of the bless colour chain: finds skyrim's tonemap and final copy, patches and dispatches
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  class DxvkDevice;
  class D3D11ImmediateContext;

  // blessed: backing flag for BlessedLook::IsEnabled(), set once at
  // static-init time from BLESSED_LOOK alone (env only, no cross-TU
  // state) -- same pattern as blessed_hook_detail::g_enabled
  namespace blessed_look_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief The blessed look: bless's colour chain after skyrim's final copy
   *
   * BLESSED_LOOK=bless turns it on; anything else (or unset) is off, and
   * then every entry point is one inline read of a cached bool.
   *
   * Two draws are watched, matched by pixel shader name like BlessedHook
   * (full "fs.<hash>" name or an 8+ char hex prefix):
   *
   *  - the tonemap, BLESSED_LOOK_PS_TONEMAP (default 716590ec, pass 175).
   *    Before it draws, its ps b2 (ISHDR's PerGeometry: c2 Param, c3
   *    Cinematic, c4 Tint, c5 Fade) is patched in the mapped cbuffer:
   *    BLESSED_LOOK_VANILLA_BLOOM=0 zeroes Param.x (vanilla's bloom),
   *    BLESSED_LOOK_VANILLA_GRADE=0 sets saturation, contrast and
   *    brightness to 1 and the tint weight to 0. Fade is never touched.
   *  - the final copy, BLESSED_LOOK_PS_FINAL (default 831de5eb, pass 181).
   *    After it draws, the look pass runs on its render target 0, once
   *    per frame, before the hud draws on top.
   *
   * BLESSED_LOOK_DEBUG proves each hook point and shows each layer; see
   * blessed_look.cpp's header comment for the list and the expected frame.
   *
   * Every number is overridable by BLESSED_LOOK_<NAME>; the full list is
   * in blessed_look.cpp next to the profile defaults.
   */
  class BlessedLook {
  public:

    static bool IsEnabled() {
      return blessed_look_detail::g_enabled;
    }

    // pre-draw: patches the tonemap pass's ps b2
    static void OnPreDraw(const D3D11ContextState& state);

    // post-draw: after the final copy, runs the look pass on rtv 0
    static void OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state, DxvkDevice* device);

    // frame boundary: per-frame guards, the 120-present log line
    static void OnPresent();
  };

}
