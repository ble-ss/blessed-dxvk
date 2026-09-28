// blessed: app-thread half of the ray-traced sun shadow pass -- config, cbuffer reads, dispatch handoff
#pragma once

#include "blessed_hook.h"

namespace dxvk {

  class D3D11ImmediateContext;

  /**
   * \brief Reads the shadow-mask draw's bound state and hands a dispatch to the cs thread
   *
   * Installed as \ref BlessedHook's callback only when
   * `BLESSED_HOOK_MODE=rtshadow` (checked once, at static init, before any
   * draw can happen -- see blessed_shadow.cpp's installer). \ref OnDraw
   * itself re-checks \ref IsEnabled so this stays cheap (one cached bool)
   * if the hook is active for some other consumer instead.
   */
  class BlessedShadow {
  public:

    // Cheap, cached: true once BLESSED_HOOK_MODE=rtshadow.
    static bool IsEnabled();

    // BlessedHookCallback signature. See blessed_hook.h.
    static void OnDraw(D3D11ImmediateContext* ctx, const BlessedHookTargets& targets);

  };

}
