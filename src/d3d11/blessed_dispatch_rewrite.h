// blessed: shader-replace, data-driven per-hash exact-match dispatch-size override for compacted compute replacements
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  namespace blessed_dispatch_rewrite_detail {
    extern const bool g_enabled;
  }

  /**
   * \brief Rewrites a dispatch's group counts when a narrowly matched replacement shader is bound
   *
   * Some shader-replace rewrites change which groups the game must launch
   * to cover the same output, not just the shader's math, because the
   * replacement maps more than one column/row per group (e.g. blur3,
   * cs.dbb99d2e...: one group per COLS=4 texel columns instead of one group
   * per column), or reshapes numthreads while keeping every invocation's
   * coordinates the same (e.g. volgen-8x8, cs.ab674eb1...: 8x8x1 groups
   * instead of 32x32x1, same total dispatch-thread space). Applying that
   * shape change on a bytecode swap alone would silently corrupt any
   * dispatch it did not anticipate, so this table only fires on an *exact*
   * (bound shader name, X, Y, Z) match -- anything else, including any
   * other dispatch of the same replaced shader, keeps the game's own
   * dispatch untouched.
   *
   * Rules are loaded from BLESSED_SHADER_REPLACE's directory: any file
   * named `<name>.dispatch` beside `<name>.dxbc` holds one or more lines
   * `matchX matchY matchZ -> newX newY newZ`. A shader with no sidecar (or
   * one that fails to parse) keeps the game's own dispatch untouched. With
   * no live rule anywhere, IsEnabled() is a single cached-bool read and
   * Dispatch() does nothing else.
   */
  class BlessedDispatchRewrite {
  public:
    static bool IsEnabled() {
      return blessed_dispatch_rewrite_detail::g_enabled;
    }

    // Pre-dispatch, called only when IsEnabled(). Rewrites x/y/z in place
    // if the currently bound compute shader and the requested group counts
    // match a known entry; otherwise leaves them untouched.
    static void Adjust(
      const D3D11ContextState& state,
            UINT&               x,
            UINT&               y,
            UINT&               z);
  };

}
