// blessed: generic post-draw hook, fired after a game draw whose pixel shader matches BLESSED_HOOK_PS
#pragma once

#include <cstdint>
#include <functional>

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;

  // blessed: backing flag for BlessedHook::IsEnabled(), split out so the
  // hot-path gate (checked after every draw) is a trivial inline read --
  // same pattern as blessed_dump_detail::g_enabled in blessed_dump.h. Set
  // once at static-init time from BLESSED_HOOK_PS/BLESSED_HOOK_MODE alone
  // (no cross-TU state -- the callback wiring stays in EnsureInit(), lazy,
  // for the reason its own comment gives: static init order across files
  // isn't defined, and a global constructor's callback assignment could
  // still be clobbered by another TU's).
  namespace blessed_hook_detail {
    extern const bool g_enabled;
  }

  /// blessed: bound render target seen at a matching draw
  struct BlessedHookRtv {
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    uint32_t    w   = 0;
    uint32_t    h   = 0;
  };

  /// blessed: what a matching draw had bound, handed to the hook callback
  struct BlessedHookTargets {
    BlessedHookRtv rtv0;       // render target 0 (the one `fill` clears)
    bool           hasRtv0 = false;
    bool           hasDsv  = false;

    // blessed: shadow-pass widening (2026-09-22) -- the actual bound views,
    // for a callback that needs to read/write real image data rather than
    // just describe it, and the full context state (cbuffers, srvs) needed
    // to pull the camera matrices and an optional depth SRV override. Valid
    // only for the duration of the callback call.
    Rc<DxvkImageView>        rtv0View;
    Rc<DxvkImageView>        dsvView;
    const D3D11ContextState* state = nullptr;
  };

  using BlessedHookCallback =
    std::function<void(D3D11ImmediateContext*, const BlessedHookTargets&)>;

  /**
   * \brief Generic "run something after a chosen draw" hook
   *
   * Enabled by BLESSED_HOOK_PS: a comma-separated list of pixel shader
   * names (as dumped by BlessedDump's "ps" field, e.g.
   * "fs.1191b64b314d79f041c33c147ec8a450") or hex prefixes of 8+ chars
   * taken from after the "fs." part of that name.
   *
   * BLESSED_HOOK_MODE selects what happens on a match:
   *  - "log":  count matches, write one summary line per 120 presents to
   *            "<BLESSED_PROBE_DIR>/hook.jsonl".
   *  - "fill": also clear render target 0 to BLESSED_HOOK_FILL (default
   *            "1,1,1,1"), via the context's own ClearRenderTargetView.
   *  - "off" or unset BLESSED_HOOK_PS: disabled, zero cost.
   *
   * BLESSED_HOOK_ONCE=1 acts on only the first match per frame.
   *
   * SetCallback() lets a later pass (the ray-query hook) replace the
   * `fill` behavior; match counting for "log" still runs either way.
   *
   * All entry points are no-ops (one cached-bool check) unless
   * BLESSED_HOOK_PS is set.
   */
  class BlessedHook {
  public:

    // Cheap, cached: true once BLESSED_HOOK_PS is set to a non-empty value.
    // Inline read of a plain global -- no out-of-line call.
    static bool IsEnabled() {
      return blessed_hook_detail::g_enabled;
    }

    // Called from the immediate context right after a draw call has been
    // recorded. Pays one pointer-cache lookup on the bound pixel shader;
    // does nothing unless that shader matches a configured hash.
    static void OnDraw(D3D11ImmediateContext* ctx, const D3D11ContextState& state);

    // Frame boundary; rolls the per-120-present log line.
    static void OnPresent();

    // Replaces the default ("fill", or nothing in "log"-only mode)
    // callback invoked on a match.
    static void SetCallback(BlessedHookCallback cb);
  };

}
