// blessed: gpu track step 2 -- stop drawing/dispatching vanilla work our own traced features already overwrite
#pragma once

#include <cstdint>

#include "d3d11_context_state.h"

namespace dxvk {

  class D3D11ImmediateContext;

  // blessed: backing flags, split out so the hot-path gate (checked on every
  // draw/dispatch) is a trivial inline read -- same pattern as
  // blessed_cascades_detail::g_enabled. Each is set once at static-init
  // time from env vars alone; see blessed_skip_replaced.cpp for exactly
  // which ones and why each default landed where it did.
  namespace blessed_skip_replaced_detail {
    extern const bool g_aoEnabled;           // BLESSED_AO=rt && BLESSED_AO_SKIP_VANILLA
    extern const bool g_volDrawEnabled;      // BLESSED_VOLUMETRICS=rt && BLESSED_VOL_SKIP_VANILLA
    extern const bool g_volDispatchEnabled;  // above, plus explicit BLESSED_VOL_SKIP_RAYMARCH=1 opt-in
    extern const bool g_bloomEnabled;        // BLESSED_LOOK=bless, bloom zeroed, && BLESSED_LOOK_SKIP_BLOOM
    extern const bool g_shadowMaskEnabled;   // BLESSED_HOOK_MODE=rtshadow && BLESSED_SHADOW_SKIP_MASK_DRAW
  }

  /**
   * \brief Skips the sao raw draw and its two blurs once BLESSED_AO=rt has overwritten their target
   *
   * The whiterun dump (bench/runs/whiterun-dump-1) confirms a clean
   * single-consumer chain: the raw sao draw (ps BLESSED_AO_SKIP_PS[0],
   * default `83560015`) feeds only its horizontal blur (`aebce6eb`), which
   * feeds only its vertical blur (`e205894f`), which feeds only
   * `ISSAOComposite` (`ddcce9bc`, pass 127) -- the exact texture
   * BlessedAo::OnDrawPre overwrites at BLESSED_AO_SRV. All three are dead
   * work once BLESSED_AO=rt has replaced what they would have fed.
   *
   * The SAO camera-z/mip chain (`6baabfc5`, `b1bcce99`, `99dcdad8`) is
   * deliberately NOT included here: the same dump shows its output resource
   * also read by two other shaders (`c65fd906`, `98e1bd7b`, the latter
   * matching the depth-of-field guess in docs/research/blessed-look.md's
   * pass table) beyond the sao chain. The dump can't tell whether that's a
   * different mip of the same texture or a real second consumer, so it is
   * left running -- skipping it without game-side confirmation risks a
   * DoF regression for a much smaller draw count than the three above.
   */
  class BlessedSkipAo {
  public:
    static bool IsEnabled() {
      return blessed_skip_replaced_detail::g_aoEnabled;
    }

    // Pre-draw, immediate context only. True: drop this draw.
    static bool ShouldSkipDraw(const D3D11ContextState& state) {
      return IsEnabled() && ShouldSkipDrawSlow(state);
    }

    static void RecordSkipped();

    // Writes <BLESSED_PROBE_DIR>/ao_skip.jsonl every 120 presents.
    static void OnPresent();

  private:
    static bool ShouldSkipDrawSlow(const D3D11ContextState& state);
  };

  /**
   * \brief Skips pass 138's own draw, and (opt-in) the froxel generate/raymarch work that feeds only it
   *
   * BLESSED_VOL_SKIP_VANILLA skips pass 138's draw (`c480e36e`,
   * BLESSED_VOL_SKIP_PS138) outright once BLESSED_VOLUMETRICS=rt is
   * writing the same target directly (BlessedVolumetrics::OnDraw already
   * overwrites it after the fact; skipping the vanilla draw just removes
   * the now-wasted raster work before it happens). Its output is read by
   * nothing but the vanilla blur chain in the dump, which stays running on
   * our replaced values exactly as it would on vanilla's.
   *
   * The froxel generate cs (`ab674eb1`) and its ~90 raymarch dispatches
   * (`1c4ebb62`) are a separate, opt-in switch (BLESSED_VOL_SKIP_RAYMARCH=1,
   * default OFF): skip-more re-checked this against the raw whiterun frame
   * dump (bench/runs/whiterun-dump-1/probe/frame-1920.jsonl) by resource
   * handle, not just format/size. The 90 raymarch dispatches ping-pong two
   * 320x192xD r16_float 3d images (`0x2805d96e960`/`0x2805d96f7a0`); the
   * *last* dispatch (index 6246) writes `0x2805d96e960`. The first
   * `69c92902` draw (index 12332, the 480x270 lens-flare downsample) binds
   * that exact same resource handle at ps srv slot 2 -- not merely a
   * matching format/size, the identical DxvkImage the raymarch just wrote.
   * The second `69c92902` draw (index 12336, 120x270) does not bind it at
   * all. So this is a confirmed, exact single real consumer: skipping the
   * raymarch would starve the first downsample draw of real data every
   * frame. Stays off until it is fed our own replacement (or that draw is
   * confirmed dead some other way); left as future work, not attempted here.
   */
  class BlessedSkipVolumetrics {
  public:
    static bool IsDrawSkipEnabled() {
      return blessed_skip_replaced_detail::g_volDrawEnabled;
    }

    static bool IsDispatchSkipEnabled() {
      return blessed_skip_replaced_detail::g_volDispatchEnabled;
    }

    // Pre-draw, immediate context only. True: drop pass 138's draw.
    static bool ShouldSkipDraw(const D3D11ContextState& state) {
      return IsDrawSkipEnabled() && ShouldSkipDrawSlow(state);
    }

    // Pre-dispatch, immediate context only. True: drop this cs dispatch.
    static bool ShouldSkipDispatch(const D3D11ContextState& state) {
      return IsDispatchSkipEnabled() && ShouldSkipDispatchSlow(state);
    }

    static void RecordSkippedDraw();
    static void RecordSkippedDispatch();

    // Writes <BLESSED_PROBE_DIR>/volumetrics_skip.jsonl every 120 presents.
    static void OnPresent();

  private:
    static bool ShouldSkipDrawSlow(const D3D11ContextState& state);
    static bool ShouldSkipDispatchSlow(const D3D11ContextState& state);
  };

  /**
   * \brief Skips vanilla's second bloom blur pass once BLESSED_LOOK=bless has zeroed its effect
   *
   * BLESSED_LOOK zeroes pass 175's `Param.x` when BLESSED_LOOK_VANILLA_BLOOM
   * stays 0 (its default), which masks vanilla's bloom contribution out of
   * the tonemap entirely regardless of what its bloom texture holds -- so
   * the draw that fills it is dead work. The dump shows that is only true
   * for the *second* bloom pass (`cda7dc03`, pass 174): its output is read
   * by nothing but the tonemap (`716590ec`).
   *
   * The first bloom-bright pass (`36988921`, pass 173) is deliberately NOT
   * included: the same dump shows its output also read by the lens-flare
   * downsample (`2f95aecb`) and the volumetric-lighting/flare composite
   * (`61eac670`, pass 167) -- both real consumers unrelated to bloom.
   * Skipping it would starve them. The 168-172 downsample/luminance/eye
   * -adaptation group is also excluded: pass 169's output alone (`22b5bc6a`)
   * feeds both a lens-flare pass (`57903b50`) and the eye-adaptation chain
   * that pass 175 needs regardless of bloom, and 168's own downsample
   * (`18800b75`) feeds the same lens-flare draw and the DoF pass at a
   * different instance of that shader. None of that group is bloom-only.
   */
  class BlessedSkipBloom {
  public:
    static bool IsEnabled() {
      return blessed_skip_replaced_detail::g_bloomEnabled;
    }

    // Pre-draw, immediate context only. True: drop this draw.
    static bool ShouldSkipDraw(const D3D11ContextState& state) {
      return IsEnabled() && ShouldSkipDrawSlow(state);
    }

    static void RecordSkipped();

    // Writes <BLESSED_PROBE_DIR>/bloom_skip.jsonl every 120 presents.
    static void OnPresent();

  private:
    static bool ShouldSkipDrawSlow(const D3D11ContextState& state);
  };

  /**
   * \brief Skips the sun shadow-mask raster draw itself once our trace has overwritten its target
   *
   * The whiterun dump confirms the sun mask draw (ps `b070feb5`) writes
   * with a write mask of R only (`write_mask: 1`, no other channel, no dsv)
   * -- exactly what BlessedShadow's trace then overwrites. Because
   * BlessedShadow::OnDraw pulls its camera/sun data from this same draw's
   * bound cbuffers via D3D11ContextState (not from the rasterized result),
   * it runs identically whether or not the raster draw actually reached the
   * gpu -- so the raster draw can be dropped outright, the same way
   * BlessedCascadeSkip drops the cascades that used to feed this same mask.
   *
   * blessed: skip-more -- this was opt-in only (BLESSED_SHADOW_SKIP_MASK_DRAW=1)
   * because it lost the traced shadows in game even with the post-draw hooks
   * run. The bug was on the dxvk side, not here: the compute dispatch's
   * output barrier (blessed_shadow.cpp's dispatch()) only named
   * COLOR_ATTACHMENT_OUTPUT as its source scope, so with the raster draw
   * gone the barrier no longer synchronized against our own previous frame's
   * compute write -- a write-after-write race. Fixed by widening that
   * barrier to also name COMPUTE_SHADER_BIT (the same fix vol-2 made for
   * pass 138's output, and the one point-lights' own barrier already
   * carries). Default on now.
   */
  class BlessedSkipShadowMask {
  public:
    static bool IsEnabled() {
      return blessed_skip_replaced_detail::g_shadowMaskEnabled;
    }

    // Pre-draw, immediate context only. True: drop this draw.
    static bool ShouldSkipDraw(const D3D11ContextState& state) {
      return IsEnabled() && ShouldSkipDrawSlow(state);
    }

    static void RecordSkipped();

    // Writes <BLESSED_PROBE_DIR>/shadow_mask_skip.jsonl every 120 presents.
    static void OnPresent();

  private:
    static bool ShouldSkipDrawSlow(const D3D11ContextState& state);
  };

}
