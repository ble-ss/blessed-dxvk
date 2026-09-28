// blessed: shader-replace, swaps a game shader's dxbc by hash and twin-draw verifies the swap
#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "../dxvk/dxvk_shader_key.h"

#include "d3d11_include.h"

namespace dxvk {

  class D3D11CommonShader;
  class D3D11Device;
  class D3D11ImmediateContext;
  struct D3D11ContextState;

  namespace blessed_shader_replace_detail {
    extern const bool g_replaceEnabled;
    extern const bool g_verifyEnabled;
  }

  /**
   * \brief Replacement shaders, keyed by the vanilla shader's hash
   *
   * BLESSED_SHADER_REPLACE=<dir>: every file in <dir> named like dxvk's
   * shader names plus ".dxbc" (e.g. "fs.b63b3bf7d8862ae508dfbb37d874f3af.dxbc",
   * "cs.ab674eb17a87e2a53c93b6c42ce02f7b.dxbc") replaces the game shader
   * with that name. The swap happens in D3D11Device::CreateShaderModule
   * before the container check, so dxvk's whole path (parse, binding mask,
   * icb, linkage, lowering) runs on the replacement. The shader key, and so
   * the "fs.<hash>" name every blessed hook matches on, stays the vanilla
   * one.
   *
   * Replaced shaders (and their verify twins) skip dxvk's disk shader
   * cache: the cache is keyed by that same name, so a cached vanilla ir
   * would be served for a replacement, and a cached replacement would be
   * served to a later vanilla run.
   *
   * A replacement is refused (vanilla runs, one warning) if its file's
   * dxbc checksum does not validate or its program type differs.
   *
   * Unset: no table is loaded; CreateShaderModule pays one cached-bool
   * check per shader creation and nothing per draw.
   */
  class BlessedShaderReplace {

  public:

    /// callback into D3D11Device::CreateShaderModule for the verify twins
    using CreateModuleFn = std::function<bool(
      const DxvkShaderHash& key, const void* code, size_t size, D3D11CommonShader* out)>;

    static bool IsEnabled() {
      return blessed_shader_replace_detail::g_replaceEnabled;
    }

    /**
     * \brief Swaps in the replacement bytecode for a shader key
     *
     * On a hit, *ppCode and *pSize point at the replacement (owned by
     * the table, alive for the process). With BLESSED_SHADER_VERIFY on,
     * also builds the two twin modules (vanilla and replacement, under
     * salted keys) through \p create.
     * \returns true if the bytecode was swapped
     */
    static bool Apply(
            D3D11Device*      device,
      const DxvkShaderHash&   key,
      const void**            ppCode,
            size_t*           pSize,
      const CreateModuleFn&   create);

    /**
     * \brief Whether a shader key must bypass dxvk's disk shader cache
     */
    static bool SkipDiskCache(const DxvkShaderHash& key);

  };


  /**
   * \brief Twin-draw verifier for replaced shaders
   *
   * BLESSED_SHADER_VERIFY=1 (needs BLESSED_SHADER_REPLACE): after a draw
   * whose pixel shader, or a dispatch whose compute shader, is replaced,
   * the same draw is issued twice more on the immediate context: once
   * with the vanilla shader and once with the replacement, each into its
   * own set of zero-cleared scratch targets (the bound rtvs' formats and
   * sizes; for a dispatch, the bound uavs' formats and sizes). Blending
   * is off and no depth buffer is bound, so both twins write the same
   * pixels in the same primitive order. A compute pass then diffs the
   * two sets per target and the result is written, one jsonl line per
   * checked draw, to "<BLESSED_PROBE_DIR>/shader-verify.jsonl" (and a
   * summary line per shader to the dxvk log).
   *
   * BLESSED_SHADER_VERIFY_LIST=<names or 8+ hex prefixes>: check only these
   * (default: every replaced shader).
   * BLESSED_SHADER_VERIFY_MAX=<n>: checks per shader for the process
   * (default 8). Each check stalls on its readback; it is a debug mode.
   *
   * Skipped (logged once per shader): draws with uavs bound on the output
   * merger, msaa targets, target formats a float srv cannot read.
   */
  class BlessedShaderVerify {

  public:

    using ReplayFn = std::function<void()>;

    static bool IsEnabled() {
      return blessed_shader_replace_detail::g_verifyEnabled;
    }

    static void OnDraw(
            D3D11ImmediateContext*  ctx,
      const D3D11ContextState&      state,
      const ReplayFn&               replay);

    static void OnDispatch(
            D3D11ImmediateContext*  ctx,
      const D3D11ContextState&      state,
      const ReplayFn&               replay);

    /**
     * rief Diffs pairs of textures with the verifier's kernel
     *
     * For other blessed passes that need a gpu-side equality proof (e.g.
     * the volumetric chain collapse). Each a[i] is compared with b[i]
     * (same format and size, 2d or 3d, float-readable), one jsonl line
     * with one target per pair. Works without BLESSED_SHADER_REPLACE.
     * Stalls on the readback.
     */
    static void CompareTextures(
            ID3D11DeviceContext*    ctx,
      const std::string&            label,
            uint32_t                check,
            uint32_t                checks,
            ID3D11Resource* const*  a,
            ID3D11Resource* const*  b,
            uint32_t                count);

  };

}
