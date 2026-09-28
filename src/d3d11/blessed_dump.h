// blessed: per-draw frame dumper for offline pass/shader classification (BLESSED_PROBE_DIR)
#pragma once

#include "d3d11_context_state.h"

namespace dxvk {

  // blessed: backing flags for BlessedDump::IsEnabled()/IsCapturing(), split
  // out of blessed_dump.cpp so both can be trivial inline reads at their (very
  // hot, e.g. Map()) call sites. g_enabled used to be read through a
  // function-local magic static, which re-checks a thread-safe init guard on
  // every single call even once initialized; g_enabled here is a plain
  // namespace-scope bool set once at static-init time (see blessed_dump.cpp),
  // same pattern as blessed::g_enabled in util_blessed_probe.h.
  namespace blessed_dump_detail {
    extern const bool g_enabled;   // BLESSED_PROBE_DIR was set at load; never changes after
    extern bool       g_capturing; // true only while a frame is actively being recorded
  }

  enum class BlessedDumpOp : uint8_t {
    Draw,
    DrawIndexed,
    DrawInstanced,
    DrawIndexedInstanced,
  };

  /// blessed: draw/instance counts, in the terms the various Draw* entry points use
  struct BlessedDrawCounts {
    UINT vertexOrIndexCount = 0;
    UINT instanceCount      = 1;
    INT  startVertexOrIndex = 0;
    INT  baseVertex         = 0;
    UINT startInstance      = 0;
  };

  /**
   * \brief Frame draw-classification dumper
   *
   * Enabled by the BLESSED_PROBE_DIR environment variable. When capturing,
   * dumps every draw/dispatch/clear/copy/update/map issued on the immediate
   * context between one D3D11SwapChain::Present call and the next to
   * "<dir>/frame-<n>.jsonl" (one json object per line, in submission order),
   * plus every resource touched to "<dir>/frame-<n>-resources.jsonl".
   *
   * A capture starts when "<dir>/dump.trigger" is found to exist (checked at
   * most once per 30 presents; the file is deleted once seen) or when the
   * upcoming present index is listed in BLESSED_DUMP_AT (comma separated).
   *
   * All entry points other than IsEnabled()/IsCapturing() are no-ops unless
   * a capture is active; when the feature is off entirely (the common case),
   * every call site pays exactly one cached-bool branch.
   */
  class BlessedDump {
  public:

    // Cheap, cached: true once BLESSED_PROBE_DIR is set. Does not by
    // itself mean a frame is being captured right now.
    static bool IsEnabled() {
      return blessed_dump_detail::g_enabled;
    }

    // True only while a frame is actively being captured. Callers should
    // gate all recording on this (IsEnabled() is implied).
    static bool IsCapturing() {
      return blessed_dump_detail::g_enabled
          && blessed_dump_detail::g_capturing;
    }

    // Call once per D3D11SwapChain::Present. Marks a present-to-present
    // frame boundary: flushes/closes the capture in progress (if any) and
    // decides whether the frame about to start should be captured.
    static void OnPresent();

    static void RecordDraw(
      const D3D11ContextState&   state,
            BlessedDumpOp        op,
      const BlessedDrawCounts&   counts);

    static void RecordDispatch(
      const D3D11ContextState&   state,
            UINT                 ThreadGroupCountX,
            UINT                 ThreadGroupCountY,
            UINT                 ThreadGroupCountZ);

    static void RecordClearRtv(ID3D11RenderTargetView* pRtv, const FLOAT color[4]); // blessed: + clear colour
    static void RecordClearDsv(ID3D11DepthStencilView* pDsv);

    static void RecordCopyResource(
            ID3D11Resource*      pDst,
            ID3D11Resource*      pSrc);

    static void RecordCopySubresourceRegion(
            ID3D11Resource*      pDst,
            UINT                 DstSubresource,
            ID3D11Resource*      pSrc,
            UINT                 SrcSubresource);

    static void RecordUpdateSubresource(
            ID3D11Resource*      pDst,
            UINT                 DstSubresource);

    static void RecordMap(
            ID3D11Resource*      pResource,
            UINT                 Subresource,
            D3D11_MAP            MapType);
  };

}
