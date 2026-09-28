// blessed: per-pass gpu timestamps at render-target, dispatch and transfer boundaries (BLESSED_GPU_PASSES)
#pragma once

#include <atomic>

#include "d3d11_context_state.h"

#include "../dxvk/dxvk_context.h"
#include "../dxvk/dxvk_gpu_query.h"

namespace dxvk {

  namespace blessed_gpu_passes_detail {
    // 0: off, 1: a timestamp at every pass boundary, 2: frame start/end only
    // (the instrument's own-cost baseline). Set once at dll load.
    extern const uint32_t g_mode;
  }

  enum class BlessedGpuPassKind : uint32_t {
    Draw,
    Compute,
    Xfer,
  };

  /// blessed: one timestamp to write on the cs thread. written counts the
  /// frame's marks that actually reached DxvkContext, so the app thread never
  /// reads a query whose begin() is still in flight on the cs thread.
  struct BlessedGpuPassMark {
    Rc<DxvkQuery>           query;
    std::atomic<uint32_t>*  written = nullptr;
  };

  /**
   * \brief GPU pass timer, the dxvk half of the gpu-parity instruments
   *
   * Enabled by BLESSED_GPU_PASSES=1 (or =frame). Immediate context only.
   * A pass begins at the first draw after the bound render-target/depth
   * resources change (or after non-draw work), at the first dispatch of a
   * run of dispatches, and at the first clear/copy/resolve of a run of
   * those. Each pass start writes one gpu timestamp through the same path
   * a D3D11 timestamp query takes (vkCmdWriteTimestamp with a host query
   * reset, so no render pass is split). Present writes the frame's end
   * mark, and one more after dxvk's own swapchain blit.
   *
   * Frames resolve three or more presents late without a gpu sync; every
   * 120 resolved frames, one window (the most common pass layout, averaged)
   * goes to "<dir>/gpu-passes.jsonl", dir being BLESSED_GPU_PASSES_DIR,
   * else BLESSED_PROBE_DIR, else %TEMP%/blessed-gpu-passes. The format is shared
   * with tools/d3d11-timing (the native proxy); bench/passdiff.py reads both.
   *
   * With BlessedGpuGaps on (default with =1, BLESSED_GPU_GAPS=0 turns it
   * off; with =frame only BLESSED_GPU_GAPS=1 turns it on), each window also gets one "gaps" line: the frame on the
   * gpu split into the app's passes, the present blit, dxvk's own work
   * outside them, and idle by what the queue waited on.
   *
   * When off, every call site pays one cached-bool branch.
   */
  class BlessedGpuPasses {
  public:

    static bool IsEnabled() {
      return blessed_gpu_passes_detail::g_mode != 0u;
    }

    /// App thread, under the context lock. Returns a mark with a null query
    /// when this call does not begin a new pass.
    static BlessedGpuPassMark OnCall(
            DxvkDevice*           device,
      const D3D11ContextState&    state,
            BlessedGpuPassKind    kind);

    /// App thread, start of D3D11SwapChain::PresentImage: the end of the
    /// frame's last pass. Null query when the frame recorded no pass.
    static BlessedGpuPassMark OnFrameEnd(DxvkDevice* device);

    /// App thread, right before the present blit is emitted: the end of
    /// dxvk's own present pass (swapchain blit). Null query when no frame.
    static BlessedGpuPassMark OnPresentBlit(DxvkDevice* device);

    /// App thread, after PresentImage: closes the frame, resolves finished
    /// frames and writes a window every 120 of them.
    static void OnPresent();

    /// App thread, under the context lock: an immediate context flush that
    /// submits (GpuFlushType as uint32_t), counted for the gaps line.
    static void OnFlush(uint32_t flushType);

    /// CS thread.
    static void WriteMark(DxvkContext* ctx, const BlessedGpuPassMark& mark) {
      ctx->writeTimestamp(mark.query);
      mark.written->fetch_add(1u, std::memory_order_release);
    }

  };

}
