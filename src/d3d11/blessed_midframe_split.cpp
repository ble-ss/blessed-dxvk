// blessed: BLESSED_MIDFRAME_SPLIT, see blessed_midframe_split.h
#include <cstdlib>
#include <string>

#include "blessed_midframe_split.h"
#include "d3d11_context_imm.h"

#include "../util/util_env.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    uint32_t g_splitAtPass = 0u;  // 0: off. 1-based pass index within the frame.
    uint32_t g_passCount   = 0u;  // passes seen since the last present
    bool     g_splitDone   = false;

  }


  bool BlessedMidframeSplit::IsEnabled() {
    static const bool s_enabled = [] {
      std::string value = env::getEnvVar("BLESSED_MIDFRAME_SPLIT");

      if (value.empty())
        return false;

      int n = std::atoi(value.c_str());

      if (n <= 0)
        return false;

      g_splitAtPass = uint32_t(n);

      Logger::info(str::format("BlessedMidframeSplit: forcing a submission "
        "boundary before pass ", g_splitAtPass, " each frame (BLESSED_MIDFRAME_SPLIT=", n, ")"));
      return true;
    } ();

    return s_enabled;
  }


  void BlessedMidframeSplit::OnPassEvent(D3D11ImmediateContext* ctx, BlessedGpuPassKind kind) {
    if (g_splitDone)
      return;

    g_passCount += 1u;

    if (g_passCount != g_splitAtPass)
      return;

    g_splitDone = true;

    // Non-blocking: same call the immediate context already uses for its own
    // implicit flush hints (queries, explicit Flush()). Ending the current
    // command list here is a normal, supported dxvk operation at any point,
    // not just at frame boundaries.
    ctx->ExecuteFlush(GpuFlushType::ImplicitStrongHint, nullptr, false);
  }


  void BlessedMidframeSplit::OnPresent() {
    g_passCount = 0u;
    g_splitDone = false;
  }

}
