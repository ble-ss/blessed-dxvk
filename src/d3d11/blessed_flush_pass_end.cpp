// blessed: defer non-synchronisation implicit flushes to the render-pass
// end instead of splitting the pass mid-flight (BLESSED_FLUSH_AT_PASS_END)
#include <chrono>
#include <cstdlib>

#include "blessed_flush_pass_end.h"

#include "../util/log/log.h"
#include "../util/util_env.h"
#include "../util/util_string.h"

namespace dxvk {

  namespace {

    // Owner thread only, see the class comment
    struct Counters {
      uint64_t frames     = 0u;
      uint64_t boundaries = 0u;
      uint64_t splits     = 0u;
      uint64_t deferred   = 0u;
      uint64_t idle       = 0u;
      uint64_t capped     = 0u;
    };

    Counters g_window = { };
    std::chrono::steady_clock::time_point g_windowStart = { };
    bool g_windowStarted = false;

    uint32_t InitMaxChunks() {
      std::string value = env::getEnvVar("BLESSED_FLUSH_AT_PASS_END_MAX_CHUNKS");
      long parsed = value.empty() ? 0l : std::strtol(value.c_str(), nullptr, 10);
      return parsed > 0l ? uint32_t(parsed) : 32u;
    }

    double PerFrame(uint64_t count, uint64_t frames) {
      return double(count) / double(frames);
    }

  }

  bool BlessedFlushPassEnd::Enabled() {
    // no Logger in the initializer: static init may run before the logger
    static const bool enabled = env::getEnvVar("BLESSED_FLUSH_AT_PASS_END") == "1";
    return enabled;
  }


  uint32_t BlessedFlushPassEnd::MaxChunks() {
    static const uint32_t maxChunks = InitMaxChunks();
    return maxChunks;
  }


  void BlessedFlushPassEnd::OnBoundary() {
    g_window.boundaries += 1u;
  }


  void BlessedFlushPassEnd::OnSplit() {
    g_window.splits += 1u;
  }


  void BlessedFlushPassEnd::OnDeferred(bool GpuIdle) {
    g_window.deferred += 1u;
    g_window.idle += GpuIdle ? 1u : 0u;
  }


  void BlessedFlushPassEnd::OnCapped() {
    g_window.capped += 1u;
  }


  void BlessedFlushPassEnd::OnFrame() {
    auto now = std::chrono::steady_clock::now();

    if (!g_windowStarted) {
      g_windowStart   = now;
      g_windowStarted = true;

      Logger::info(str::format("BlessedFlushPassEnd: on, max chunks ", MaxChunks()));
    }

    g_window.frames += 1u;

    if (now - g_windowStart < std::chrono::seconds(5))
      return;

    uint64_t f = g_window.frames;

    Logger::info(str::format("BlessedFlushPassEnd: ", f, " frames, per frame:",
      " rt changes ", PerFrame(g_window.boundaries, f),
      ", mid-pass flushes ", PerFrame(g_window.splits, f),
      ", deferred ", PerFrame(g_window.deferred, f),
      " (gpu idle ", PerFrame(g_window.idle, f), ")",
      ", capped ", PerFrame(g_window.capped, f)));

    g_window      = Counters();
    g_windowStart = now;
  }

}
