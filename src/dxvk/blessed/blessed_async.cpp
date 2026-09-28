// blessed: async compute -- env gates and the DxvkContext side of kick/wait (BLESSED_ASYNC)
#include <filesystem>
#include <fstream>
#include <string>

#include "blessed_async.h"

#include "../dxvk_cmdlist.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/util_time.h"
#include "../../util/log/log.h"

namespace dxvk {

  namespace {

    uint32_t ParsePassMask() {
      std::string list = env::getEnvVar("BLESSED_ASYNC_PASSES");

      if (list.empty())
        return uint32_t(BlessedAsyncPass::Gi) | uint32_t(BlessedAsyncPass::Vol);

      uint32_t mask = 0u;
      size_t start = 0u;

      while (start <= list.size()) {
        size_t end = list.find(',', start);
        if (end == std::string::npos)
          end = list.size();

        std::string item = list.substr(start, end - start);

        if (item == "gi")
          mask |= uint32_t(BlessedAsyncPass::Gi);
        else if (item == "vol")
          mask |= uint32_t(BlessedAsyncPass::Vol);

        start = end + 1u;
      }

      return mask;
    }


    /**
     * \brief BLESSED_ASYNC_TIMING=1: how much async work hid, and what waiting cost
     *
     * Two timestamps per kick on the async queue (start, end) and two per
     * wait on graphics (end of the chunk before it, start of the chunk after
     * it: the stall plus the submit gap). Read back without blocking a ring
     * lap later; a sample not ready by then is dropped. Every ~2 s one line
     * goes to the log and to <BLESSED_PROBE_DIR>/async.jsonl. hidden work =
     * kick ms - stall ms. cs thread only; off, it is never constructed.
     */
    class BlessedAsyncTimer {
      static constexpr uint32_t PairCount = 64u;
    public:

      static bool IsEnabled() {
        static const bool s_enabled = env::getEnvVar("BLESSED_ASYNC_TIMING") == "1";
        return s_enabled;
      }

      static BlessedAsyncTimer* Get(DxvkDevice* device) {
        static BlessedAsyncTimer s_timer(device);
        return &s_timer;
      }

      explicit BlessedAsyncTimer(DxvkDevice* device)
      : m_device(device) {
        VkQueryPoolCreateInfo info = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = PairCount * 2u;

        if (device->vkd()->vkCreateQueryPool(device->handle(), &info, nullptr, &m_pool) != VK_SUCCESS) {
          Logger::warn("blessed: async: no timestamp query pool, BLESSED_ASYNC_TIMING off");
          m_pool = VK_NULL_HANDLE;
        }

        m_windowStart = dxvk::high_resolution_clock::now();
      }

      // never destroyed: the pool dies with the device, like the passes' own

      uint32_t begin(const Rc<DxvkCommandList>& cmd, bool kick) {
        if (m_pool == VK_NULL_HANDLE)
          return UINT32_MAX;

        maybeFlush();

        uint32_t pair = m_next;
        m_next = (m_next + 1u) % PairCount;

        if (m_used[pair])
          collect(pair);

        m_used[pair] = true;
        m_kick[pair] = kick;

        cmd->cmdResetQueryPool(DxvkCmdBuffer::ExecBuffer, m_pool, pair * 2u, 2u);
        cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer, kick
          ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
          : VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_pool, pair * 2u);
        return pair * 2u;
      }

      void end(const Rc<DxvkCommandList>& cmd, uint32_t base, bool kick) {
        if (base == UINT32_MAX)
          return;

        cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer, kick
          ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
          : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_pool, base + 1u);
      }

    private:

      DxvkDevice*  m_device;
      VkQueryPool  m_pool = VK_NULL_HANDLE;
      uint32_t     m_next = 0u;
      bool         m_used[PairCount] = { };
      bool         m_kick[PairCount] = { };

      uint32_t     m_kicks    = 0u;
      double       m_kickMs   = 0.0;
      uint32_t     m_waits    = 0u;
      double       m_stallMs  = 0.0;
      uint32_t     m_dropped  = 0u;

      dxvk::high_resolution_clock::time_point m_windowStart;
      std::ofstream m_file;
      bool          m_fileTried = false;

      void collect(uint32_t pair) {
        struct { uint64_t ts; uint64_t avail; } results[2] = { };

        VkResult vr = m_device->vkd()->vkGetQueryPoolResults(
          m_device->handle(), m_pool, pair * 2u, 2u,
          sizeof(results), results, sizeof(results[0]),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

        if (vr != VK_SUCCESS || !results[0].avail || !results[1].avail || results[1].ts < results[0].ts) {
          m_dropped++;
          return;
        }

        double period = double(m_device->properties().core.properties.limits.timestampPeriod);
        double ms = double(results[1].ts - results[0].ts) * period / 1.0e6;

        if (m_kick[pair]) {
          m_kicks++;
          m_kickMs += ms;
        } else {
          m_waits++;
          m_stallMs += ms;
        }
      }

      void maybeFlush() {
        auto now = dxvk::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(now - m_windowStart).count();

        if (secs < 2.0)
          return;

        double kickAvg  = m_kicks ? m_kickMs  / double(m_kicks) : 0.0;
        double stallAvg = m_waits ? m_stallMs / double(m_waits) : 0.0;

        Logger::info(str::format("blessed: async: ", secs, " s: kicks=", m_kicks,
          " (", kickAvg, " ms each on the async queue), waits=", m_waits,
          " (", stallAvg, " ms each on graphics), hidden~", m_kickMs - m_stallMs,
          " ms total, dropped=", m_dropped));

        if (!m_fileTried) {
          m_fileTried = true;
          std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

          if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            m_file.open(dir + env::PlatformDirSlash + "async.jsonl", std::ios::out | std::ios::app);
          }
        }

        if (m_file.is_open()) {
          m_file << str::format("{\"secs\":", secs,
            ",\"kicks\":", m_kicks, ",\"kick_ms\":", m_kickMs,
            ",\"waits\":", m_waits, ",\"stall_ms\":", m_stallMs,
            ",\"dropped\":", m_dropped, "}\n");
          m_file.flush();
        }

        m_kicks = 0u;
        m_kickMs = 0.0;
        m_waits = 0u;
        m_stallMs = 0.0;
        m_dropped = 0u;
        m_windowStart = now;
      }

    };

    // the open kick's timestamp pair (cs thread only)
    uint32_t g_kickQuery = UINT32_MAX;

  }


  bool BlessedAsync::IsRequested() {
    static const bool s_requested = env::getEnvVar("BLESSED_ASYNC") == "1"
      || BlessedAsync::VanillaVolRequested(); // blessed: vanilla-vol-async, its own switch
    return s_requested;
  }


  bool BlessedAsync::WantsGraphicsFamily() {
    // blessed: vanilla-vol-async always needs the same-family queue (the
    // images it touches are the game's own, never tagged concurrent across
    // families -- see src/d3d11/blessed_vol_async.h)
    static const bool s_graphics = env::getEnvVar("BLESSED_ASYNC_QUEUE") == "graphics"
      || BlessedAsync::VanillaVolRequested();
    return s_graphics;
  }


  bool BlessedAsync::PassRequested(BlessedAsyncPass pass) {
    static const uint32_t s_mask = ParsePassMask();
    return (s_mask & uint32_t(pass)) != 0u;
  }


  bool BlessedAsync::VanillaVolRequested() {
    return VanillaVolMode() != 0u;
  }


  uint32_t BlessedAsync::VanillaVolMode() {
    static const uint32_t s_mode = [] {
      std::string v = env::getEnvVar("BLESSED_VOL_ASYNC");
      return v == "1" ? 1u : v == "2" ? 2u : 0u;
    }();
    return s_mode;
  }


  bool DxvkContext::blessedAsyncAvailable() const {
    return m_device->blessedHasAsyncQueue();
  }


  void DxvkContext::blessedAsyncBegin() {
    // blessed: ends the render pass, finalizes the context's barrier batches
    // into the producer chunk and starts a fresh graphics buffer, the same
    // way upstream splits a command list; then the command list swaps the
    // exec buffer for an async-family one.
    this->splitCommands();
    m_cmd->blessedAsyncBegin();

    if (unlikely(BlessedAsyncTimer::IsEnabled()))
      g_kickQuery = BlessedAsyncTimer::Get(m_device.ptr())->begin(m_cmd, true);
  }


  void DxvkContext::blessedAsyncEnd() {
    if (unlikely(BlessedAsyncTimer::IsEnabled())) {
      BlessedAsyncTimer::Get(m_device.ptr())->end(m_cmd, g_kickQuery, true);
      g_kickQuery = UINT32_MAX;
    }

    m_cmd->blessedAsyncEnd();
    m_blessedAsyncPending = true;
  }


  void DxvkContext::blessedAsyncSync() {
    if (likely(!m_blessedAsyncPending))
      return;

    // the timer's reset must be outside a render pass
    this->endCurrentPass(true);

    uint32_t query = UINT32_MAX;

    if (unlikely(BlessedAsyncTimer::IsEnabled()))
      query = BlessedAsyncTimer::Get(m_device.ptr())->begin(m_cmd, false);

    // the wait applies to a whole submission, so everything recorded from
    // here on goes into a new graphics chunk
    this->splitCommands();
    m_cmd->blessedAsyncWait();
    m_blessedAsyncPending = false;

    // blessed: vol-async-3 -- BLESSED_VOL_ASYNC=2 hands the froxel volumes
    // back to graphics first thing after the wait (one bool test otherwise)
    blessedVolAcquireIfPending();

    if (unlikely(BlessedAsyncTimer::IsEnabled()))
      BlessedAsyncTimer::Get(m_device.ptr())->end(m_cmd, query, false);
  }


  void DxvkDeviceCapabilities::blessedPickAsyncQueue() {
    m_queueMapping.blessedCompute = DxvkDeviceQueueIndex();

    DxvkDeviceQueueIndex pick = { };

    if (BlessedAsync::WantsGraphicsFamily()) {
      // a second queue of the graphics family: no sharing changes at all
      pick.family = m_queueMapping.graphics.family;
      pick.index  = m_queueMapping.graphics.index + 1u;
    } else {
      pick.family = findQueueFamily(
        VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
        VK_QUEUE_COMPUTE_BIT);
      pick.index  = 0u;

      // never share a VkQueue with dxvk's own transfer queue
      if (pick.family != VK_QUEUE_FAMILY_IGNORED && pick.family == m_queueMapping.transfer.family)
        pick.index = m_queueMapping.transfer.index + 1u;
    }

    if (pick.family == VK_QUEUE_FAMILY_IGNORED) {
      Logger::warn("blessed: async: no compute-only queue family, BLESSED_ASYNC stays off");
      return;
    }

    const auto& props = m_queuesAvailable[pick.family].core.queueFamilyProperties;

    if (pick.index >= props.queueCount) {
      Logger::warn(str::format("blessed: async: family ", pick.family, " has ",
        props.queueCount, " queue(s), need index ", pick.index, ", BLESSED_ASYNC stays off"));
      return;
    }

    // the passes write timestamps from the async queue
    if (!props.timestampValidBits) {
      Logger::warn(str::format("blessed: async: family ", pick.family,
        " has no timestamp bits, BLESSED_ASYNC stays off"));
      return;
    }

    m_queueMapping.blessedCompute = pick;

    Logger::info(str::format("blessed: async: compute queue family ", pick.family,
      " index ", pick.index, " (graphics family ", m_queueMapping.graphics.family, ")"));

    // blessed: vol-async-3 -- BLESSED_VOL_ASYNC=2 also wants a queue of the
    // compute-only family for the chain (generate stays on the one above)
    if (BlessedAsync::VanillaVolMode() == 2u)
      blessedPickVolComputeQueue();
  }


  void DxvkDeviceCapabilities::blessedPickVolComputeQueue() {
    m_queueMapping.blessedVolCompute = DxvkDeviceQueueIndex();

    DxvkDeviceQueueIndex pick = { };
    pick.family = findQueueFamily(
      VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
      VK_QUEUE_COMPUTE_BIT);
    pick.index  = 0u;

    if (pick.family != VK_QUEUE_FAMILY_IGNORED && pick.family == m_queueMapping.transfer.family)
      pick.index = m_queueMapping.transfer.index + 1u;

    if (pick.family == VK_QUEUE_FAMILY_IGNORED) {
      Logger::warn("blessed: vol-async: no compute-only queue family, BLESSED_VOL_ASYNC=2 runs as =1");
      return;
    }

    const auto& props = m_queuesAvailable[pick.family].core.queueFamilyProperties;

    if (pick.index >= props.queueCount || !props.timestampValidBits) {
      Logger::warn(str::format("blessed: vol-async: compute family ", pick.family,
        " has no usable queue (", props.queueCount, " queues, need index ", pick.index,
        ", timestamp bits ", props.timestampValidBits, "), BLESSED_VOL_ASYNC=2 runs as =1"));
      return;
    }

    m_queueMapping.blessedVolCompute = pick;

    Logger::info(str::format("blessed: vol-async: chain queue family ", pick.family,
      " index ", pick.index, " (compute only)"));
  }

}
