// blessed: gpu timestamps at every command buffer's begin and end, so a frame splits into app passes, dxvk's own work and idle (BLESSED_GPU_PASSES)
#include <deque>
#include <string>

#include "blessed_gpu_gaps.h"

#include "../dxvk_cmdlist.h"
#include "../dxvk_device.h"

#include "../../util/thread.h"
#include "../../util/util_env.h"
#include "../../util/util_time.h"

namespace dxvk {

  namespace {

    constexpr uint32_t PoolSize       = 16384u; // queries, reused in a ring
    constexpr size_t   MaxPending     = 64u;    // closed frames waiting for the gpu
    constexpr size_t   MaxResolved    = 64u;    // resolved frames waiting for d3d11
    constexpr size_t   MaxOpenEntries = 4096u;  // a frame without present (loading) is dropped

    bool InitEnabled() {
      std::string passes = env::getEnvVar("BLESSED_GPU_PASSES");
      std::string gaps   = env::getEnvVar("BLESSED_GPU_GAPS");

      // =frame is the pass timer's own-cost baseline: no extra marks there
      // unless asked for
      if (passes == "1")
        return gaps != "0";

      if (passes == "frame")
        return gaps == "1";

      return false;
    }

    struct PendingFrame {
      std::vector<BlessedGapSubmitEntry> entries;
      BlessedGapFrame           counts;   // gfx left empty until resolved
      uint32_t                  age      = 0u;
      bool                      overflow = false;
    };

    // query pool, shared by every recording thread
    dxvk::mutex   g_poolMutex;
    DxvkDevice*   g_device        = nullptr;  // not owned: an Rc here would hang dll unload
    VkQueryPool   g_pool          = VK_NULL_HANDLE;
    bool          g_poolFailed    = false;
    bool          g_transferMarks = false;
    uint32_t      g_head          = 0u;

    // frame assembly, submission thread
    dxvk::mutex               g_submitMutex;
    PendingFrame              g_open;
    std::deque<PendingFrame>  g_pending;
    uint64_t                  g_lastEnd = 0u;   // 0: previous frame unknown
    uint32_t                  g_prevEndQuery  = ~0u;  // last graphics end mark of the last closed frame
    uint64_t                  g_presentCallNs = 0u;   // last present call, cpu
    uint64_t                  g_presentEndNs  = 0u;   // its return time, 0 if none since

    // resolved frames, taken by the app thread
    dxvk::mutex                  g_resolvedMutex;
    std::deque<BlessedGapFrame>  g_resolved;

    bool CreatePool(DxvkDevice* device) {
      auto vkd = device->vkd();

      VkQueryPoolCreateInfo info = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
      info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
      info.queryCount = PoolSize;

      if (vkd->vkCreateQueryPool(vkd->device(), &info, nullptr, &g_pool) != VK_SUCCESS) {
        Logger::err("BlessedGpuGaps: cannot create the timestamp query pool");
        g_poolFailed = true;
        g_pool = VK_NULL_HANDLE;
        return false;
      }

      vkd->vkResetQueryPool(vkd->device(), g_pool, 0u, PoolSize);

      const auto& transfer = device->queues().transfer;
      g_transferMarks = transfer.properties.core.queueFamilyProperties.timestampValidBits != 0u;
      g_device = device;

      Logger::info(str::format("BlessedGpuGaps: ", PoolSize, " timestamps, transfer queue marks ",
        g_transferMarks ? "on" : "off (no timestamps on that family)"));
      return true;
    }

    bool ReadQuery(const Rc<vk::DeviceFn>& vkd, uint32_t query, uint64_t& ts) {
      uint64_t data[2] = { 0u, 0u };

      VkResult vr = vkd->vkGetQueryPoolResults(vkd->device(), g_pool, query, 1u,
        sizeof(data), data, sizeof(data), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

      if ((vr != VK_SUCCESS && vr != VK_NOT_READY) || !data[1])
        return false;

      ts = data[0];
      return true;
    }

    // oldest first; stops at the first frame the gpu has not finished
    void ResolvePending(DxvkDevice* device) {
      auto vkd = device->vkd();

      while (!g_pending.empty()) {
        PendingFrame& pf = g_pending.front();

        BlessedGapFrame frame = pf.counts;
        frame.gfx.reserve(pf.entries.size());

        bool ready = true;

        for (const auto& e : pf.entries) {
          uint64_t b = 0u, en = 0u;

          if (!ReadQuery(vkd, e.beginQuery, b) || !ReadQuery(vkd, e.endQuery, en)) {
            ready = false;
            break;
          }

          if (e.graphics) {
            BlessedGapSegment seg;
            seg.begin = b;
            seg.end   = std::max(b, en);
            seg.type  = e.type;
            seg.wait  = e.wait;
            frame.gfx.push_back(seg);
          } else if (en > b) {
            frame.sdmaTicks += en - b;
          }
        }

        if (!ready) {
          // a frame that never resolves (a mark that did not execute) must
          // not block every later one
          if (++pf.age > MaxPending) {
            g_pending.pop_front();
            g_lastEnd = 0u;
            continue;
          }

          return;
        }

        g_pending.pop_front();

        if (frame.gfx.empty()) {
          g_lastEnd = 0u;
          continue;
        }

        frame.prevEnd = g_lastEnd;
        g_lastEnd = frame.gfx.back().end;

        std::lock_guard lock(g_resolvedMutex);

        if (g_resolved.size() >= MaxResolved)
          g_resolved.pop_front();

        g_resolved.push_back(std::move(frame));
      }
    }

  }


  namespace blessed_gpu_gaps_detail {
    extern const bool g_enabled = InitEnabled();
  }


  uint32_t BlessedGpuGaps::AllocQuery(DxvkDevice* device, bool transferQueue) {
    std::lock_guard lock(g_poolMutex);

    if (!g_pool) {
      if (g_poolFailed || !CreatePool(device))
        return ~0u;
    }

    if (device != g_device || (transferQueue && !g_transferMarks))
      return ~0u;

    uint32_t query = g_head;
    g_head = (g_head + 1u) % PoolSize;

    // reused PoolSize marks later, long after its frame resolved or was dropped
    auto vkd = device->vkd();
    vkd->vkResetQueryPool(vkd->device(), g_pool, query, 1u);
    return query;
  }


  VkQueryPool BlessedGpuGaps::QueryPool() {
    return g_pool;
  }


  void BlessedGpuGaps::OnSubmit(DxvkDevice* device, const BlessedGapSubmitInfo& info) {
    if (device != g_device)
      return;

    std::lock_guard lock(g_submitMutex);

    // first command list of a frame: had the gpu already run out of work?
    if (!g_open.counts.cmdLists) {
      if (g_prevEndQuery != ~0u) {
        uint64_t ts = 0u;
        g_open.counts.startState = ReadQuery(device->vkd(), g_prevEndQuery, ts) ? 0u : 1u;
      }

      if (g_presentEndNs) {
        uint64_t now = NowNs();
        g_open.counts.presentCallNs     = g_presentCallNs;
        g_open.counts.presentToSubmitNs = now > g_presentEndNs ? now - g_presentEndNs : 0u;
      }
    }

    g_open.counts.cmdLists        += 1u;
    g_open.counts.graphicsSubmits += info.graphicsSubmits;
    g_open.counts.sdmaChunks      += info.sdmaChunks;
    g_open.counts.barriers        += info.barriers;
    g_open.counts.renderPasses    += info.renderPasses;

    if (!g_open.overflow) {
      g_open.entries.insert(g_open.entries.end(), info.entries.begin(), info.entries.end());

      if (g_open.entries.size() > MaxOpenEntries) {
        g_open.overflow = true;
        g_open.entries.clear();
      }
    }

    if (!info.present)
      return;

    g_prevEndQuery = ~0u;
    g_presentEndNs = 0u;

    for (auto e = g_open.entries.rbegin(); e != g_open.entries.rend(); e++) {
      if (e->graphics) {
        g_prevEndQuery = e->endQuery;
        break;
      }
    }

    if (g_open.overflow) {
      g_lastEnd = 0u;
    } else {
      if (g_pending.size() >= MaxPending) {
        g_pending.pop_front();
        g_lastEnd = 0u;
      }

      g_pending.push_back(std::move(g_open));
    }

    g_open = PendingFrame();
    ResolvePending(device);
  }


  void BlessedGpuGaps::OnPresentCall(uint64_t beginNs, uint64_t endNs) {
    std::lock_guard lock(g_submitMutex);
    g_presentCallNs = endNs > beginNs ? endNs - beginNs : 0u;
    g_presentEndNs  = endNs;
  }


  uint64_t BlessedGpuGaps::NowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
      dxvk::high_resolution_clock::now().time_since_epoch()).count());
  }


  bool BlessedGpuGaps::NoTransferQueue() {
    static const bool s_enabled = [] {
      bool on = env::getEnvVar("BLESSED_NO_TRANSFER_QUEUE") == "1";

      if (on)
        Logger::info("BlessedGpuGaps: transfer queue mapped onto the graphics queue (BLESSED_NO_TRANSFER_QUEUE=1)");

      return on;
    } ();

    return s_enabled;
  }


  bool BlessedGpuGaps::TakeFrame(BlessedGapFrame& frame) {
    std::lock_guard lock(g_resolvedMutex);

    if (g_resolved.empty())
      return false;

    frame = std::move(g_resolved.front());
    g_resolved.pop_front();
    return true;
  }


  // ---- DxvkCommandList side (declared in dxvk_cmdlist.h) ----

  void DxvkCommandList::blessedGapMark(VkCommandBuffer cmdBuffer, DxvkCmdBuffer type, bool end) {
    // end marks only for buffers that got a begin mark
    if (end) {
      bool begun = false;

      for (const auto& m : m_blessedGapMarks)
        begun |= m.cmdBuffer == cmdBuffer && !m.end;

      if (!begun)
        return;
    }

    bool transferQueue = type >= DxvkCmdBuffer::SdmaBuffer && m_device->hasDedicatedTransferQueue();
    uint32_t query = BlessedGpuGaps::AllocQuery(m_device, transferQueue);

    if (query == ~0u)
      return;

    // straight to vulkan: cmdWriteTimestamp would flag an unused exec
    // buffer as used and get it submitted
    m_vkd->vkCmdWriteTimestamp2(cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      BlessedGpuGaps::QueryPool(), query);

    BlessedGapMark mark;
    mark.cmdBuffer = cmdBuffer;
    mark.query     = query;
    mark.type      = uint8_t(type);
    mark.end       = end;
    m_blessedGapMarks.push_back(mark);
  }


  void DxvkCommandList::blessedGapSubmit() {
    BlessedGapSubmitInfo info;
    info.entries.reserve(m_cmdSubmissions.size() * 3u);

    bool dedicated = m_device->hasDedicatedTransferQueue();

    auto findMark = [this] (VkCommandBuffer cmdBuffer, bool end) {
      for (const auto& m : m_blessedGapMarks) {
        if (m.cmdBuffer == cmdBuffer && m.end == end)
          return m.query;
      }

      return ~0u;
    };

    static const std::array<DxvkCmdBuffer, 5> Order = {
      DxvkCmdBuffer::SdmaBarriers, DxvkCmdBuffer::SdmaBuffer,
      DxvkCmdBuffer::InitBarriers, DxvkCmdBuffer::InitBuffer,
      DxvkCmdBuffer::ExecBuffer,
    };

    for (size_t i = 0; i < m_cmdSubmissions.size(); i++) {
      const auto& cmd = m_cmdSubmissions[i];
      bool isFirst = i == 0;
      bool isLast  = i == m_cmdSubmissions.size() - 1;

      bool hasSdma = cmd.cmdBuffers[uint32_t(DxvkCmdBuffer::SdmaBarriers)]
                  || cmd.cmdBuffers[uint32_t(DxvkCmdBuffer::SdmaBuffer)];

      // what the chunk's graphics submission waits on, strongest first
      BlessedGapWait wait = BlessedGapWait::None;

      if (isFirst && m_wsiSemaphores.acquire)
        wait = BlessedGapWait::Acquire;
      else if (cmd.blessedWaitCompute)
        wait = BlessedGapWait::Async;
      else if (dedicated && hasSdma)
        wait = BlessedGapWait::Sdma;
      else if (isFirst && !m_waitSemaphores.empty())
        wait = BlessedGapWait::Fence;
      else if (cmd.sparseBind)
        wait = BlessedGapWait::Sparse;

      if (dedicated && hasSdma)
        info.sdmaChunks += 1u;

      info.graphicsSubmits += 1u;

      if (isLast && (m_wsiSemaphores.present || m_wsiSemaphores.acquire))
        info.graphicsSubmits += 1u;

      bool firstGraphics = true;

      for (DxvkCmdBuffer type : Order) {
        VkCommandBuffer cmdBuffer = cmd.cmdBuffers[uint32_t(type)];

        if (!cmdBuffer || (type == DxvkCmdBuffer::ExecBuffer && !cmd.execCommands))
          continue;

        uint32_t b = findMark(cmdBuffer, false);
        uint32_t e = findMark(cmdBuffer, true);

        if (b == ~0u || e == ~0u)
          continue;

        BlessedGapSubmitEntry entry;
        entry.beginQuery = b;
        entry.endQuery   = e;
        entry.type       = uint8_t(type);
        entry.graphics   = !(dedicated && type >= DxvkCmdBuffer::SdmaBuffer);

        if (entry.graphics && firstGraphics) {
          entry.wait = wait;
          firstGraphics = false;
        }

        info.entries.push_back(entry);
      }
    }

    info.barriers     = uint32_t(m_statCounters.getCtr(DxvkStatCounter::CmdBarrierCount));
    info.renderPasses = uint32_t(m_statCounters.getCtr(DxvkStatCounter::CmdRenderPassCount));
    info.present      = m_wsiSemaphores.present != VK_NULL_HANDLE;

    BlessedGpuGaps::OnSubmit(m_device, info);
    m_blessedGapMarks.clear();
  }

}
