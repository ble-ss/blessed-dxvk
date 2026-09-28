// blessed: present-idle -- env gates and the present queue pick (BLESSED_PRESENT)
#include <cstdlib>
#include <string>

#include "blessed_present.h"

#include "../dxvk_device_info.h"
#include "../dxvk_queue.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/util_time.h"
#include "../../util/log/log.h"

namespace dxvk {

  BlessedPresentMode BlessedPresent::Mode() {
    static const BlessedPresentMode s_mode = [] {
      std::string v = env::getEnvVar("BLESSED_PRESENT");

      if (v == "queue")
        return BlessedPresentMode::Queue;
      if (v == "defer")
        return BlessedPresentMode::Defer;
      if (v == "bridge")
        return BlessedPresentMode::Bridge;

      return BlessedPresentMode::Off;
    } ();

    return s_mode;
  }


  uint64_t BlessedPresent::DeferNs() {
    static const uint64_t s_ns = [] {
      std::string v = env::getEnvVar("BLESSED_PRESENT_DEFER_US");
      long us = v.empty() ? 4000 : std::strtol(v.c_str(), nullptr, 10);
      return uint64_t(us > 0 ? us : 0) * 1000u;
    } ();

    return s_ns;
  }


  bool BlessedPresent::LazyAcquire() {
    static const bool s_lazy = env::getEnvVar("BLESSED_PRESENT_ACQUIRE") == "lazy";
    return s_lazy;
  }


  void DxvkDeviceCapabilities::blessedPickPresentQueue() {
    m_queueMapping.blessedPresent = DxvkDeviceQueueIndex();

    // one past every queue already taken from the graphics family
    DxvkDeviceQueueIndex pick = { };
    pick.family = m_queueMapping.graphics.family;
    pick.index  = m_queueMapping.graphics.index + 1u;

    for (const auto& q : { m_queueMapping.transfer, m_queueMapping.sparse, m_queueMapping.blessedCompute }) {
      if (q.family == pick.family && q.index >= pick.index)
        pick.index = q.index + 1u;
    }

    const auto& props = m_queuesAvailable[pick.family].core.queueFamilyProperties;

    if (pick.index >= props.queueCount) {
      Logger::warn(str::format("blessed: present: graphics family ", pick.family, " has ",
        props.queueCount, " queue(s), need index ", pick.index, ", presenting on the graphics queue"));
      return;
    }

    m_queueMapping.blessedPresent = pick;

    Logger::info(str::format("blessed: present: dedicated present queue, family ",
      pick.family, " index ", pick.index));
  }



  size_t DxvkSubmissionQueue::blessedDeferPick(
          std::unique_lock<dxvk::mutex>& lock,
          bool&                          deferred) {
    if (deferred || m_submitQueue.front().present.presenter == nullptr)
      return 0u;

    // one hold per present: after this it goes out next, whatever happens
    deferred = true;

    auto ready = [this] {
      return m_stopped.load() || m_submitQueue.size() > 1u || m_blessedWaiters.load();
    };

    auto deadline = dxvk::high_resolution_clock::now()
      + std::chrono::nanoseconds(BlessedPresent::DeferNs());

    // dxvk::condition_variable::wait_until is not usable here (it passes a
    // negative timeout), so wait in whole milliseconds; appends wake it early
    while (!ready()) {
      auto now = dxvk::high_resolution_clock::now();

      if (now >= deadline)
        break;

      auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
      uint32_t ms = uint32_t(std::max<int64_t>(left, 1));
#ifdef _WIN32
      // straight to the os: the wrapper's wait_for narrows a 64-bit count
      SleepConditionVariableSRW(m_appendCond.native_handle(), lock.mutex()->native_handle(), ms, 0);
#else
      m_appendCond.wait_for(lock, std::chrono::milliseconds(ms));
#endif
    }

    // only a command list may jump the present, never a second present
    if (m_submitQueue.size() > 1u && m_submitQueue.at(1u).submit.cmdList != nullptr)
      return 1u;

    return 0u;
  }

}
