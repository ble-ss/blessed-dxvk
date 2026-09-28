// blessed: threaded-fe-2 -- where the front end thread runs (d3d11.blessedFrontEndCpu): off skyrim's render thread's core and its smt sibling
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "blessed_threaded_context.h"

namespace dxvk {

  /// Logical processor masks of each physical core, processor group 0
  static std::vector<KAFFINITY> BlessedCoreMasks() {
    std::vector<KAFFINITY> cores;
    DWORD size = 0u;

    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size);

    if (!size)
      return cores;

    std::vector<char> data(size);
    auto first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(data.data());

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &size))
      return cores;

    for (DWORD offset = 0u; offset < size; ) {
      auto info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(data.data() + offset);

      if (info->Relationship == RelationProcessorCore) {
        for (WORD i = 0; i < info->Processor.GroupCount; i++) {
          if (info->Processor.GroupMask[i].Group == 0u && info->Processor.GroupMask[i].Mask)
            cores.push_back(info->Processor.GroupMask[i].Mask);
        }
      }

      offset += info->Size;
    }

    return cores;
  }


  static std::string BlessedHex(uint64_t Value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(Value));
    return text;
  }


  static uint32_t BlessedLowestBit(KAFFINITY Mask) {
    uint32_t bit = 0u;

    while (bit < 63u && !(Mask & (KAFFINITY(1u) << bit)))
      bit += 1u;

    return bit;
  }


  void D3D11ThreadedContext::PlaceFrontEnd() {
    // Game side, at each recorded or drained Present: one compare unless
    // the recording thread changed (the loading screen hands over to the
    // render thread, for one)
    DWORD thread = GetCurrentThreadId();

    if (likely(thread == m_placedFor))
      return;

    m_placedFor = thread;

    const std::string& option = m_parent->GetOptions()->blessedFrontEndCpu;
    HANDLE handle = m_thread.native_handle();

    if (option.empty() || !handle)
      return;

    if (option.rfind("mask:", 0u) == 0u) {
      KAFFINITY mask = KAFFINITY(std::strtoull(option.c_str() + 5u, nullptr, 16));

      if (!mask || !SetThreadAffinityMask(handle, mask)) {
        Logger::warn(str::format("d3d11.blessedFrontEndCpu: affinity mask ", option.c_str() + 5u, " refused"));
        return;
      }

      Logger::info(str::format("d3d11.blessedFrontEndCpu: front end affinity mask ", BlessedHex(mask)));
      return;
    }

    if (option != "auto" && option != "auto-mask") {
      char* end = nullptr;
      unsigned long number = std::strtoul(option.c_str(), &end, 10);

      if (!end || *end || number >= 64u) {
        Logger::warn(str::format("d3d11.blessedFrontEndCpu: unknown value '", option, "'"));
        return;
      }

      PROCESSOR_NUMBER ideal = { };
      ideal.Group  = 0u;
      ideal.Number = BYTE(number);

      if (SetThreadIdealProcessorEx(handle, &ideal, nullptr))
        Logger::info(str::format("d3d11.blessedFrontEndCpu: front end ideal processor ", number));
      return;
    }

    // auto: find the core of the recording thread's ideal processor (the
    // one the scheduler prefers for it), then keep the front end off it
    std::vector<KAFFINITY> cores = BlessedCoreMasks();
    PROCESSOR_NUMBER gameIdeal = { };

    if (cores.size() < 2u || !GetThreadIdealProcessorEx(GetCurrentThread(), &gameIdeal) || gameIdeal.Group) {
      Logger::warn("d3d11.blessedFrontEndCpu: auto needs one processor group with two or more cores");
      return;
    }

    size_t gameCore = cores.size();

    for (size_t i = 0; i < cores.size(); i++) {
      if (cores[i] & (KAFFINITY(1u) << gameIdeal.Number))
        gameCore = i;
    }

    if (gameCore == cores.size())
      return;

    if (option == "auto-mask") {
      KAFFINITY mask = 0u;

      for (size_t i = 0; i < cores.size(); i++) {
        if (i != gameCore)
          mask |= cores[i];
      }

      if (SetThreadAffinityMask(handle, mask)) {
        Logger::info(str::format("d3d11.blessedFrontEndCpu: recording thread ideal cpu ", uint32_t(gameIdeal.Number),
          " (core ", gameCore, "), front end affinity mask ", BlessedHex(mask)));
      }

      return;
    }

    // The core half way round from the recording thread's
    size_t feCore = (gameCore + cores.size() / 2u) % cores.size();

    PROCESSOR_NUMBER ideal = { };
    ideal.Group  = 0u;
    ideal.Number = BYTE(BlessedLowestBit(cores[feCore]));

    if (SetThreadIdealProcessorEx(handle, &ideal, nullptr)) {
      Logger::info(str::format("d3d11.blessedFrontEndCpu: recording thread ideal cpu ", uint32_t(gameIdeal.Number),
        " (core ", gameCore, "), front end ideal cpu ", uint32_t(ideal.Number), " (core ", feCore, ")"));
    }
  }

}
