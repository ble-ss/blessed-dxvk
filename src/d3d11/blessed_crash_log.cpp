// blessed: crash-log -- std::terminate, abort() and unhandled exceptions leave their story in the dxvk log before the process dies
//
// Why: perf runs since blessed 4f507ba7 exit silently. c28-perf-r1 caught
// the exit code, 0xC0000005, with no WER event and nothing in the dxvk log.
// So three paths, all on by default, all free until something goes wrong
// (BLESSED_CRASH_LOG=0 turns them off):
//
// - std::terminate and SIGABRT: one line each (an exception escaping one of
//   our threads, or an abort()). These may allocate: the heap is fine then.
// - the last chance: an unhandled-exception filter, installed at load and
//   re-armed every 30 presents (the game, skse or a plugin may install
//   theirs later; ours chains to whatever it replaced). It writes the code,
//   the faulting address as module+offset, the access type and target, the
//   thread id and name, the registers and 32 unwound frames, without
//   allocating, to the dxvk log and to <exe>_d3d11_crash.log next to it.
// - first chance: a vectored handler records the first 32 fatal-type
//   exceptions (access violations and the like) to
//   <exe>_d3d11_firstchance.log, marked as first chance: the game may have
//   handled them. It never touches the dxvk log and never changes the
//   outcome. It covers a handler that catches the fault and then kills
//   the process without reaching the filter.
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <typeinfo>

#include <windows.h>

#include "blessed_crash_log.h"

#include "../util/util_env.h"
#include "../util/util_error.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // ---------------- std::terminate / SIGABRT ----------------

    std::terminate_handler g_previousTerminate = nullptr;

    std::string DescribeCurrentException() {
      std::exception_ptr e = std::current_exception();
      if (!e)
        return "no active exception (std::terminate called directly)";

      try {
        std::rethrow_exception(e);
      } catch (const DxvkError& err) {
        return "DxvkError: " + err.message();
      } catch (const std::exception& err) {
        return str::format(typeid(err).name(), ": ", err.what());
      } catch (...) {
        return "an exception of unknown type";
      }
    }

    void OnTerminate() {
      Logger::err(str::format("blessed: std::terminate on thread ", GetCurrentThreadId(),
        ": ", DescribeCurrentException()));

      if (g_previousTerminate)
        g_previousTerminate();
      std::abort();
    }

    void OnAbort(int) {
      Logger::err(str::format("blessed: abort() on thread ", GetCurrentThreadId()));
    }

#if defined(_WIN64)

    // ---------------- no-allocation text writer ----------------

    // blessed: fe-crash-2 -- the writer lives in blessed_crash_log.h now,
    // so an extra section (the front end's forensics) can share it
    using Writer = BlessedCrashWriter;

    // blessed: fe-crash-2 -- one extra section for the last-chance report
    BlessedCrashExtraFn volatile g_extraFn   = nullptr;
    void* volatile               g_extraUser = nullptr;

    const char* CodeName(DWORD code) {
      switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:         return "access violation";
        case EXCEPTION_IN_PAGE_ERROR:            return "in-page error";
        case EXCEPTION_STACK_OVERFLOW:           return "stack overflow";
        case EXCEPTION_ILLEGAL_INSTRUCTION:      return "illegal instruction";
        case EXCEPTION_PRIV_INSTRUCTION:         return "privileged instruction";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:       return "integer divide by zero";
        case EXCEPTION_INT_OVERFLOW:             return "integer overflow";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:    return "array bounds exceeded";
        case EXCEPTION_DATATYPE_MISALIGNMENT:    return "datatype misalignment";
        case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "noncontinuable exception";
        case EXCEPTION_INVALID_DISPOSITION:      return "invalid disposition";
        case 0xC0000374u:                        return "heap corruption";
        case 0xC0000409u:                        return "stack buffer overrun / fast fail";
        case 0xE06D7363u:                        return "msvc c++ exception";
        case 0x20474343u:                        return "gcc/clang c++ exception";
        default:                                 return "other";
      }
    }

    bool IsFatalType(DWORD code) {
      switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
        case EXCEPTION_IN_PAGE_ERROR:
        case EXCEPTION_STACK_OVERFLOW:
        case EXCEPTION_ILLEGAL_INSTRUCTION:
        case EXCEPTION_PRIV_INSTRUCTION:
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        case EXCEPTION_DATATYPE_MISALIGNMENT:
        case 0xC0000374u:
          return true;
        default:
          return false;
      }
    }

    // one report's scratch: static, so a stack overflow still has room
    struct ReportScratch {
      char    text[16384];
      char    path[MAX_PATH];
      CONTEXT ctx;
    };

    ReportScratch g_lastScratch;   // the unhandled filter
    ReportScratch g_firstScratch;  // the first-chance handler

    HANDLE g_dxvkLog    = INVALID_HANDLE_VALUE;  // opened at the first present
    HANDLE g_crashFile  = INVALID_HANDLE_VALUE;  // opened at load
    HANDLE g_firstFile  = INVALID_HANDLE_VALUE;  // opened at load

    LPTOP_LEVEL_EXCEPTION_FILTER g_previousFilter = nullptr;
    LPTOP_LEVEL_EXCEPTION_FILTER g_originalFilter = nullptr;  // the one before ours at load
    volatile LONG g_filterDepth   = 0;
    volatile LONG g_reported      = 0;
    volatile LONG g_firstBusy     = 0;
    volatile LONG g_firstCount    = 0;
    bool          g_enabled       = false;
    uint32_t      g_presents      = 0;

    void Report(ReportScratch& s, const EXCEPTION_POINTERS* ep, const char* kind, uint32_t frames, bool registers,
                HANDLE fileA, HANDLE fileB) {
      Writer w = { s.text, sizeof(s.text), 0 };
      const EXCEPTION_RECORD* er = ep->ExceptionRecord;
      DWORD tid = GetCurrentThreadId();
      const char* name = env::blessedThreadName(uint32_t(tid));

      w.Put("err:   blessed: ");
      w.Put(kind);
      w.Put(" exception ");
      w.PutHex(er->ExceptionCode);
      w.Put(" (");
      w.Put(CodeName(er->ExceptionCode));
      w.Put(") at ");
      w.PutAddr(reinterpret_cast<uint64_t>(er->ExceptionAddress), s.path, MAX_PATH);

      if ((er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION || er->ExceptionCode == EXCEPTION_IN_PAGE_ERROR)
       && er->NumberParameters >= 2) {
        ULONG_PTR op = er->ExceptionInformation[0];
        w.Put(op == 0 ? ", read of " : op == 1 ? ", write to " : op == 8 ? ", execute (dep) at " : ", access to ");
        w.PutAddr(er->ExceptionInformation[1], s.path, MAX_PATH);
      }

      w.Put(", thread ");
      w.PutDec(tid);
      w.Put(" (");
      w.Put(name ? name : "unnamed");
      w.Put(")\n");

      const CONTEXT* c = ep->ContextRecord;

      if (registers && c) {
        struct { const char* n; DWORD64 v; } regs[] = {
          { "rip", c->Rip }, { "rsp", c->Rsp }, { "rbp", c->Rbp }, { "rax", c->Rax },
          { "rbx", c->Rbx }, { "rcx", c->Rcx }, { "rdx", c->Rdx }, { "rsi", c->Rsi },
          { "rdi", c->Rdi }, { "r8",  c->R8  }, { "r9",  c->R9  }, { "r10", c->R10 },
          { "r11", c->R11 }, { "r12", c->R12 }, { "r13", c->R13 }, { "r14", c->R14 },
          { "r15", c->R15 } };
        w.Put("err:   blessed:  ");
        for (const auto& r : regs) {
          w.Put(" ");
          w.Put(r.n);
          w.Put("=");
          w.PutHex(r.v);
        }
        w.Put("\n");
      }

      if (frames && c) {
        // unwind a copy of the faulting context; the stack bounds come from
        // this thread's tib (the filter runs on the faulting thread)
        s.ctx = *c;
        const NT_TIB* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
        uint64_t lo = reinterpret_cast<uint64_t>(tib->StackLimit);
        uint64_t hi = reinterpret_cast<uint64_t>(tib->StackBase);

        for (uint32_t i = 0; i < frames; i++) {
          DWORD64 pc = s.ctx.Rip;
          if (!pc)
            break;

          w.Put("err:   blessed:   #");
          if (i < 10u)
            w.Put("0");
          w.PutDec(i);
          w.Put(" ");
          w.PutAddr(pc, s.path, MAX_PATH);
          w.Put("\n");

          DWORD64 sp = s.ctx.Rsp;
          if (sp < lo || sp + 8u > hi)
            break;

          DWORD64 imageBase = 0;
          PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(pc, &imageBase, nullptr);

          if (fn) {
            PVOID handlerData = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, pc, fn, &s.ctx, &handlerData, &establisher, nullptr);
          } else {
            // a leaf, or a jump to nowhere: the return address is on top
            s.ctx.Rip = *reinterpret_cast<const DWORD64*>(sp);
            s.ctx.Rsp = sp + 8u;
          }

          if (s.ctx.Rip == pc && s.ctx.Rsp == sp)
            break;
        }
      }

      w.WriteTo(fileA);
      w.WriteTo(fileB);
      if (w.len) {
        s.text[w.len] = '\0';
        OutputDebugStringA(s.text);
      }

      // blessed: fe-crash-2 -- the extra section, last chance only, after
      // the main report is on disk: a fault in here loses only itself
      BlessedCrashExtraFn extra = g_extraFn;

      if (registers && extra) {
        Writer x = { s.text, sizeof(s.text), 0 };
        extra(g_extraUser, x, uint32_t(tid), name);
        x.WriteTo(fileA);
        x.WriteTo(fileB);
      }
    }

    LONG WINAPI OnUnhandled(EXCEPTION_POINTERS* ep) {
      // a filter we replaced may chain back to us: say nothing twice, and
      // hand that second visit to the filter that was there before ours at
      // load, so the chain still ends where it would have
      LONG depth = InterlockedIncrement(&g_filterDepth);
      if (depth != 1) {
        LONG result = EXCEPTION_CONTINUE_SEARCH;
        LPTOP_LEVEL_EXCEPTION_FILTER original = g_originalFilter;
        if (depth == 2 && original && original != &OnUnhandled && original != g_previousFilter)
          result = original(ep);
        InterlockedDecrement(&g_filterDepth);
        return result;
      }

      if (!InterlockedExchange(&g_reported, 1))
        Report(g_lastScratch, ep, "LAST CHANCE (unhandled)", 32u, true, g_dxvkLog, g_crashFile);

      LONG result = EXCEPTION_CONTINUE_SEARCH;
      LPTOP_LEVEL_EXCEPTION_FILTER previous = g_previousFilter;
      if (previous && previous != &OnUnhandled)
        result = previous(ep);

      InterlockedDecrement(&g_filterDepth);
      return result;
    }

    LONG CALLBACK OnFirstChance(EXCEPTION_POINTERS* ep) {
      if (!IsFatalType(ep->ExceptionRecord->ExceptionCode))
        return EXCEPTION_CONTINUE_SEARCH;

      if (InterlockedIncrement(&g_firstCount) > 32)
        return EXCEPTION_CONTINUE_SEARCH;

      if (InterlockedExchange(&g_firstBusy, 1))
        return EXCEPTION_CONTINUE_SEARCH;

      Report(g_firstScratch, ep, "first chance (the game may handle it)", 16u, false, g_firstFile, INVALID_HANDLE_VALUE);

      InterlockedExchange(&g_firstBusy, 0);
      return EXCEPTION_CONTINUE_SEARCH;
    }

    HANDLE OpenForAppend(const std::string& path, bool truncate) {
      if (path.empty())
        return INVALID_HANDLE_VALUE;
      return CreateFileA(path.c_str(), FILE_APPEND_DATA | SYNCHRONIZE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        truncate ? CREATE_ALWAYS : OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    }

    // where Logger puts <exe>_<base> (see Logger::getFileName)
    std::string LogPath(const char* base) {
      std::string path = env::getEnvVar("DXVK_LOG_PATH");
      if (path == "none")
        return std::string();
      if (!path.empty() && *path.rbegin() != '/' && *path.rbegin() != '\\')
        path += '/';
      return path + env::getExeBaseName() + "_" + base;
    }

    void InstallFilters() {
      g_crashFile = OpenForAppend(LogPath("d3d11_crash.log"), true);
      g_firstFile = OpenForAppend(LogPath("d3d11_firstchance.log"), true);
      g_previousFilter = SetUnhandledExceptionFilter(&OnUnhandled);
      g_originalFilter = g_previousFilter;
      AddVectoredExceptionHandler(0u, &OnFirstChance); // never removed: lives as long as the process
      env::blessedRecordThreadName(uint32_t(GetCurrentThreadId()), "dll-load");
    }

#endif // _WIN64

    struct CrashLogInstaller {
      CrashLogInstaller() {
        // env only here: no Logger call at static init
        if (env::getEnvVar("BLESSED_CRASH_LOG") == "0")
          return;
        g_previousTerminate = std::set_terminate(&OnTerminate);
        std::signal(SIGABRT, &OnAbort);
#if defined(_WIN64)
        g_enabled = true;
        InstallFilters();
#endif
      }
    };

    CrashLogInstaller g_crashLogInstaller;

  }


#if defined(_WIN64)
  void BlessedCrashWriter::PutAddr(uint64_t addr, char* pathBuf, uint32_t pathCap) {
    HMODULE mod = nullptr;
    if (addr && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                                 | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(addr), &mod) && mod) {
      DWORD n = GetModuleFileNameA(mod, pathBuf, DWORD(pathCap));
      const char* base = pathBuf;
      for (DWORD i = 0; i < n; i++) {
        if (pathBuf[i] == '\\' || pathBuf[i] == '/')
          base = pathBuf + i + 1;
      }
      Put(n ? base : "?");
      Put("+");
      PutHex(addr - reinterpret_cast<uint64_t>(mod));
    } else {
      PutHex(addr);
    }
  }


  void BlessedCrashWriter::WriteTo(void* file) const {
    if (file == INVALID_HANDLE_VALUE || !len)
      return;
    DWORD written = 0;
    WriteFile(file, buf, DWORD(len), &written, nullptr);
    FlushFileBuffers(file);
  }
#else
  void BlessedCrashWriter::PutAddr(uint64_t addr, char*, uint32_t) {
    PutHex(addr);
  }


  void BlessedCrashWriter::WriteTo(void*) const {
  }
#endif


  void BlessedCrashLogSetExtra(BlessedCrashExtraFn pfnExtra, void* pUser) {
    g_extraUser = pUser;
    g_extraFn   = pfnExtra;
  }


  void BlessedCrashLogClearExtra(void* pUser) {
    if (g_extraUser == pUser) {
      g_extraFn   = nullptr;
      g_extraUser = nullptr;
    }
  }


  void BlessedCrashLogOnPresent() {
#if defined(_WIN64)
    if (!g_enabled || (g_presents++ % 30u) != 0u)
      return;

    if (g_presents == 1u) {
      // the dxvk log exists by now; append behind whatever Logger wrote
      g_dxvkLog = OpenForAppend(LogPath("d3d11.log"), false);
      DWORD tid = GetCurrentThreadId();
      if (!env::blessedThreadName(uint32_t(tid)))
        env::blessedRecordThreadName(uint32_t(tid), "present");
    }

    LPTOP_LEVEL_EXCEPTION_FILTER previous = SetUnhandledExceptionFilter(&OnUnhandled);

    if (previous != &OnUnhandled) {
      g_previousFilter = previous;

      char text[256];
      char path[MAX_PATH];
      Writer w = { text, sizeof(text), 0 };
      w.Put("blessed: crash-log: re-armed the unhandled-exception filter over ");
      w.PutAddr(reinterpret_cast<uint64_t>(previous), path, MAX_PATH);
      text[w.len] = '\0';
      Logger::info(text);
    }
#endif
  }

}
