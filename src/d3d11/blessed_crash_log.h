// blessed: crash-log -- the unhandled-exception filter's re-arm point (see blessed_crash_log.cpp)
#pragma once

#include <cstddef>
#include <cstdint>

namespace dxvk {

  /**
   * \brief Re-arms the crash log's unhandled-exception filter
   *
   * Every 30th call puts our filter back in front if the game, skse or a
   * plugin installed its own since, and chains to it. The first call also
   * opens the dxvk log for the crash report and names the calling thread.
   * Call once per present: a counter check when nothing is due.
   */
  void BlessedCrashLogOnPresent();

  /**
   * \brief blessed: fe-crash-2 -- the crash report's no-allocation text writer
   *
   * Safe inside an exception filter: fixed buffer, no heap, no locks.
   */
  struct BlessedCrashWriter {
    char*  buf;
    size_t cap;
    size_t len;

    void Put(const char* s) {
      while (*s && len + 1 < cap)
        buf[len++] = *s++;
    }

    void PutHex(uint64_t v) {
      char tmp[20];
      int n = 0;
      do {
        uint32_t d = uint32_t(v & 0xfu);
        tmp[n++] = char(d < 10u ? '0' + d : 'a' + d - 10u);
        v >>= 4;
      } while (v && n < 16);
      Put("0x");
      while (n > 0 && len + 1 < cap)
        buf[len++] = tmp[--n];
    }

    void PutDec(uint64_t v) {
      char tmp[24];
      int n = 0;
      do {
        tmp[n++] = char('0' + v % 10u);
        v /= 10u;
      } while (v && n < 20);
      while (n > 0 && len + 1 < cap)
        buf[len++] = tmp[--n];
    }

    /// Two hex digits, no prefix
    void PutByte(uint8_t v) {
      const char* digits = "0123456789abcdef";
      if (len + 2 < cap) {
        buf[len++] = digits[v >> 4];
        buf[len++] = digits[v & 0xfu];
      }
    }

    /// "module.dll+0x1234", or the bare address outside any module
    void PutAddr(uint64_t addr, char* pathBuf, uint32_t pathCap);

    /// Appends the text to a file handle (a HANDLE), flushed
    void WriteTo(void* file) const;
  };

  /**
   * \brief blessed: fe-crash-2 -- an extra section for the last-chance report
   *
   * Runs inside the unhandled-exception filter, after the main report is
   * already on disk, so a fault inside it loses only itself. Must not
   * allocate or lock. \c pUser is whatever was registered with it.
   */
  using BlessedCrashExtraFn = void (*)(void* pUser, BlessedCrashWriter& Writer, uint32_t ThreadId, const char* pThreadName);

  /**
   * \brief Registers (or, with \c nullptr, clears) the extra section
   *
   * One slot. Cleared only by the owner that set it.
   */
  void BlessedCrashLogSetExtra(BlessedCrashExtraFn pfnExtra, void* pUser);

  void BlessedCrashLogClearExtra(void* pUser);

}
