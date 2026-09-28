// blessed: fe-crash-2 -- the front end's section of the last-chance crash report: ring, packet, op walk, raw bytes, publish and thread-switch logs
//
// Runs inside the unhandled-exception filter (blessed_crash_log.cpp), so
// no heap, no locks, no Logger. Every pointer it follows is the ring's
// own memory, clamped to the ring; nothing it reads is dereferenced
// beyond that. The producer may still be running while this writes, so
// producer-side numbers are a snapshot taken a few microseconds after
// the fault, not at it; the FE-side ones (range, packet probe) are exact.
#include "blessed_crash_log.h"
#include "blessed_threaded_context.h"
#include "blessed_threaded_packet.h"

namespace dxvk {

  FeReplayProbe g_feReplayProbe = { };


  /// Op size by code, 0 for an unknown code (same table as the checker)
  static uint32_t BlessedFeForensicOpSize(uint8_t op) {
    switch (FeOp(op)) {
      case FeOp::CbRename:             return uint32_t(sizeof(FeOpCbRename));
      case FeOp::SetCb:
      case FeOp::SetSrv:
      case FeOp::SetSampler:
      case FeOp::SetShader:
      case FeOp::SetInputLayout:       return uint32_t(sizeof(FeOpPtr));
      case FeOp::SetVb:                return uint32_t(sizeof(FeOpVb));
      case FeOp::SetIb:                return uint32_t(sizeof(FeOpIb));
      case FeOp::SetTopology:          return uint32_t(sizeof(FeOpTopology));
      case FeOp::Draw:
      case FeOp::DrawIndexed:          return uint32_t(sizeof(FeOpDraw));
      case FeOp::DrawInstanced:
      case FeOp::DrawIndexedInstanced: return uint32_t(sizeof(FeOpDrawInstanced));
    }

    return 0u;
  }


  void D3D11ThreadedContext::WriteCrashForensics(void* pUser, BlessedCrashWriter& Writer, uint32_t ThreadId, const char* pThreadName) {
    static_cast<D3D11ThreadedContext*>(pUser)->WriteCrashForensics(Writer, ThreadId, pThreadName);
  }


  void D3D11ThreadedContext::WriteCrashForensics(BlessedCrashWriter& w, uint32_t ThreadId, const char* pThreadName) {
    constexpr uint64_t Mask = uint64_t(RingSize - 1u);

    bool onFrontEnd = pThreadName && pThreadName[0] == 'd' && pThreadName[1] == 'x'
      && pThreadName[2] == 'v' && pThreadName[3] == 'k' && pThreadName[4] == '-'
      && pThreadName[5] == 'f' && pThreadName[6] == 'e' && !pThreadName[7];

    uint64_t ringBase = reinterpret_cast<uint64_t>(m_ring);

    // --- the ring, both sides ---
    w.Put("err:   blessed: fe forensics (faulting thread ");
    w.PutDec(ThreadId);
    w.Put(onFrontEnd ? " is the front end)\n" : " is not the front end; producer state only is meaningful)\n");

    w.Put("err:   blessed:   ring base=");
    w.PutHex(ringBase);
    w.Put(" size=");
    w.PutHex(RingSize);
    w.Put(" writePos=");
    w.PutHex(m_writePos);
    w.Put(" writeLimit=");
    w.PutHex(m_writeLimit);
    w.Put(" publishedLocal=");
    w.PutHex(m_publishedLocal);
    w.Put(" published=");
    w.PutHex(m_published.load(std::memory_order_relaxed));
    w.Put(" replayed=");
    w.PutHex(m_replayed.load(std::memory_order_relaxed));
    w.Put("\n");

    w.Put("err:   blessed:   open packet=");
    w.PutHex(reinterpret_cast<uint64_t>(m_packet));
    w.Put(" packetEnd=");
    w.PutHex(m_packetEnd);
    w.Put(" pending=");
    w.PutDec(m_packetPending);
    w.Put(" unpublished=");
    w.PutDec(m_unpublished);
    w.Put(" publishes=");
    w.PutDec(m_publishes);
    w.Put(" records=");
    w.PutDec(m_records);
    w.Put(" packetOps=");
    w.PutDec(m_packetOps);
    w.Put(" recorder=");
    w.PutDec(m_producerThreadId);
    w.Put(" threadSwitches=");
    w.PutDec(m_threadSwitches);
    w.Put(" foreignPublishes=");
    w.PutDec(m_foreignPublishes);
    w.Put("\n");

    // --- the packet the front end was replaying ---
    uint64_t feBegin = m_feBegin;
    uint64_t feEnd   = m_feEnd;

    w.Put("err:   blessed:   front end range begin=");
    w.PutHex(feBegin);
    w.Put(" end=");
    w.PutHex(feEnd);
    w.Put("\n");

    const FePacketRec* rec = g_feReplayProbe.rec;
    uint64_t recAddr = reinterpret_cast<uint64_t>(rec);

    if (onFrontEnd && rec && recAddr >= ringBase && recAddr + sizeof(FePacketRec) <= ringBase + RingSize) {
      uint64_t offset = recAddr - ringBase;

      // Absolute position: the one in [feBegin, feBegin + RingSize) that
      // lands on this slot
      uint64_t pos = (feBegin & ~Mask) + offset;

      if (pos < feBegin)
        pos += RingSize;

      uint32_t sizeRead  = g_feReplayProbe.size;
      uint32_t countRead = g_feReplayProbe.count;

      w.Put("err:   blessed:   packet at ring+");
      w.PutHex(offset);
      w.Put(" pos=");
      w.PutHex(pos);
      w.Put(" size read=");
      w.PutDec(sizeRead);
      w.Put(" count read=");
      w.PutDec(countRead);
      w.Put(" | header now: size=");
      w.PutDec(rec->size);
      w.Put(" count=");
      w.PutDec(rec->count);
      w.Put(" fn=");
      w.PutHex(reinterpret_cast<uint64_t>(rec->fn));
      w.Put(offset + sizeRead > RingSize ? " | PACKET CROSSES THE RING END" : "");
      w.Put(pos + sizeRead > feEnd ? " | PACKET RUNS PAST THE PUBLISHED END" : "");
      w.Put(pos >= feEnd ? " | PACKET STARTS AT OR PAST THE PUBLISHED END" : "");
      w.Put(m_writePos > pos + RingSize ? " | WRITER HAS LAPPED THIS SLOT" : "");
      w.Put(recAddr == reinterpret_cast<uint64_t>(m_packet) ? " | IS THE PRODUCER'S CURRENT PACKET" : "");
      w.Put("\n");

      // Walk the ops as the replay did, by the size it read, and stop at
      // the first op the replay would have choked on
      uint64_t limit = std::min<uint64_t>(sizeRead, RingSize - offset);
      uint64_t at = sizeof(FePacketRec);
      uint64_t badAt = ~0ull;
      const char* why = nullptr;
      uint32_t walked = 0u;
      const char* base = reinterpret_cast<const char*>(rec);

      w.Put("err:   blessed:   ops (offset:op):");

      while (at + sizeof(FeOpHeader) <= limit && walked < 256u) {
        uint8_t op = uint8_t(base[at]);
        uint32_t opSize = BlessedFeForensicOpSize(op);

        w.Put(" ");
        w.PutDec(at);
        w.Put(":");
        w.PutDec(op);

        if (!opSize) {
          badAt = at;
          why = "unknown op code";
          break;
        }

        if (at + opSize > sizeRead) {
          badAt = at;
          why = "op runs past the packet's size";
          break;
        }

        if (FeOp(op) == FeOp::CbRename && at + opSize <= limit) {
          auto cb = reinterpret_cast<const FeOpCbRename*>(base + at);

          if (!cb->block || !cb->buffer) {
            badAt = at;
            why = !cb->block ? "CbRename with a null block" : "CbRename with a null buffer";
            break;
          }
        }

        at += opSize;
        walked += 1u;
      }

      w.Put("\n");

      if (why) {
        w.Put("err:   blessed:   first bad op at packet offset ");
        w.PutDec(badAt);
        w.Put(": ");
        w.Put(why);
        w.Put(" (ops before it: ");
        w.PutDec(walked);
        w.Put(")\n");
      } else {
        w.Put("err:   blessed:   no bad op in the walk now (the bytes may have changed since the fault)\n");
        badAt = std::min<uint64_t>(at, sizeRead);
      }

      // Raw bytes: 64 before the bad op (or the packet start) to 64 past
      // it, clamped to the ring. Rows of 16, offsets relative to the packet.
      uint64_t first = badAt > 64u ? badAt - 64u : 0u;
      uint64_t last  = std::min<uint64_t>(badAt + 64u, RingSize - offset);
      first &= ~uint64_t(15u);

      for (uint64_t row = first; row < last; row += 16u) {
        w.Put("err:   blessed:   +");
        w.PutHex(row);
        w.Put(":");

        for (uint64_t i = row; i < row + 16u && i < last; i++) {
          w.Put(i == row + 8u ? "  " : " ");
          w.PutByte(uint8_t(base[i]));
        }

        w.Put(row <= sizeRead && sizeRead < row + 16u ? "   <- size read ends in this row\n" : "\n");
      }
    } else if (onFrontEnd) {
      w.Put("err:   blessed:   no packet in replay (probe ");
      w.PutHex(recAddr);
      w.Put(")\n");
    }

    // --- the last publishes, oldest first ---
    uint64_t publishes = m_publishes;
    uint64_t logged = std::min<uint64_t>(publishes, PublishLogSize);

    for (uint64_t i = publishes - logged; i < publishes; i++) {
      const FePublishEvent& e = m_publishLog[i & (PublishLogSize - 1u)];
      uint64_t packet = reinterpret_cast<uint64_t>(e.packet);

      w.Put("err:   blessed:   publish #");
      w.PutDec(i);
      w.Put(" pos=");
      w.PutHex(e.writePos);
      w.Put(" prev=");
      w.PutHex(e.prevPublished);
      w.Put(" packet=ring+");
      w.PutHex(packet >= ringBase ? packet - ringBase : 0u);
      w.Put(" size=");
      w.PutDec(e.size);
      w.Put(" count=");
      w.PutDec(e.count);
      w.Put(" packetEnd=");
      w.PutHex(e.packetEnd);
      w.Put(" pending=");
      w.PutDec(e.pending);
      w.Put(" thread=");
      w.PutDec(e.thread);
      w.Put(e.pending ? "  <- MID-OP PUBLISH" : "");
      w.Put(e.thread != m_producerThreadId ? "  <- NOT THE RECORDER" : "");
      w.Put("\n");
    }

    // --- recording thread switches, oldest first ---
    uint32_t switches = m_threadSwitches;
    uint32_t kept = std::min<uint32_t>(switches, SwitchLogSize);

    for (uint32_t i = switches - kept; i < switches; i++) {
      const FeSwitchEvent& e = m_switchLog[i & (SwitchLogSize - 1u)];

      w.Put("err:   blessed:   switch #");
      w.PutDec(i + 1u);
      w.Put(" thread ");
      w.PutDec(e.from);
      w.Put(" -> ");
      w.PutDec(e.to);
      w.Put(" at writePos=");
      w.PutHex(e.writePos);
      w.Put(" publish=");
      w.PutDec(e.publishes);
      w.Put(e.packetEnd != ~0ull ? " packet open\n" : " packet closed\n");
    }
  }

}
