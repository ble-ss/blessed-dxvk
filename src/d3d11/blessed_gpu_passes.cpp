// blessed: per-pass gpu timestamps at render-target, dispatch and transfer boundaries (BLESSED_GPU_PASSES)
#include "blessed_gpu_passes.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "d3d11_texture.h"
#include "d3d11_view_dsv.h"
#include "d3d11_view_rtv.h"

#include "../dxvk/blessed/blessed_gpu_gaps.h"

#include "../util/util_env.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    constexpr uint32_t ModeOff    = 0u;
    constexpr uint32_t ModePasses = 1u;
    constexpr uint32_t ModeFrame  = 2u;

    constexpr uint32_t RingSize     = 8u;   // frames in flight or waiting for readback
    constexpr uint32_t MinAge       = 3u;   // presents before a frame is read back
    constexpr uint32_t WindowFrames = 120u;
    constexpr uint32_t MaxHashes    = 4u;

    // pass kinds as written to the json (Frame and Present are output-only)
    enum PassKind : uint32_t {
      KindNone    = 0u,
      KindDraw    = 1u,
      KindCompute = 2u,
      KindXfer    = 3u,
      KindFrame   = 4u,
      KindPresent = 5u,
    };

    const char* KindName(uint32_t kind) {
      switch (kind) {
        case KindDraw:    return "draw";
        case KindCompute: return "compute";
        case KindXfer:    return "xfer";
        case KindFrame:   return "frame";
        case KindPresent: return "present";
        default:          return "none";
      }
    }

    struct TargetInfo {
      uint32_t slot = 0u;
      uint32_t fmt  = 0u;
      uint32_t w    = 0u;
      uint32_t h    = 0u;
    };

    struct PassInfo {
      uint32_t                   kind       = KindNone;
      uint32_t                   rtvCount   = 0u;
      std::array<TargetInfo, 8>  rtv        = { };
      bool                       hasDsv     = false;
      TargetInfo                 dsv        = { };
      uint32_t                   draws      = 0u;
      uint32_t                   dispatches = 0u;
      uint32_t                   xfer       = 0u;
      uint32_t                   hashCount  = 0u;
      std::array<const void*, MaxHashes> hashKeys = { };
      std::array<std::string, MaxHashes> hashes;
    };

    enum class SlotState : uint32_t { Free, Recording, Pending };

    struct FrameSlot {
      SlotState                   state   = SlotState::Free;
      uint64_t                    present = 0u;
      std::vector<Rc<DxvkQuery>>  queries;
      uint32_t                    used    = 0u;   // marks handed out this frame
      std::atomic<uint32_t>       written = { 0u };
      std::vector<PassInfo>       passes;
      bool                        ended   = false;
      bool                        blit    = false;
    };

    // one resolved frame, kept until its window is written
    struct ResolvedFrame {
      std::string            signature;
      std::vector<PassInfo>  passes;
      std::vector<double>    ms;       // per pass, including the present pass
      double                 frameMs = 0.0;
      uint64_t               tsFirst = 0u;   // raw ticks: first pass start
      uint64_t               tsEnd   = 0u;   // frame end mark
      uint64_t               tsBlit  = 0u;   // after the present blit (tsEnd if none)
    };

    // ---- gaps: the frame outside the app's passes (BlessedGpuGaps) ----

    constexpr uint32_t WaitKinds = uint32_t(BlessedGapWait::Count);

    const char* WaitName(uint32_t wait) {
      switch (BlessedGapWait(wait)) {
        case BlessedGapWait::None:    return "none";
        case BlessedGapWait::Acquire: return "acquire";
        case BlessedGapWait::Async:   return "async";
        case BlessedGapWait::Sdma:    return "sdma";
        case BlessedGapWait::Fence:   return "fence";
        case BlessedGapWait::Sparse:  return "sparse";
        default:                      return "other";
      }
    }

    // a resolved frame's pass span, waiting for its gap frame
    struct GapSpan {
      uint64_t first = 0u;
      uint64_t end   = 0u;
      uint64_t blit  = 0u;
      uint32_t age   = 0u;
    };

    // sums over the window's matched frames, in ticks unless noted
    struct GapAcc {
      uint32_t  frames      = 0u;
      uint32_t  unmatched   = 0u;
      uint64_t  frame       = 0u;
      uint64_t  idle        = 0u;
      std::array<uint64_t, WaitKinds> idleBy = { };
      uint64_t  idleInPasses = 0u;
      uint64_t  passesSpan  = 0u;
      uint64_t  passes      = 0u;
      uint64_t  present     = 0u;
      uint64_t  init        = 0u;
      uint64_t  sdma        = 0u;
      uint64_t  execPre     = 0u;
      uint64_t  execPost    = 0u;
      uint64_t  initInPasses = 0u;
      uint64_t  sdmaQueue   = 0u;
      uint64_t  cmdLists    = 0u;   // counts
      uint64_t  submits     = 0u;
      uint64_t  sdmaChunks  = 0u;
      uint64_t  barriers    = 0u;
      uint64_t  renderPasses = 0u;
      // the gap right before a frame's first graphics buffer, split by
      // whether that frame's first submit came after the gpu ran dry
      uint64_t  idleStart       = 0u;
      uint64_t  idleStartLate   = 0u;
      uint64_t  idleStartQueued = 0u;
      uint32_t  framesLate      = 0u;
      uint32_t  framesQueued    = 0u;
      uint32_t  presentFrames   = 0u;   // frames with a present call time
      uint64_t  presentCallNs   = 0u;
      uint64_t  presentToSubmitNs = 0u;
    };

    // ---- state (single immediate context, every entry point runs under
    // the context lock or in D3D11SwapChain::Present on the app thread) ----

    std::string                 g_dir;
    // heap-allocated and never freed on purpose: its queries hold
    // Rc<DxvkDevice>, and dropping the last device ref from a static
    // destructor (dll unload, under the loader lock) joins the device's
    // threads and hangs process exit. Seen in the selftest.
    std::array<FrameSlot, RingSize>& g_ring = *new std::array<FrameSlot, RingSize>();
    uint64_t                    g_presentIndex = 0u;
    uint32_t                    g_recording    = ~0u;  // ring index of the open frame
    uint64_t                    g_nextSlot     = 0u;
    uint64_t                    g_oldestPending = 0u;  // ring sequence number
    double                      g_periodNs     = 1.0;
    bool                        g_periodKnown  = false;
    bool                        g_skipFrame    = false;  // no free slot: this frame goes unrecorded

    // current pass tracking
    uint32_t                    g_curKind = KindNone;
    uint64_t                    g_omGen   = ~0ull;
    std::array<const void*, 9>  g_curKey  = { };
    std::array<const void*, 9>  g_omKey   = { };

    std::vector<ResolvedFrame>  g_window;
    uint32_t                    g_windowIndex = 0u;
    uint32_t                    g_windowPresents = 0u;  // presents since the last window

    std::deque<BlessedGapFrame> g_gapFrames;
    std::deque<GapSpan>         g_gapSpans;
    GapAcc                      g_gapAcc;
    // GpuFlushType 0..3: explicit, sync, strong hint, weak hint
    std::array<uint32_t, 4>     g_flushes = { };
    uint32_t                    g_dropped     = 0u;
    std::ofstream               g_file;
    bool                        g_fileFailed  = false;

    FrameSlot& SlotAt(uint64_t seq) {
      return g_ring[seq % RingSize];
    }

    void EnsurePeriod(DxvkDevice* device) {
      if (!g_periodKnown) {
        g_periodNs = double(device->properties().core.properties.limits.timestampPeriod);
        g_periodKnown = true;
      }
    }

    BlessedGpuPassMark AllocMark(DxvkDevice* device, FrameSlot& slot) {
      if (slot.used == slot.queries.size())
        slot.queries.push_back(device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0u, 0u));

      BlessedGpuPassMark mark;
      mark.query   = slot.queries[slot.used++];
      mark.written = &slot.written;
      return mark;
    }

    // oldest pending frame is dropped when the ring has no free slot left
    void DropOldestPending() {
      while (g_oldestPending < g_nextSlot) {
        FrameSlot& slot = SlotAt(g_oldestPending++);

        if (slot.state == SlotState::Pending) {
          slot.state = SlotState::Free;
          g_dropped += 1u;
          return;
        }
      }
    }

    FrameSlot* BeginFrame() {
      FrameSlot* slot = &SlotAt(g_nextSlot);

      if (slot->state == SlotState::Pending)
        DropOldestPending();

      // a dropped slot can still have marks in flight on the cs thread;
      // reusing it then would race DxvkQuery::begin with a later getData,
      // so the frame goes unrecorded instead
      if (slot->state != SlotState::Free
       || slot->written.load(std::memory_order_acquire) != slot->used) {
        g_skipFrame = true;
        return nullptr;
      }

      slot->state   = SlotState::Recording;
      slot->present = g_presentIndex;
      slot->used    = 0u;
      slot->written.store(0u, std::memory_order_relaxed);
      slot->passes.clear();
      slot->ended   = false;
      slot->blit    = false;

      g_recording = uint32_t(g_nextSlot % RingSize);
      g_nextSlot += 1u;
      return slot;
    }

    FrameSlot* CurrentFrame() {
      return g_recording != ~0u ? &g_ring[g_recording] : nullptr;
    }

    TargetInfo TargetFromView(const D3D11_VK_VIEW_INFO& vi, DXGI_FORMAT fmt, uint32_t slot) {
      TargetInfo info;
      info.slot = slot;
      info.fmt  = uint32_t(fmt);

      if (vi.Dimension != D3D11_RESOURCE_DIMENSION_BUFFER) {
        if (D3D11CommonTexture* tex = GetCommonTexture(vi.pResource)) {
          VkExtent3D ext = tex->MipLevelExtent(vi.Image.MinLevel);
          info.w = ext.width;
          info.h = ext.height;
        }
      }

      return info;
    }

    void RefreshOmKey(const D3D11ContextState& state) {
      if (g_omGen == state.om.blessedOmGeneration)
        return;

      g_omGen = state.om.blessedOmGeneration;

      for (uint32_t i = 0u; i < 8u; i++) {
        auto* rtv = state.om.rtvs[i].ptr();
        g_omKey[i] = rtv ? static_cast<const void*>(rtv->GetViewInfo().pResource) : nullptr;
      }

      auto* dsv = state.om.dsv.ptr();
      g_omKey[8] = dsv ? static_cast<const void*>(dsv->GetViewInfo().pResource) : nullptr;
    }

    void FillTargets(const D3D11ContextState& state, PassInfo& pass) {
      for (uint32_t i = 0u; i < 8u; i++) {
        auto* rtv = state.om.rtvs[i].ptr();

        if (!rtv)
          continue;

        D3D11_RENDER_TARGET_VIEW_DESC desc = { };
        rtv->GetDesc(&desc);
        pass.rtv[pass.rtvCount++] = TargetFromView(rtv->GetViewInfo(), desc.Format, i);
      }

      if (auto* dsv = state.om.dsv.ptr()) {
        D3D11_DEPTH_STENCIL_VIEW_DESC desc = { };
        dsv->GetDesc(&desc);
        pass.hasDsv = true;
        pass.dsv    = TargetFromView(dsv->GetViewInfo(), desc.Format, 0u);
      }
    }

    template<typename Shader>
    void NoteHash(PassInfo& pass, Shader* shader) {
      if (!shader || pass.hashCount >= MaxHashes)
        return;

      const void* key = shader->GetCommonShader();

      for (uint32_t i = 0u; i < pass.hashCount; i++) {
        if (pass.hashKeys[i] == key)
          return;
      }

      pass.hashKeys[pass.hashCount] = key;
      pass.hashes[pass.hashCount]   = shader->GetCommonShader()->GetName();
      pass.hashCount += 1u;
    }

    std::string Signature(const std::vector<PassInfo>& passes) {
      std::string sig;
      sig.reserve(passes.size() * 16u);

      for (const auto& p : passes) {
        sig += char('0' + p.kind);

        for (uint32_t i = 0u; i < p.rtvCount; i++)
          sig += str::format("r", p.rtv[i].slot, ":", p.rtv[i].fmt, ":", p.rtv[i].w, "x", p.rtv[i].h);

        if (p.hasDsv)
          sig += str::format("d", p.dsv.fmt, ":", p.dsv.w, "x", p.dsv.h);

        sig += ';';
      }

      return sig;
    }

    bool TryResolve(FrameSlot& slot, ResolvedFrame& out) {
      if (slot.written.load(std::memory_order_acquire) != slot.used)
        return false;

      std::vector<uint64_t> ts(slot.used);

      for (uint32_t i = 0u; i < slot.used; i++) {
        DxvkQueryData data = { };

        if (slot.queries[i]->getData(data) != DxvkGpuQueryStatus::Available)
          return false;

        ts[i] = data.timestamp.time;
      }

      auto toMs = [] (uint64_t a, uint64_t b) {
        return b >= a ? double(b - a) * g_periodNs / 1.0e6 : 0.0;
      };

      // marks: one per pass start, the frame end, then (optionally) the blit
      uint32_t passCount = uint32_t(slot.passes.size());

      out.passes = slot.passes;
      out.ms.resize(passCount);

      for (uint32_t i = 0u; i < passCount; i++)
        out.ms[i] = toMs(ts[i], ts[i + 1u]);

      out.frameMs = toMs(ts[0], ts[passCount]);
      out.tsFirst = ts[0];
      out.tsEnd   = ts[passCount];
      out.tsBlit  = ts[passCount];

      if (slot.blit && slot.used == passCount + 2u) {
        PassInfo present;
        present.kind = KindPresent;
        out.passes.push_back(present);
        out.ms.push_back(toMs(ts[passCount], ts[passCount + 1u]));
        out.tsBlit = std::max(ts[passCount + 1u], ts[passCount]);
      }

      out.signature = Signature(out.passes);
      return true;
    }

    bool OpenFile() {
      if (g_file.is_open())
        return true;

      if (g_fileFailed)
        return false;

      std::error_code ec;
      std::filesystem::create_directories(str::topath(g_dir.c_str()), ec);

      std::string path = g_dir + "/gpu-passes.jsonl";
      g_file.open(str::topath(path.c_str()), std::ios_base::app);

      if (!g_file.is_open()) {
        Logger::err(str::format("BlessedGpuPasses: cannot open ", path));
        g_fileFailed = true;
        return false;
      }

      Logger::info(str::format("BlessedGpuPasses: writing ", path));
      return true;
    }

    std::string Fixed(double v, int decimals) {
      char buf[48];
      std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
      return buf;
    }

    void WriteTarget(std::string& s, const TargetInfo& t, bool withSlot) {
      s += '{';
      if (withSlot)
        s += str::format("\"slot\":", t.slot, ",");
      s += str::format("\"fmt\":", t.fmt, ",\"w\":", t.w, ",\"h\":", t.h, "}");
    }

    uint64_t Overlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) {
      uint64_t lo = std::max(a0, b0);
      uint64_t hi = std::min(a1, b1);
      return hi > lo ? hi - lo : 0u;
    }

    // splits one gap frame along its pass span: passes [first, end], the
    // present blit [end, blit], everything else is dxvk's own work or idle
    void AccumulateGaps(const BlessedGapFrame& f, const GapSpan& span) {
      GapAcc& acc = g_gapAcc;

      uint64_t start  = f.gfx.front().begin;
      uint64_t w0     = f.prevEnd && f.prevEnd <= start ? f.prevEnd : start;
      uint64_t cursor = w0;

      for (const auto& seg : f.gfx) {
        // idle before this buffer, filed under what its chunk waited on
        if (seg.begin > cursor) {
          uint64_t len = seg.begin - cursor;
          acc.idle += len;
          acc.idleBy[std::min(uint32_t(seg.wait), WaitKinds - 1u)] += len;
          acc.idleInPasses += Overlap(cursor, seg.begin, span.first, span.end);

          if (&seg == &f.gfx.front()) {
            acc.idleStart += len;

            if (f.startState == 0u)
              acc.idleStartLate += len;
            else if (f.startState == 1u)
              acc.idleStartQueued += len;
          }
        }

        uint64_t b = std::max(seg.begin, cursor);
        uint64_t e = std::max(seg.end, b);

        uint64_t inPasses  = Overlap(b, e, span.first, span.end);
        uint64_t inPresent = Overlap(b, e, span.end, span.blit);
        uint64_t outside   = (e - b) - inPasses - inPresent;

        acc.passes  += inPasses;
        acc.present += inPresent;

        switch (DxvkCmdBuffer(seg.type)) {
          case DxvkCmdBuffer::InitBarriers:
          case DxvkCmdBuffer::InitBuffer:
            acc.init += outside;
            acc.initInPasses += inPasses;
            break;

          case DxvkCmdBuffer::SdmaBarriers:
          case DxvkCmdBuffer::SdmaBuffer:
            acc.sdma += outside;
            break;

          default: {
            uint64_t pre = Overlap(b, e, 0u, span.first);
            acc.execPre  += pre;
            acc.execPost += outside - std::min(outside, pre);
          }
        }

        cursor = std::max(cursor, e);
      }

      acc.frames       += 1u;
      acc.frame        += cursor - w0;
      acc.passesSpan   += span.end - span.first;
      acc.sdmaQueue    += f.sdmaTicks;
      acc.cmdLists     += f.cmdLists;
      acc.submits      += f.graphicsSubmits;
      acc.sdmaChunks   += f.sdmaChunks;
      acc.barriers     += f.barriers;
      acc.renderPasses += f.renderPasses;

      acc.framesLate   += f.startState == 0u ? 1u : 0u;
      acc.framesQueued += f.startState == 1u ? 1u : 0u;

      if (f.presentCallNs) {
        acc.presentFrames     += 1u;
        acc.presentCallNs     += f.presentCallNs;
        acc.presentToSubmitNs += f.presentToSubmitNs;
      }
    }

    // pairs resolved pass spans with gap frames by timestamp: the gap frame
    // whose graphics work contains the span's first mark
    void MatchGaps() {
      BlessedGapFrame gf;

      while (BlessedGpuGaps::TakeFrame(gf)) {
        if (g_gapFrames.size() >= 32u)
          g_gapFrames.pop_front();

        g_gapFrames.push_back(std::move(gf));
      }

      while (!g_gapSpans.empty()) {
        GapSpan& span = g_gapSpans.front();

        while (!g_gapFrames.empty() && g_gapFrames.front().gfx.back().end < span.first)
          g_gapFrames.pop_front();

        if (g_gapFrames.empty()) {
          // its gap frame may still be on the gpu; give up after a while
          if (++span.age > 16u) {
            g_gapAcc.unmatched += 1u;
            g_gapSpans.pop_front();
            continue;
          }

          return;
        }

        const BlessedGapFrame& f = g_gapFrames.front();

        if (span.first < f.gfx.front().begin || span.blit > f.gfx.back().end) {
          // its gap frame was dropped, or the span crosses a present
          g_gapAcc.unmatched += 1u;
          g_gapSpans.pop_front();
          continue;
        }

        AccumulateGaps(f, span);
        g_gapFrames.pop_front();
        g_gapSpans.pop_front();
      }
    }

    // appPasses: the window's pass count (its mode layout), so menu and
    // loading windows can be told from game frames; timerMs: the pass
    // timer's own gpu_ms_all for the same window, the self-check
    void WriteGaps(std::string& out, size_t appPasses, double timerMs) {
      const GapAcc& a = g_gapAcc;
      double presents = double(std::max(g_windowPresents, 1u));

      if (!a.frames) {
        out += str::format("{\"type\":\"gaps\",\"source\":\"dxvk\",\"window\":", g_windowIndex,
          ",\"passes\":", appPasses,
          ",\"frames\":0,\"unmatched\":", a.unmatched, "}\n");
        return;
      }

      // self-check: busy time inside the pass span (gap marks) against the
      // pass timer's span (its own marks) for the same window
      double passesMs = double(a.passes) * g_periodNs / 1.0e6 / double(a.frames);
      double checkRatio = timerMs > 0.0 ? passesMs / timerMs : 0.0;
      bool checkOk = checkRatio >= 0.9 && checkRatio <= 1.1;

      if (!checkOk) {
        Logger::warn(str::format("BlessedGpuPasses: window ", g_windowIndex,
          ": gaps passes_ms ", Fixed(passesMs, 4), " vs pass timer gpu_ms_all ", Fixed(timerMs, 4)));
      }

      auto nsMs = [] (uint64_t ns, uint32_t count) {
        return Fixed(count ? double(ns) / 1.0e6 / double(count) : 0.0, 4);
      };

      double n = double(a.frames);
      auto ms = [n] (uint64_t ticks) {
        return Fixed(double(ticks) * g_periodNs / 1.0e6 / n, 4);
      };
      auto per = [n] (uint64_t count) {
        return Fixed(double(count) / n, 2);
      };

      uint64_t internal = a.init + a.sdma + a.execPre + a.execPost;

      out += str::format("{\"type\":\"gaps\",\"source\":\"dxvk\",\"window\":", g_windowIndex,
        ",\"passes\":", appPasses,
        ",\"frames\":", a.frames,
        ",\"unmatched\":", a.unmatched,
        ",\"check\":{\"pass_timer_gpu_ms\":", Fixed(timerMs, 4),
          ",\"passes_ms\":", Fixed(passesMs, 4),
          ",\"ratio\":", Fixed(checkRatio, 3),
          ",\"ok\":", checkOk ? "true" : "false", "}",
        ",\"frame_ms\":", ms(a.frame),
        ",\"busy_ms\":", ms(a.frame - a.idle),
        ",\"passes_ms\":", ms(a.passes),
        ",\"passes_span_ms\":", ms(a.passesSpan),
        ",\"present_ms\":", ms(a.present),
        ",\"internal_ms\":", ms(internal),
        ",\"internal_by\":{\"init\":", ms(a.init),
          ",\"sdma\":", ms(a.sdma),
          ",\"exec_pre\":", ms(a.execPre),
          ",\"exec_post\":", ms(a.execPost), "}",
        ",\"idle_ms\":", ms(a.idle),
        ",\"idle_by\":{");

      for (uint32_t i = 0u; i < WaitKinds; i++)
        out += str::format(i ? "," : "", "\"", WaitName(i), "\":", ms(a.idleBy[i]));

      out += str::format("}",
        ",\"idle_in_passes_ms\":", ms(a.idleInPasses),
        ",\"idle_frame_start_ms\":", ms(a.idleStart),
        ",\"idle_frame_start_by\":{\"late_submit\":", ms(a.idleStartLate),
          ",\"queued\":", ms(a.idleStartQueued), "}",
        ",\"frames_started_late\":", a.framesLate,
        ",\"frames_started_queued\":", a.framesQueued,
        ",\"present_call_ms\":", nsMs(a.presentCallNs, a.presentFrames),
        ",\"present_to_submit_ms\":", nsMs(a.presentToSubmitNs, a.presentFrames),
        ",\"init_in_passes_ms\":", ms(a.initInPasses),
        ",\"sdma_queue_ms\":", ms(a.sdmaQueue),
        ",\"submits_per_frame\":", per(a.submits),
        ",\"cmdlists_per_frame\":", per(a.cmdLists),
        ",\"sdma_chunks_per_frame\":", per(a.sdmaChunks),
        ",\"barriers_per_frame\":", per(a.barriers),
        ",\"render_passes_per_frame\":", per(a.renderPasses),
        ",\"flushes_per_frame\":{\"explicit\":", Fixed(double(g_flushes[0]) / presents, 2),
          ",\"sync\":", Fixed(double(g_flushes[1]) / presents, 2),
          ",\"strong\":", Fixed(double(g_flushes[2]) / presents, 2),
          ",\"weak\":", Fixed(double(g_flushes[3]) / presents, 2), "}}\n");
    }

    void WriteWindow() {
      if (g_window.empty() || !OpenFile())
        return;

      // the window's mode: the most common pass layout
      std::map<std::string, uint32_t> counts;
      for (const auto& f : g_window)
        counts[f.signature] += 1u;

      const std::string* modeSig = nullptr;
      uint32_t modeCount = 0u;

      for (const auto& e : counts) {
        if (e.second > modeCount) {
          modeSig   = &e.first;
          modeCount = e.second;
        }
      }

      const ResolvedFrame* first = nullptr;
      double frameSum = 0.0;
      double frameSumAll = 0.0;

      for (const auto& f : g_window) {
        frameSumAll += f.frameMs;

        if (f.signature == *modeSig) {
          if (!first)
            first = &f;
          frameSum += f.frameMs;
        }
      }

      size_t passCount = first->passes.size();
      const char* mode = blessed_gpu_passes_detail::g_mode == ModeFrame ? "frame" : "passes";

      std::string out;
      out.reserve(256u + passCount * 256u);

      // present-pass (dxvk's swapchain blit) is not a boundary of the app's
      // frame: timestamps count the app's pass starts plus the frame end
      size_t appPasses = passCount;
      if (appPasses && first->passes.back().kind == KindPresent)
        appPasses -= 1u;

      out += str::format("{\"type\":\"window\",\"source\":\"dxvk\",\"mode\":\"", mode, "\"",
        ",\"window\":", g_windowIndex,
        ",\"present\":", g_presentIndex,
        ",\"frames\":", g_window.size(),
        ",\"matched\":", modeCount,
        ",\"passes\":", passCount,
        ",\"timestamps\":", appPasses + 1u,
        ",\"gpu_ms\":", Fixed(frameSum / double(modeCount), 4),
        ",\"gpu_ms_all\":", Fixed(frameSumAll / double(g_window.size()), 4),
        ",\"dropped\":", g_dropped,
        ",\"disjoint\":0}\n");

      for (size_t i = 0u; i < passCount; i++) {
        double sum = 0.0, lo = 0.0, hi = 0.0;
        double draws = 0.0, dispatches = 0.0, xfer = 0.0;
        bool any = false;

        for (const auto& f : g_window) {
          if (f.signature != *modeSig)
            continue;

          double ms = f.ms[i];
          sum += ms;
          lo = any ? std::min(lo, ms) : ms;
          hi = any ? std::max(hi, ms) : ms;
          any = true;

          draws      += double(f.passes[i].draws);
          dispatches += double(f.passes[i].dispatches);
          xfer       += double(f.passes[i].xfer);
        }

        double n = double(modeCount);
        const PassInfo& p = first->passes[i];

        out += str::format("{\"type\":\"pass\",\"source\":\"dxvk\",\"mode\":\"", mode, "\"",
          ",\"window\":", g_windowIndex,
          ",\"index\":", i,
          ",\"kind\":\"", KindName(p.kind), "\",\"rtv\":[");

        for (uint32_t r = 0u; r < p.rtvCount; r++) {
          if (r)
            out += ',';
          WriteTarget(out, p.rtv[r], true);
        }

        out += "],\"dsv\":";

        if (p.hasDsv)
          WriteTarget(out, p.dsv, false);
        else
          out += "null";

        out += str::format(",\"draws\":", Fixed(draws / n, 1),
          ",\"dispatches\":", Fixed(dispatches / n, 1),
          ",\"xfer\":", Fixed(xfer / n, 1));

        // ps hashes for draw passes, cs hashes for compute passes
        std::string hashList;
        for (uint32_t h = 0u; h < p.hashCount; h++) {
          if (h)
            hashList += ',';
          hashList += "\"" + p.hashes[h] + "\"";
        }

        bool isCompute = p.kind == KindCompute;
        out += str::format(",\"ps\":[", isCompute ? "" : hashList, "]",
          ",\"cs\":[", isCompute ? hashList : "", "]",
          ",\"gpu_ms\":", Fixed(sum / n, 4),
          ",\"gpu_ms_min\":", Fixed(lo, 4),
          ",\"gpu_ms_max\":", Fixed(hi, 4), "}\n");
      }

      if (BlessedGpuGaps::IsEnabled())
        WriteGaps(out, passCount, frameSumAll / double(g_window.size()));

      g_file << out;
      g_file.flush();

      g_windowIndex += 1u;
      g_dropped = 0u;

      g_gapAcc = GapAcc();
      g_flushes = { };
      g_windowPresents = 0u;
    }

    uint32_t InitMode() {
      std::string mode = env::getEnvVar("BLESSED_GPU_PASSES");

      uint32_t result = ModeOff;

      if (mode == "1")
        result = ModePasses;
      else if (mode == "frame")
        result = ModeFrame;

      if (result == ModeOff)
        return ModeOff;

      // own dir first: BLESSED_PROBE_DIR also switches on the cpu probe
      g_dir = env::getEnvVar("BLESSED_GPU_PASSES_DIR");

      if (g_dir.empty())
        g_dir = env::getEnvVar("BLESSED_PROBE_DIR");

      if (g_dir.empty()) {
        std::string tmp = env::getEnvVar("TEMP");
        g_dir = (tmp.empty() ? std::string(".") : tmp) + "/blessed-gpu-passes";
      }

      // no Logger here: this runs during static init, possibly before the
      // logger's own static instance exists (dll load failed with 1114).
      // OpenFile logs the path on first use instead.
      return result;
    }

  }


  namespace blessed_gpu_passes_detail {
    // must stay after g_dir: InitMode writes it during static init
    extern const uint32_t g_mode = InitMode();
  }


  BlessedGpuPassMark BlessedGpuPasses::OnCall(
          DxvkDevice*           device,
    const D3D11ContextState&    state,
          BlessedGpuPassKind    kind) {
    uint32_t newKind = kind == BlessedGpuPassKind::Draw    ? KindDraw
                     : kind == BlessedGpuPassKind::Compute ? KindCompute
                     :                                       KindXfer;

    bool newPass = g_curKind != newKind;

    if (newKind == KindDraw) {
      RefreshOmKey(state);
      newPass |= g_omKey != g_curKey;
    }

    FrameSlot* frame = CurrentFrame();
    BlessedGpuPassMark mark;

    if (newPass) {
      g_curKind = newKind;
      g_curKey  = newKind == KindDraw ? g_omKey : std::array<const void*, 9>{ };

      bool firstOfFrame = frame == nullptr;

      if (firstOfFrame) {
        if (g_skipFrame)
          return mark;

        EnsurePeriod(device);
        frame = BeginFrame();

        if (!frame)
          return mark;
      }

      if (blessed_gpu_passes_detail::g_mode == ModePasses || firstOfFrame) {
        PassInfo& pass = frame->passes.emplace_back();
        pass.kind = blessed_gpu_passes_detail::g_mode == ModeFrame ? KindFrame : newKind;

        if (newKind == KindDraw && pass.kind == KindDraw)
          FillTargets(state, pass);

        mark = AllocMark(device, *frame);
      }
    }

    if (!frame || frame->passes.empty())
      return mark;

    PassInfo& pass = frame->passes.back();

    switch (newKind) {
      case KindDraw:
        pass.draws += 1u;
        if (pass.kind == KindDraw)
          NoteHash(pass, state.ps.ptr());
        break;

      case KindCompute:
        pass.dispatches += 1u;
        if (pass.kind == KindCompute)
          NoteHash(pass, state.cs.ptr());
        break;

      default:
        pass.xfer += 1u;
    }

    return mark;
  }


  BlessedGpuPassMark BlessedGpuPasses::OnFrameEnd(DxvkDevice* device) {
    FrameSlot* frame = CurrentFrame();

    if (!frame || frame->passes.empty() || frame->ended)
      return BlessedGpuPassMark();

    frame->ended = true;
    return AllocMark(device, *frame);
  }


  BlessedGpuPassMark BlessedGpuPasses::OnPresentBlit(DxvkDevice* device) {
    FrameSlot* frame = CurrentFrame();

    if (!frame || !frame->ended || frame->blit)
      return BlessedGpuPassMark();

    frame->blit = true;
    return AllocMark(device, *frame);
  }


  void BlessedGpuPasses::OnFlush(uint32_t flushType) {
    if (flushType < g_flushes.size())
      g_flushes[flushType] += 1u;
  }


  void BlessedGpuPasses::OnPresent() {
    g_presentIndex += 1u;
    g_windowPresents += 1u;
    g_curKind   = KindNone;
    g_skipFrame = false;

    // close the frame only once its end mark went out; a failed present
    // before PresentImage leaves it open and it continues into the next one
    if (FrameSlot* frame = CurrentFrame()) {
      if (frame->ended) {
        frame->state = SlotState::Pending;
        g_recording  = ~0u;
      }
    }

    // resolve oldest first, stop at the first frame that is not ready yet
    while (g_oldestPending < g_nextSlot) {
      FrameSlot& slot = SlotAt(g_oldestPending);

      if (slot.state == SlotState::Free) {
        g_oldestPending += 1u;
        continue;
      }

      if (slot.state != SlotState::Pending || g_presentIndex - slot.present < MinAge)
        break;

      ResolvedFrame frame;

      if (!TryResolve(slot, frame))
        break;

      slot.state = SlotState::Free;
      g_oldestPending += 1u;

      if (BlessedGpuGaps::IsEnabled()) {
        GapSpan span;
        span.first = frame.tsFirst;
        span.end   = frame.tsEnd;
        span.blit  = frame.tsBlit;

        if (g_gapSpans.size() >= 32u)
          g_gapSpans.pop_front();

        g_gapSpans.push_back(span);
        MatchGaps();
      }

      g_window.push_back(std::move(frame));

      if (g_window.size() >= WindowFrames) {
        WriteWindow();
        g_window.clear();
      }
    }
  }

}
