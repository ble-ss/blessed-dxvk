// blessed: render-thread + cs-thread timing probe implementation
#include "util_blessed_probe.h"

// blessed: -Dblessed_probe=false -- the header already reduced every scope
// and Call/MapType/BindKind to trivial stand-ins with no state; this whole
// file is dead weight in that build (nothing left to sample, nowhere to
// write), so it compiles to an empty translation unit instead.
#if BLESSED_PROBE_ENABLED

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <vector>

#include "util_env.h"
#include "util_string.h"

#include "thread.h" // blessed: threaded-fe, census thread ids

// Standard D3D11 bind flag bits (D3D11_BIND_VERTEX_BUFFER = 0x1,
// D3D11_BIND_INDEX_BUFFER = 0x2, D3D11_BIND_CONSTANT_BUFFER = 0x4).
// Spelled out numerically so this file has no dependency on d3d11 headers.
namespace {
  constexpr uint32_t kBindVertexBuffer   = 0x1u;
  constexpr uint32_t kBindIndexBuffer    = 0x2u;
  constexpr uint32_t kBindConstantBuffer = 0x4u;

  constexpr uint32_t kWindowFrames = 120u;

  constexpr uint32_t kFeThreads = 4u; // blessed: threaded-fe
}

namespace dxvk::blessed {

  const char* const g_callNames[uint32_t(Call::Count)] = {
    "Map",
    "Unmap",
    "UpdateSubresource",
    "UpdateSubresource1",
    "SetConstantBuffers",
    "SetShaderResources",
    "SetSamplers",
    "SetShader",
    "IASetVertexBuffers",
    "IASetIndexBuffer",
    "IASetInputLayout",
    "IASetPrimitiveTopology",
    "OMSetRenderTargets",
    "OMSetBlendState",
    "OMSetDepthStencilState",
    "RSSetState",
    "RSSetViewports",
    "RSSetScissorRects",
    "Draw",
    "DrawIndexed",
    "DrawInstanced",
    "DrawIndexedInstanced",
    "Dispatch",
    "CopyResource",
    "CopySubresourceRegion",
    "ClearRenderTargetView",
    "ClearDepthStencilView",
    "DrawIndexed.cascade_check",
    "DrawIndexed.scene_capture",
    "DrawIndexed.gi",
    "DrawIndexed.core",
    "DrawIndexed.post_hook",
    // blessed: hook-cpu sub-splits, see the enum's comment
    "DrawIndexed.scene_capture_resolve",
    "DrawIndexed.gi_patch",
    // blessed: hook-cpu-2 sub-splits, see the enum's comment
    "DrawIndexed.scene_static",
    "DrawIndexed.scene_static_emit",
    "DrawIndexed.scene_skinned",
    "DrawIndexed.scene_skinned_stage",
    "DrawIndexed.gi_resolve",
    "DrawIndexed.gi_cbread",
    "DrawIndexed.gi_sample",
    "DrawIndexed.gi_write",
    // blessed: hook-cpu-2 round two, see the enum's comment
    "DrawIndexed.scene_skinned_acquire",
    "DrawIndexed.scene_skinned_grow",
    "DrawIndexed.scene_skinned_slices",
    "DrawIndexed.scene_skinned_push",
    "DrawIndexed.gi_sample_cell",
    "DrawIndexed.gi_emit", // blessed: gi-cs
  };

  static const char* mapTypeName(MapType t) {
    switch (t) {
      case MapType::Discard:     return "discard";
      case MapType::NoOverwrite: return "nooverwrite";
      case MapType::Ring:        return "ring"; // blessed: cb-ring
      default:                   return "other";
    }
  }

  static const char* bindKindName(BindKind b) {
    switch (b) {
      case BindKind::Constant: return "cb";
      case BindKind::Vertex:   return "vb";
      case BindKind::Index:    return "ib";
      default:                 return "other";
    }
  }

  BindKind classifyBindFlags(uint32_t bindFlags) {
    if (bindFlags & kBindConstantBuffer)
      return BindKind::Constant;
    if (bindFlags & kBindVertexBuffer)
      return BindKind::Vertex;
    if (bindFlags & kBindIndexBuffer)
      return BindKind::Index;
    return BindKind::Other;
  }


  struct Sample {
    uint64_t n    = 0;
    int64_t  ticks= 0; // raw stamp() ticks, converted at flush
  };


  // All state below is touched only from the single thread that owns the
  // D3D11 immediate context (Map/Draw/... calls, and Present, are only ever
  // valid from that thread), so none of this needs synchronization.
  struct ProbeState {
    bool                    initDone   = false;
    std::ofstream           file;
    dxvk::high_resolution_clock::time_point processStart;
    dxvk::high_resolution_clock::time_point windowStart;
    dxvk::high_resolution_clock::time_point lastPresent;
    bool                    havePresent = false;

    std::array<Sample, uint32_t(Call::Count)> calls;
    std::array<std::array<Sample, uint32_t(BindKind::Count)>, uint32_t(MapType::Count)> map;
    Sample                  discardSlice;
    std::array<Sample, uint32_t(CbRingEvent::Count)> cbRing; // blessed: cb-ring
    Sample                  stallEmitBlock;

    std::vector<double>     frameTimesMs;   // untimed frames only when every > 1
    uint32_t                every        = 1;
    uint32_t                frameIndex   = 0;
    uint32_t                activeFrames = 0;
    uint32_t                windowFrames = 0;
    int64_t                 windowStamp  = 0;
    double                  pairTicks    = 0.0; // cost of one empty stamp() pair

    // blessed: threaded-fe -- census per facade entry and calling thread
    // (slot kFeThreads - 1 collects every thread past the first ones)
    std::array<std::array<Sample, kFeThreads>, uint32_t(FeCall::Count)> feCalls;
    std::array<uint32_t, kFeThreads> feThreadIds = { };
    uint32_t                feThreadCount = 0;
    std::array<Sample, uint32_t(FeCall::Count)> feDirect; // census entries that drained
    std::array<Sample, uint32_t(FeDrain::Count)> feDrains;
    Sample                  feRingFull;
    Sample                  feReplay;
    Sample                  fePresentWait; // blessed: threaded-fe-2
    uint64_t                feOccupancySum = 0;
    uint64_t                feOccupancyMax = 0;
    uint64_t                feOccupancyN   = 0;
    FeSnapshot              feSnap;
    FeSnapshot              feSnapLast;

    uint64_t                lastCsIdleTicks     = 0;
    uint64_t                lastCsChunkCount    = 0;
    uint64_t                lastCsChunkCmdCount = 0;
    uint64_t                lastCsSyncCount     = 0;
    uint64_t                lastCsSyncTicks     = 0;
  };

  static ProbeState g_state;


  const bool g_enabled = !env::getEnvVar("BLESSED_PROBE_DIR").empty();
  bool       g_active  = g_enabled;
  const bool g_nested  = env::getEnvVar("BLESSED_PROBE_NEST") != "0";
  int64_t    g_feNestedTicks = 0; // blessed: threaded-fe, see FeScope
  uint64_t   g_feDrainCount  = 0; // blessed: threaded-fe, see FeScope


  static void ensureInit() {
    if (g_state.initDone)
      return;

    g_state.initDone     = true;
    g_state.processStart = dxvk::high_resolution_clock::now();
    g_state.windowStart  = g_state.processStart;
    g_state.windowStamp  = stamp();

    std::string every = env::getEnvVar("BLESSED_PROBE_EVERY");
    if (!every.empty())
      g_state.every = std::max(1, std::atoi(every.c_str()));

    // what an empty scope measures: subtracted per sample at flush.
    // blessed: perf-halfrate -- the median, not the minimum: back-to-back
    // rdtsc pairs occasionally read 1 tick apart, and the minimum flipped
    // between 10 ns and 0.29 ns from one process to the next (bis-rest vs
    // bis-rebar), which moved every bucket by n * 9.7 ns.
    std::vector<int64_t> pairs(20000u);
    for (auto& d : pairs) {
      int64_t t0 = stamp();
      int64_t t1 = stamp();
      d = t1 - t0;
    }
    std::nth_element(pairs.begin(), pairs.begin() + pairs.size() / 2, pairs.end());
    g_state.pairTicks = double(pairs[pairs.size() / 2]);

    std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
    std::string path = dir + env::PlatformDirSlash + "probe-frames.jsonl";

    g_state.file.open(str::topath(path.c_str()), std::ios::out | std::ios::trunc);
  }


  void addCallSample(Call id, int64_t ticks) {
    ensureInit();

    auto& s = g_state.calls[uint32_t(id)];
    s.n     += 1;
    s.ticks += ticks;
  }


  void addMapSample(MapType type, BindKind bind, int64_t ticks) {
    ensureInit();

    auto& s = g_state.map[uint32_t(type)][uint32_t(bind)];
    s.n     += 1;
    s.ticks += ticks;
  }


  void addDiscardSliceSample(int64_t ticks) {
    ensureInit();

    g_state.discardSlice.n     += 1;
    g_state.discardSlice.ticks += ticks;
  }


  void addCbRingSample(CbRingEvent event, int64_t ticks) {
    ensureInit();

    auto& s = g_state.cbRing[uint32_t(event)];
    s.n     += 1;
    s.ticks += ticks;
  }


  void addEmitBlockSample(int64_t ticks) {
    ensureInit();

    g_state.stallEmitBlock.n     += 1;
    g_state.stallEmitBlock.ticks += ticks;
  }


  // blessed: threaded-fe -- census and front end buckets

  static const char* const g_feCallNames[uint32_t(FeCall::Count)] = {
#define BLESSED_FE_NAME(name) #name,
    BLESSED_FE_CALLS(BLESSED_FE_NAME)
  };

  static const char* const g_feDrainNames[uint32_t(FeDrain::Count)] = {
    BLESSED_FE_DRAINS(BLESSED_FE_NAME)
#undef BLESSED_FE_NAME
  };


  void feInit() {
    if (enabled())
      ensureInit();
  }


  static uint32_t feThreadSlot() {
    // Census entries from other threads (device-side calls) may race a
    // slot in; the count is clamped so a race can only miscount
    uint32_t tid = uint32_t(dxvk::this_thread::get_id());
    uint32_t count = std::min(g_state.feThreadCount, kFeThreads);

    for (uint32_t i = 0; i < count; i++) {
      if (g_state.feThreadIds[i] == tid)
        return i;
    }

    if (count < kFeThreads) {
      g_state.feThreadIds[count] = tid;
      g_state.feThreadCount = count + 1u;
      return count;
    }

    return kFeThreads - 1u;
  }


  void addFeCallSample(FeCall call, int64_t ticks, bool drained) {
    ensureInit();

    auto& s = g_state.feCalls[uint32_t(call)][feThreadSlot()];
    s.n     += 1;
    s.ticks += ticks;

    if (drained) {
      auto& d = g_state.feDirect[uint32_t(call)];
      d.n     += 1;
      d.ticks += ticks;
    }
  }


  void addFeDrainSample(FeDrain reason, int64_t ticks) {
    ensureInit();

    auto& s = g_state.feDrains[uint32_t(reason)];
    s.n     += 1;
    s.ticks += ticks;

    g_feNestedTicks += ticks;
    g_feDrainCount  += 1u;
  }


  void addFeRingFullSample(int64_t ticks) {
    ensureInit();

    g_state.feRingFull.n     += 1;
    g_state.feRingFull.ticks += ticks;

    g_feNestedTicks += ticks;
  }


  // blessed: threaded-fe-2 -- the game thread waiting at Present for the
  // front end to finish the previous one (the one-frame cap)
  void addFePresentWaitSample(int64_t ticks) {
    ensureInit();

    g_state.fePresentWait.n     += 1;
    g_state.fePresentWait.ticks += ticks;

    g_feNestedTicks += ticks;
  }


  void addFeReplaySample(int64_t ticks) {
    ensureInit();

    g_state.feReplay.n     += 1;
    g_state.feReplay.ticks += ticks;

    g_feNestedTicks += ticks;
  }


  void addFeOccupancy(uint64_t bytes) {
    g_state.feOccupancySum += bytes;
    g_state.feOccupancyMax  = std::max(g_state.feOccupancyMax, bytes);
    g_state.feOccupancyN   += 1;
  }


  void setFeSnapshot(const FeSnapshot& fe) {
    if (!enabled())
      return;

    if (!g_state.feSnap.mode)
      g_state.feSnapLast = fe;

    g_state.feSnap = fe;
  }


  // stamp() ticks per millisecond for the window being flushed
  static double g_ticksPerMs = 1.0;

  // a bucket's time per timed frame, less the probe's own empty-pair cost
  static double msOf(const Sample& s, uint32_t frames) {
    double ticks = double(s.ticks) - double(s.n) * g_state.pairTicks;
    return std::max(ticks, 0.0) / g_ticksPerMs / double(std::max(frames, 1u));
  }

  static double perFrame(uint64_t n, uint32_t frames) {
    return double(n) / double(std::max(frames, 1u));
  }


  static double percentile(std::vector<double>& sorted, double p) {
    if (sorted.empty())
      return 0.0;

    size_t idx = size_t(std::ceil(p * double(sorted.size()))) - 1u;
    idx = std::min(idx, sorted.size() - 1u);
    return sorted[idx];
  }


  static void flushWindow(const CsSnapshot& cs, dxvk::high_resolution_clock::time_point now) {
    if (!g_state.file.is_open())
      return;

    uint32_t frames = uint32_t(g_state.frameTimesMs.size());
    if (frames == 0)
      return;

    std::vector<double> sorted = g_state.frameTimesMs;
    std::sort(sorted.begin(), sorted.end());

    double sum = 0.0;
    for (double v : sorted)
      sum += v;
    double meanFrameMs = sum / double(frames);

    double t = std::chrono::duration<double>(now - g_state.processStart).count();

    double windowElapsedUs = std::chrono::duration<double, std::micro>(now - g_state.windowStart).count();
    int64_t nowStamp = stamp();
    g_ticksPerMs = std::max(1.0, double(nowStamp - g_state.windowStamp) / (windowElapsedUs / 1000.0));

    uint32_t timed = g_state.activeFrames;
    uint32_t all   = std::max(g_state.windowFrames, 1u);
    Sample app;
    uint64_t samples = 0;
    for (uint32_t i = 0; i < uint32_t(Call::SubFirst); i++) {
      app.n += g_state.calls[i].n;
      app.ticks += g_state.calls[i].ticks;
    }
    samples = app.n + g_state.discardSlice.n + g_state.stallEmitBlock.n;
    for (auto& row : g_state.map)
      for (auto& s : row)
        samples += s.n;
    double idleUs = double(cs.idleTicks - g_state.lastCsIdleTicks);
    double busyUs = windowElapsedUs - idleUs;
    if (busyUs < 0.0)
      busyUs = 0.0;

    uint64_t chunkDelta    = cs.chunkCount    - g_state.lastCsChunkCount;
    uint64_t chunkCmdDelta = cs.chunkCmdCount - g_state.lastCsChunkCmdCount;
    uint64_t syncCountDelta= cs.syncCount     - g_state.lastCsSyncCount;
    uint64_t syncTicksDelta= cs.syncTicks     - g_state.lastCsSyncTicks;

    std::string line;
    line.reserve(1024);
    line += str::format("{\"t\":", t,
      ",\"frames\":", frames,
      ",\"frame_ms\":{\"mean\":", meanFrameMs,
      ",\"p50\":", percentile(sorted, 0.50),
      ",\"p95\":", percentile(sorted, 0.95),
      ",\"p99\":", percentile(sorted, 0.99), "}",
      ",\"probe\":{\"every\":", g_state.every, ",\"nested\":", g_nested ? 1 : 0, ",\"timed_frames\":", timed,
      ",\"samples\":", perFrame(samples, timed), ",\"pair_ns\":", g_state.pairTicks / g_ticksPerMs * 1.0e6, "}",
      ",\"app_in_dxvk_ms\":", msOf(app, timed));

    line += ",\"calls\":{";
    for (uint32_t i = 0; i < uint32_t(Call::Count); i++) {
      auto& s = g_state.calls[i];
      if (i) line += ",";
      line += str::format("\"", g_callNames[i], "\":{\"n\":",
        perFrame(s.n, timed), ",\"ms\":", msOf(s, timed), "}");
    }
    line += "}";

    line += ",\"map\":{";
    bool first = true;
    for (uint32_t t2 = 0; t2 < uint32_t(MapType::Count); t2++) {
      for (uint32_t b = 0; b < uint32_t(BindKind::Count); b++) {
        auto& s = g_state.map[t2][b];
        if (!first) line += ",";
        first = false;
        line += str::format("\"", mapTypeName(MapType(t2)), "_", bindKindName(BindKind(b)),
          "\":{\"n\":", perFrame(s.n, timed), ",\"ms\":", msOf(s, timed), "}");
      }
    }
    line += str::format(",\"discard_slice\":{\"n\":", perFrame(g_state.discardSlice.n, timed),
      ",\"ms\":", msOf(g_state.discardSlice, timed), "}");

    // blessed: cb-ring -- slow-path events, nested inside ring_cb
    static const char* const cbRingNames[uint32_t(CbRingEvent::Count)] = { "advance", "new_block", "fallback", "rename_cmd" };
    line += ",\"cb_ring\":{";
    for (uint32_t e = 0; e < uint32_t(CbRingEvent::Count); e++) {
      auto& s = g_state.cbRing[e];
      line += str::format(e ? "," : "", "\"", cbRingNames[e], "\":{\"n\":",
        perFrame(s.n, timed), ",\"ms\":", msOf(s, timed), "}");
    }
    line += "}}";

    // blessed: threaded-fe -- census and front end buckets. census and
    // drain/replay/ring_full are per timed frame like calls; records,
    // bytes, publishes, releases and idle are per frame over all frames.
    // census entries hold their own time only (drains, inline replay
    // and ring-full waits are subtracted, see FeScope). direct_ms is the
    // part of it spent in entries that drained and then ran dxvk's own
    // method. record_ms is the facade's own cost on the recording thread:
    // every entry that did not drain, except Present, the app's GetData
    // polling and the device-side ones.
    { Sample census;
      Sample recording;
      Sample direct;
      bool anyCensus = false;

      for (uint32_t i = 0; i < uint32_t(FeCall::Count); i++) {
        FeCall call = FeCall(i);
        bool recorder = call != FeCall::Present && call != FeCall::GetData
          && call != FeCall::DeviceInternal && call != FeCall::CreateDeferredContext
          && call != FeCall::GetImmediateContext && call != FeCall::QueryInterface;

        for (const auto& s : g_state.feCalls[i]) {
          census.n     += s.n;
          census.ticks += s.ticks;

          if (recorder) {
            recording.n     += s.n;
            recording.ticks += s.ticks;
          }
        }

        const auto& d = g_state.feDirect[i];
        direct.n     += d.n;
        direct.ticks += d.ticks;

        if (recorder) {
          recording.n     -= d.n;
          recording.ticks -= d.ticks;
        }
      }

      anyCensus = census.n != 0;

      if (anyCensus || g_state.feSnap.mode) {
        const FeSnapshot& a = g_state.feSnap;
        const FeSnapshot& b = g_state.feSnapLast;

        Sample drains;
        for (const auto& d : g_state.feDrains) {
          drains.n     += d.n;
          drains.ticks += d.ticks;
        }

        double records  = double(a.records - b.records) / double(all);
        double censusMs = msOf(census, timed);
        double recordMs = msOf(recording, timed);
        double idleMs   = double(a.idleUs - b.idleUs) / 1000.0 / double(all);
        double frameMs  = windowElapsedUs / 1000.0 / double(all);
        const auto& present = g_state.feDrains[uint32_t(FeDrain::Present)];

        line += str::format(",\"fe\":{\"mode\":\"",
          a.mode == 2 ? "threaded" : a.mode == 1 ? "loopback" : "off", "\"",
          ",\"records\":", records,
          ",\"bytes\":", double(a.bytes - b.bytes) / double(all),
          ",\"publishes\":", double(a.publishes - b.publishes) / double(all),
          ",\"releases\":", double(a.releases - b.releases) / double(all),
          ",\"wakes\":", double(a.wakes - b.wakes) / double(all),
          ",\"census_ms\":", censusMs,
          ",\"drain_ms\":", msOf(drains, timed),
          ",\"direct_ms\":", msOf(direct, timed),
          ",\"record_ms\":", recordMs,
          ",\"ns_per_record\":", records > 0.0 ? recordMs * 1.0e6 / records : 0.0,
          ",\"calls\":", perFrame(recording.n, timed),
          ",\"ns_per_call\":", recording.n ? recordMs * 1.0e6 / perFrame(recording.n, timed) : 0.0,
          ",\"folded\":", double(a.folded - b.folded) / double(all),
          ",\"packet_ops\":", double(a.packetOps - b.packetOps) / double(all),
          ",\"replay_inline_ms\":", msOf(g_state.feReplay, timed),
          ",\"busy_ms\":", a.mode == 2 ? std::max(0.0, frameMs - idleMs) : 0.0,
          ",\"idle_ms\":", a.mode == 2 ? idleMs : 0.0,
          ",\"lag_us\":", present.n ? msOf(present, timed) * double(std::max(timed, 1u)) * 1000.0 / double(present.n) : 0.0,
          ",\"occupancy_kb\":{\"mean\":", g_state.feOccupancyN ? double(g_state.feOccupancySum) / double(g_state.feOccupancyN) / 1024.0 : 0.0,
          ",\"max\":", double(g_state.feOccupancyMax) / 1024.0, "}",
          ",\"ring_full\":{\"n\":", perFrame(g_state.feRingFull.n, timed), ",\"ms\":", msOf(g_state.feRingFull, timed), "}",
          ",\"present_wait\":{\"n\":", perFrame(g_state.fePresentWait.n, timed), ",\"ms\":", msOf(g_state.fePresentWait, timed), "}");

        line += ",\"drain\":{";
        bool firstDrain = true;
        for (uint32_t i = 0; i < uint32_t(FeDrain::Count); i++) {
          const auto& d = g_state.feDrains[i];

          if (!d.n)
            continue;

          line += str::format(firstDrain ? "" : ",", "\"", g_feDrainNames[i], "\":{\"n\":",
            perFrame(d.n, timed), ",\"ms\":", msOf(d, timed), "}");
          firstDrain = false;
        }
        line += "}";

        uint32_t threadCount = std::min(g_state.feThreadCount, kFeThreads);

        line += ",\"threads\":[";
        for (uint32_t i = 0; i < threadCount; i++)
          line += str::format(i ? "," : "", g_state.feThreadIds[i]);
        line += "]";

        line += ",\"census\":{";
        bool firstCall = true;
        for (uint32_t i = 0; i < uint32_t(FeCall::Count); i++) {
          Sample sum;
          for (const auto& s : g_state.feCalls[i]) {
            sum.n     += s.n;
            sum.ticks += s.ticks;
          }

          if (!sum.n)
            continue;

          line += str::format(firstCall ? "" : ",", "\"", g_feCallNames[i], "\":{\"n\":",
            perFrame(sum.n, timed), ",\"ms\":", msOf(sum, timed));

          if (threadCount > 1u) {
            line += ",\"t\":[";
            for (uint32_t t = 0; t < threadCount; t++)
              line += str::format(t ? "," : "", perFrame(g_state.feCalls[i][t].n, timed));
            line += "]";
          }

          line += "}";
          firstCall = false;
        }
        line += "}}";
      }
    }

    line += str::format(",\"stalls\":{\"sync_cs\":{\"n\":", double(syncCountDelta) / double(all),
      ",\"ms\":", double(syncTicksDelta) / 1000.0 / double(all), "}",
      ",\"emit_block\":{\"n\":", perFrame(g_state.stallEmitBlock.n, timed),
      ",\"ms\":", msOf(g_state.stallEmitBlock, timed), "}}");

    line += str::format(",\"cs\":{\"busy_ms\":", busyUs / 1000.0 / double(all),
      ",\"idle_ms\":", idleUs / 1000.0 / double(all),
      ",\"chunks\":", double(chunkDelta) / double(all),
      ",\"cmds\":", double(chunkCmdDelta) / double(all), "}}\n");

    g_state.file << line;
    g_state.file.flush();

    // reset for the next window
    g_state.calls.fill(Sample{});
    for (auto& row : g_state.map)
      row.fill(Sample{});
    g_state.discardSlice   = Sample{};
    g_state.cbRing.fill(Sample{}); // blessed: cb-ring
    g_state.stallEmitBlock = Sample{};
    // blessed: threaded-fe
    for (auto& row : g_state.feCalls)
      row.fill(Sample{});
    g_state.feDrains.fill(Sample{});
    g_state.feDirect.fill(Sample{});
    g_state.feRingFull     = Sample{};
    g_state.feReplay       = Sample{};
    g_state.fePresentWait  = Sample{}; // blessed: threaded-fe-2
    g_state.feOccupancySum = 0;
    g_state.feOccupancyMax = 0;
    g_state.feOccupancyN   = 0;
    g_state.feSnapLast     = g_state.feSnap;
    g_state.frameTimesMs.clear();
    g_state.activeFrames       = 0;
    g_state.windowFrames       = 0;
    g_state.windowStamp        = nowStamp;
    g_state.windowStart        = now;
    g_state.lastCsIdleTicks    = cs.idleTicks;
    g_state.lastCsChunkCount   = cs.chunkCount;
    g_state.lastCsChunkCmdCount= cs.chunkCmdCount;
    g_state.lastCsSyncCount    = cs.syncCount;
    g_state.lastCsSyncTicks    = cs.syncTicks;
  }


  void onPresent(const CsSnapshot& cs) {
    if (!enabled())
      return;

    ensureInit();

    auto now = dxvk::high_resolution_clock::now();

    if (g_state.havePresent) {
      // the frame that just ended was timed if g_active is still set
      double deltaMs = std::chrono::duration<double, std::milli>(now - g_state.lastPresent).count();
      g_state.windowFrames += 1;
      if (g_active)
        g_state.activeFrames += 1;
      if (!g_active || g_state.every == 1)
        g_state.frameTimesMs.push_back(deltaMs);
    } else {
      g_state.havePresent      = true;
      g_state.windowStart      = now;
      g_state.lastCsIdleTicks  = cs.idleTicks;
      g_state.lastCsChunkCount = cs.chunkCount;
      g_state.lastCsChunkCmdCount = cs.chunkCmdCount;
      g_state.lastCsSyncCount  = cs.syncCount;
      g_state.lastCsSyncTicks  = cs.syncTicks;
    }

    g_state.lastPresent = now;
    g_state.frameIndex += 1;

    if (g_state.frameIndex % kWindowFrames == 0)
      flushWindow(cs, now);

    g_active = (g_state.frameIndex % g_state.every) == 0;
  }

}

#endif // BLESSED_PROBE_ENABLED
