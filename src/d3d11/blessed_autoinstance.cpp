// blessed: auto-instancing stage 0 -- the census that counts which draws one instanced draw could replace
#include "blessed_autoinstance.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#include "d3d11_buffer.h"
#include "d3d11_depth_stencil.h"
#include "d3d11_input_layout.h"
#include "d3d11_shader.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    constexpr uint32_t WindowPresents = 120u;
    constexpr uint32_t SlotCount      = D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT;
    constexpr uint32_t StageCount     = 2u; // 0: vs, 1: ps
    constexpr uint32_t MaxCopyBytes   = 4096u;

    bool ComputeCensusEnabled() {
      return env::getEnvVar("BLESSED_AUTOINSTANCE_CENSUS") == "1";
    }

    // 64-bit mixing, order dependent. Not cryptographic: a collision
    // only miscounts one census entry.
    inline uint64_t Mix(uint64_t h, uint64_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h *= 0xff51afd7ed558ccdull;
      return h ^ (h >> 33);
    }

    inline uint64_t MixPtr(uint64_t h, const void* p) {
      return Mix(h, uint64_t(reinterpret_cast<uintptr_t>(p)));
    }

    uint64_t HashBytes(const void* data, size_t size) {
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
      uint64_t h = 0xcbf29ce484222325ull;
      size_t i = 0;

      for (; i + 8u <= size; i += 8u) {
        uint64_t w;
        std::memcpy(&w, bytes + i, 8u);
        h = Mix(h, w);
      }

      for (; i < size; i++)
        h = Mix(h, bytes[i]);

      return h;
    }

    const D3D11ShaderType Stages[StageCount] = { D3D11ShaderType::eVertex, D3D11ShaderType::ePixel };

    // Contents of one bound cbuffer slot as a draw saw it: hash plus the
    // byte count read. Zero hash with zero size: unbound or unreadable.
    struct SlotContent {
      uint64_t hash  = 0u;
      uint32_t bytes = 0u;
      bool     valid = false;
    };

    using DrawContents = std::array<std::array<SlotContent, SlotCount>, StageCount>;

    // One bucket of the current segment: how many candidate draws share
    // the key, and the first member's cbuffer contents.
    struct Bucket {
      uint32_t      count = 0u;
      DrawContents  first = { };
    };

    struct Counters {
      uint64_t draws             = 0u;
      uint64_t candidates        = 0u;
      uint64_t depthOnlyDraws    = 0u;
      // consecutive runs
      uint64_t consecSaved       = 0u;
      uint64_t consecRuns        = 0u;
      uint64_t consecMaxRun      = 0u;
      uint64_t consecSavedDepth  = 0u;
      uint64_t consecIdentical   = 0u;   // merged draws with no cbuffer change at all
      uint64_t consecBytes       = 0u;   // sum of changing bytes over merged draws
      std::array<std::array<uint64_t, SlotCount>, StageCount> consecVary = { };
      // bucketed, sortable segments
      uint64_t sortSegments      = 0u;
      uint64_t sortDraws         = 0u;
      uint64_t sortSaved         = 0u;
      uint64_t sortSavedVsOnly   = 0u;   // merged draws whose ps cbuffers match the bucket's first
      uint64_t sortMaxBucket     = 0u;
      uint64_t sortBytes         = 0u;
      std::array<std::array<uint64_t, SlotCount>, StageCount> sortVary = { };
      // bucketed, segments where reordering is not legal as-is
      uint64_t otherSaved        = 0u;
      // segment ends by cause
      uint64_t barriers          = 0u;
      uint64_t targetChanges     = 0u;
    };

    struct CensusState {
      bool                  logTried = false;
      std::ofstream         log;
      dxvk::high_resolution_clock::time_point start;
      bool                  startSet = false;
      uint32_t              presents = 0u;
      Counters              c;

      // current segment
      uint64_t              passKey       = 0u;
      bool                  segOpen       = false;
      bool                  segSortable   = false;
      uint64_t              segCandidates = 0u;
      std::unordered_map<uint64_t, Bucket> buckets;

      // previous draw, for consecutive runs
      bool                  prevCandidate = false;
      uint64_t              prevKey       = 0u;
      uint64_t              runLength     = 0u;
      DrawContents          prevContents  = { };

      // input layout -> has per-instance elements, cached for the last layout
      const D3D11InputLayout* lastLayout  = nullptr;
      bool                  lastLayoutInstanced = false;
    };

    CensusState& State() {
      static CensusState s_state;
      return s_state;
    }

    bool LayoutHasInstanceData(CensusState& s, const D3D11InputLayout* layout) {
      if (layout == s.lastLayout)
        return s.lastLayoutInstanced;

      bool instanced = false;

      if (layout) {
        uint32_t first = layout->GetAttributeCount();
        uint32_t count = layout->GetBindingCount();

        for (uint32_t i = 0; i < count; i++) {
          if (layout->GetInput(first + i).binding().inputRate == VK_VERTEX_INPUT_RATE_INSTANCE)
            instanced = true;
        }
      }

      s.lastLayout = layout;
      s.lastLayoutInstanced = instanced;
      return instanced;
    }

    uint64_t HashTargets(const D3D11ContextState& state) {
      uint64_t h = 0x7a2fu;

      for (uint32_t i = 0; i < state.om.maxRtv; i++)
        h = MixPtr(h, state.om.rtvs[i].ptr());

      return MixPtr(h, state.om.dsv.ptr());
    }

    bool HasRtv(const D3D11ContextState& state) {
      for (uint32_t i = 0; i < state.om.maxRtv; i++) {
        if (state.om.rtvs[i].ptr())
          return true;
      }

      return false;
    }

    bool HasUav(const D3D11ContextState& state) {
      if (state.om.maxUav > state.om.minUav) {
        for (uint32_t i = state.om.minUav; i < state.om.maxUav; i++) {
          if (state.om.uavs[i].ptr())
            return true;
        }
      }

      return false;
    }

    bool HasStreamOut(const D3D11ContextState& state) {
      for (const auto& t : state.so.targets) {
        if (t.buffer.ptr())
          return true;
      }

      return false;
    }

    // Depth-only and commutative: the final depth buffer does not depend
    // on the order such draws run in. A draw with no rtv, no uav, no
    // stream-out, no stencil and no depth write has no output at all, so
    // it commutes with anything (queries end a segment on their own).
    bool IsSortable(const D3D11ContextState& state) {
      if (HasRtv(state) || HasUav(state) || HasStreamOut(state) || !state.om.dsv.ptr())
        return false;

      D3D11DepthStencilState* ds = state.om.dsState.ptr();

      if (!ds)
        return true; // d3d11 default: depth on, write all, LESS, stencil off

      const D3D11_DEPTH_STENCIL_DESC& desc = ds->Desc();

      if (desc.StencilEnable)
        return false;

      if (!desc.DepthEnable || desc.DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ALL)
        return true;

      return desc.DepthFunc == D3D11_COMPARISON_LESS
          || desc.DepthFunc == D3D11_COMPARISON_LESS_EQUAL;
    }

    uint64_t HashBufferIdentity(uint64_t h, D3D11Buffer* buffer) {
      h = MixPtr(h, buffer);
      // a discarded dynamic buffer has new contents: its map pointer moved
      return MixPtr(h, buffer ? buffer->GetMapPtr() : nullptr);
    }

    // Everything a merged draw must share, cbuffer contents excepted.
    uint64_t HashDrawKey(const D3D11ContextState& state, const BlessedAutoInstanceDraw& draw) {
      uint64_t h = 0x51u;

      h = MixPtr(h, state.vs.ptr());
      h = MixPtr(h, state.ps.ptr());
      h = MixPtr(h, state.ia.inputLayout.ptr());
      h = Mix(h, uint64_t(state.ia.primitiveTopology));

      for (uint32_t i = 0; i < state.ia.maxVbCount; i++) {
        const auto& vb = state.ia.vertexBuffers[i];
        h = HashBufferIdentity(h, vb.buffer.ptr());
        h = Mix(h, (uint64_t(vb.offset) << 32) | vb.stride);
      }

      h = HashBufferIdentity(h, state.ia.indexBuffer.buffer.ptr());
      h = Mix(h, (uint64_t(state.ia.indexBuffer.offset) << 32) | uint32_t(state.ia.indexBuffer.format));

      h = MixPtr(h, state.rs.state.ptr());
      h = Mix(h, (uint64_t(state.rs.numViewports) << 32) | state.rs.numScissors);

      if (state.rs.numViewports)
        h = Mix(h, HashBytes(&state.rs.viewports[0], sizeof(D3D11_VIEWPORT)));

      if (state.rs.numScissors)
        h = Mix(h, HashBytes(&state.rs.scissors[0], sizeof(D3D11_RECT)));

      h = MixPtr(h, state.om.cbState.ptr());
      h = MixPtr(h, state.om.dsState.ptr());
      h = Mix(h, (uint64_t(state.om.stencilRef) << 32) | state.om.sampleMask);
      h = Mix(h, HashBytes(state.om.blendFactor, sizeof(state.om.blendFactor)));

      for (uint32_t s = 0; s < StageCount; s++) {
        const auto& srvs = state.srv[Stages[s]];
        const auto& smps = state.samplers[Stages[s]];
        const auto& cbvs = state.cbv[Stages[s]];

        for (uint32_t i = 0; i < srvs.maxCount; i++)
          h = MixPtr(h, srvs.views[i].ptr());

        for (uint32_t i = 0; i < smps.maxCount; i++)
          h = MixPtr(h, smps.samplers[i].ptr());

        // binding only: the contents are what may differ
        for (uint32_t i = 0; i < cbvs.maxCount; i++) {
          const auto& cb = cbvs.buffers[i];
          h = MixPtr(h, cb.buffer.ptr());
          h = Mix(h, (uint64_t(cb.constantOffset) << 32) | cb.constantCount);
        }
      }

      h = Mix(h, uint64_t(draw.indexed));
      h = Mix(h, (uint64_t(draw.count) << 32) | draw.first);
      h = Mix(h, uint64_t(uint32_t(draw.baseVertex)));
      return h;
    }

    void ReadContents(const D3D11ContextState& state, DrawContents& out) {
      for (uint32_t s = 0; s < StageCount; s++) {
        const auto& cbvs = state.cbv[Stages[s]];

        for (uint32_t i = 0; i < SlotCount; i++) {
          SlotContent& slot = out[s][i];
          slot = SlotContent();

          if (i >= cbvs.maxCount)
            continue;

          const auto& cb = cbvs.buffers[i];
          D3D11Buffer* buffer = cb.buffer.ptr();

          if (!buffer)
            continue;

          const uint8_t* base = reinterpret_cast<const uint8_t*>(buffer->GetMapPtr());

          if (!base)
            continue; // not host visible (a DEFAULT cbuffer): unknown

          buffer->GetBuffer()->blessedMarkCpuRead(); // blessed: perf-halfrate -- census reads: keep it cached

          size_t offset = size_t(cb.constantOffset) * 16u;
          size_t size   = size_t(cb.constantCount) * 16u;
          size_t total  = buffer->Desc()->ByteWidth;

          if (offset >= total)
            continue;

          size = std::min(std::min(size, total - offset), size_t(MaxCopyBytes));

          slot.hash  = HashBytes(base + offset, size);
          slot.bytes = uint32_t(size);
          slot.valid = true;
        }
      }
    }

    // Counts slots whose contents differ between a and b; returns the
    // bytes that would go per instance. Unreadable slots count as
    // changing (the merge would have to fetch them per instance).
    uint64_t CountVarying(const DrawContents& a, const DrawContents& b,
            std::array<std::array<uint64_t, SlotCount>, StageCount>& vary,
            bool* psVaries) {
      uint64_t bytes = 0u;
      *psVaries = false;

      for (uint32_t s = 0; s < StageCount; s++) {
        for (uint32_t i = 0; i < SlotCount; i++) {
          const SlotContent& x = a[s][i];
          const SlotContent& y = b[s][i];

          if (!x.valid && !y.valid && !x.bytes && !y.bytes)
            continue;

          if (x.valid && y.valid && x.hash == y.hash && x.bytes == y.bytes)
            continue;

          vary[s][i]++;
          bytes += std::max(x.bytes, y.bytes);

          if (s == 1u)
            *psVaries = true;
        }
      }

      return bytes;
    }

    void CloseSegment(CensusState& s) {
      if (!s.segOpen)
        return;

      uint64_t saved = s.segCandidates - s.buckets.size();

      if (s.segSortable) {
        s.c.sortSegments++;
        s.c.sortSaved += saved;

        for (const auto& e : s.buckets)
          s.c.sortMaxBucket = std::max<uint64_t>(s.c.sortMaxBucket, e.second.count);
      } else {
        s.c.otherSaved += saved;
      }

      s.buckets.clear();
      s.segOpen       = false;
      s.segCandidates = 0u;
    }

    void CloseRun(CensusState& s) {
      if (s.runLength >= 2u) {
        s.c.consecRuns++;
        s.c.consecMaxRun = std::max(s.c.consecMaxRun, s.runLength);
      }

      s.prevCandidate = false;
      s.runLength     = 0u;
    }

    void EnsureLog(CensusState& s) {
      if (s.logTried)
        return;

      s.logTried = true;
      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

      if (dir.empty()) {
        Logger::warn("BlessedAutoInstance: census on but BLESSED_PROBE_DIR is not set, no output");
        return;
      }

      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      s.log.open(dir + env::PlatformDirSlash + "autoinstance.jsonl", std::ios::out | std::ios::app);
    }

    std::string VaryJson(const std::array<std::array<uint64_t, SlotCount>, StageCount>& vary, double frames) {
      static const char* names[StageCount] = { "vs", "ps" };
      std::string out = "{";
      bool first = true;

      for (uint32_t s = 0; s < StageCount; s++) {
        for (uint32_t i = 0; i < SlotCount; i++) {
          if (!vary[s][i])
            continue;

          out += str::format(first ? "" : ",", "\"", names[s], ":b", i, "\":", double(vary[s][i]) / frames);
          first = false;
        }
      }

      return out + "}";
    }

  }


  namespace blessed_autoinstance_detail {
    extern const bool g_census = ComputeCensusEnabled();
  }


  void BlessedAutoInstance::OnDrawSlow(const D3D11ContextState& state, const BlessedAutoInstanceDraw& draw) {
    CensusState& s = State();
    s.c.draws++;

    // segment: same render targets, same sortability, no barrier since
    // the last draw. A non-sortable draw between sortable ones is an
    // order point: buckets never move a draw across it.
    uint64_t passKey = HashTargets(state);
    bool sortable = IsSortable(state);

    if (s.segOpen && passKey != s.passKey) {
      s.c.targetChanges++;
      CloseSegment(s);
      CloseRun(s);
    } else if (s.segOpen && sortable != s.segSortable) {
      CloseSegment(s);
    }

    if (!s.segOpen)
      s.segSortable = sortable;

    s.passKey = passKey;
    s.segOpen = true;

    if (sortable)
      s.c.depthOnlyDraws++;

    bool candidate = draw.instanceCount == 1u && draw.startInstance == 0u
      && !state.gs.ptr() && !state.hs.ptr() && !state.ds.ptr()
      && !HasUav(state) && !HasStreamOut(state) && !state.pr.predicateObject.ptr()
      && !LayoutHasInstanceData(s, state.ia.inputLayout.ptr());

    if (!candidate) {
      CloseRun(s);
      return;
    }

    s.c.candidates++;
    s.segCandidates++;

    uint64_t key = HashDrawKey(state, draw);

    DrawContents contents;
    ReadContents(state, contents);

    // consecutive run
    if (s.prevCandidate && key == s.prevKey) {
      bool psVaries = false;
      uint64_t bytes = CountVarying(s.prevContents, contents, s.c.consecVary, &psVaries);

      s.c.consecSaved++;
      s.c.consecBytes += bytes;

      if (!bytes)
        s.c.consecIdentical++;

      if (sortable)
        s.c.consecSavedDepth++;

      s.runLength++;
    } else {
      CloseRun(s);
      s.runLength = 1u;
    }

    s.prevCandidate = true;
    s.prevKey       = key;
    s.prevContents  = contents;

    // bucket inside the segment
    Bucket& bucket = s.buckets[key];

    if (!bucket.count) {
      bucket.first = contents;
    } else if (s.segSortable) {
      bool psVaries = false;
      s.c.sortBytes += CountVarying(bucket.first, contents, s.c.sortVary, &psVaries);

      if (!psVaries)
        s.c.sortSavedVsOnly++;
    }

    bucket.count++;

    if (s.segSortable)
      s.c.sortDraws++;
  }


  void BlessedAutoInstance::OnBarrierSlow() {
    CensusState& s = State();

    if (s.segOpen)
      s.c.barriers++;

    CloseSegment(s);
    CloseRun(s);
  }


  void BlessedAutoInstance::OnPresent() {
    if (!IsCensusEnabled())
      return;

    CensusState& s = State();

    CloseSegment(s);
    CloseRun(s);
    s.passKey = 0u;

    if (!s.startSet) {
      s.startSet = true;
      s.start = dxvk::high_resolution_clock::now();
      Logger::info("BlessedAutoInstance: census on (BLESSED_AUTOINSTANCE_CENSUS=1), autoinstance.jsonl every 120 presents");
    }

    if (++s.presents < WindowPresents)
      return;

    EnsureLog(s);

    double t = std::chrono::duration<double>(dxvk::high_resolution_clock::now() - s.start).count();
    double f = double(s.presents);
    const Counters& c = s.c;

    if (s.log.is_open()) {
      s.log << str::format("{\"t\":", t,
        ",\"frames\":", s.presents,
        ",\"draws\":", double(c.draws) / f,
        ",\"candidates\":", double(c.candidates) / f,
        ",\"depth_only_draws\":", double(c.depthOnlyDraws) / f,
        ",\"consec_saved\":", double(c.consecSaved) / f,
        ",\"consec_saved_depth_only\":", double(c.consecSavedDepth) / f,
        ",\"consec_runs\":", double(c.consecRuns) / f,
        ",\"consec_max_run\":", c.consecMaxRun,
        ",\"consec_identical\":", double(c.consecIdentical) / f,
        ",\"consec_bytes_per_merged\":", c.consecSaved ? double(c.consecBytes) / double(c.consecSaved) : 0.0,
        ",\"consec_vary\":", VaryJson(c.consecVary, f),
        ",\"sort_segments\":", double(c.sortSegments) / f,
        ",\"sort_candidates\":", double(c.sortDraws) / f,
        ",\"sort_saved\":", double(c.sortSaved) / f,
        ",\"sort_saved_vs_only\":", double(c.sortSavedVsOnly) / f,
        ",\"sort_max_bucket\":", c.sortMaxBucket,
        ",\"sort_bytes_per_merged\":", c.sortSaved ? double(c.sortBytes) / double(c.sortSaved) : 0.0,
        ",\"sort_vary\":", VaryJson(c.sortVary, f),
        ",\"other_saved_if_reordered\":", double(c.otherSaved) / f,
        ",\"segment_ends_barrier\":", double(c.barriers) / f,
        ",\"segment_ends_targets\":", double(c.targetChanges) / f,
        "}\n");
      s.log.flush();
    }

    s.presents = 0u;
    s.c = Counters();
  }

}
