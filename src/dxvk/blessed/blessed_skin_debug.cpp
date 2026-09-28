// blessed: actor-skinning diagnostics -- see blessed_skin_debug.h
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "blessed_skin_debug.h"
#include "blessed_scene.h"

#include "../dxvk_cmdlist.h"
#include "../dxvk_context.h"

#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"

namespace dxvk {

  namespace {

    constexpr uint32_t MaxTraceEntries = 16u;
    constexpr uint32_t MaxRawVertices  = 4u;

    VkDeviceSize AlignUp(VkDeviceSize v) {
      return (v + 15u) & ~VkDeviceSize(15u);
    }

    float HalfToFloat(uint16_t h) {
      uint32_t sign = uint32_t(h >> 15) << 31;
      uint32_t exp  = (h >> 10) & 0x1Fu;
      uint32_t man  = h & 0x3FFu;
      uint32_t bits;

      if (exp == 0u) {
        if (man == 0u) {
          bits = sign;
        } else {
          // subnormal half -> normal float
          exp = 127u - 15u + 1u;
          while (!(man & 0x400u)) { man <<= 1; exp--; }
          man &= 0x3FFu;
          bits = sign | (exp << 23) | (man << 13);
        }
      } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (man << 13);
      } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (man << 13);
      }

      float f;
      std::memcpy(&f, &bits, sizeof(f));
      return f;
    }

    void DecodePosition(VkFormat fmt, const uint8_t* p, float out[4]) {
      out[3] = 0.0f;

      if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
        uint16_t h[4];
        std::memcpy(h, p, sizeof(h));
        for (uint32_t i = 0; i < 4; i++)
          out[i] = HalfToFloat(h[i]);
      } else {
        uint32_t n = fmt == VK_FORMAT_R32G32B32A32_SFLOAT ? 4u : 3u;
        std::memcpy(out, p, n * sizeof(float));
      }
    }

    // mirrors blessed_skin.comp exactly (and Skinned::GetBoneTransformMatrix)
    void CpuSkin(const float* bones, const float pos[3], const uint32_t rows[4],
        const float w[4], const float pivot[3], float out[3]) {
      float m[3][4] = { };

      for (uint32_t k = 0; k < 4; k++) {
        for (uint32_t r = 0; r < 3; r++) {
          uint32_t row = std::min(rows[k] + r, 239u);
          for (uint32_t c = 0; c < 4; c++)
            m[r][c] += w[k] * bones[row * 4 + c];
        }
      }

      for (uint32_t r = 0; r < 3; r++) {
        m[r][3] -= pivot[r];
        out[r] = m[r][0] * pos[0] + m[r][1] * pos[1] + m[r][2] * pos[2] + m[r][3];
      }
    }

    std::string Vec(const float* v, uint32_t n) {
      std::string s = "(";
      for (uint32_t i = 0; i < n; i++) {
        if (i) s += ", ";
        s += str::format(v[i]);
      }
      return s + ")";
    }

  }


  bool BlessedSkinTrace::enabled() {
    static bool s_enabled = env::getEnvVar("BLESSED_SCENE_SKIN_TRACE") == "1";
    return s_enabled;
  }


  uint64_t BlessedSkinTrace::startFrame() {
    static uint64_t s_frame = [] {
      std::string s = env::getEnvVar("BLESSED_SCENE_SKIN_TRACE_FRAME");
      return s.empty() ? uint64_t(300u) : uint64_t(std::strtoull(s.c_str(), nullptr, 10));
    }();
    return s_frame;
  }


  void BlessedSkinTrace::record(
          DxvkDevice*                         device,
          DxvkContext*                        ctx,
    const Rc<DxvkCommandList>&                cmd,
    const std::vector<BlessedSkinTraceInput>& inputs) {
    if (m_state != State::Idle || inputs.empty())
      return;

    // 1. lay out the readback buffer
    VkDeviceSize total = 0;

    for (const BlessedSkinTraceInput& in : inputs) {
      if (m_entries.size() >= MaxTraceEntries)
        break;

      const BlessedSceneSkinnedDraw& d = *in.draw;

      if (!in.outPositions.ptr() || d.vertexCount == 0 || uint32_t(d.baseVertex) >= d.vertexCount)
        continue;

      Entry e;
      e.vertexCount = d.vertexCount;
      e.indexCount  = d.indexCount;
      e.startIndex  = d.startIndex;
      e.baseVertex  = d.baseVertex;
      e.index32     = d.indexType == VK_INDEX_TYPE_UINT32;
      e.occurrence  = in.occurrence;
      e.posFormat   = d.posFormat;
      e.posStride   = d.posStride;
      e.idxStride   = d.idxStride;
      e.wtStride    = d.wtStride;
      e.rawCount    = std::min(MaxRawVertices, d.vertexCount - uint32_t(d.baseVertex));
      e.posVb       = uint64_t(reinterpret_cast<uintptr_t>(d.posVb.buffer().ptr()));
      e.idxVb       = uint64_t(reinterpret_cast<uintptr_t>(d.idxVb.buffer().ptr()));
      e.wtVb        = uint64_t(reinterpret_cast<uintptr_t>(d.wtVb.buffer().ptr()));
      e.ib          = uint64_t(reinterpret_cast<uintptr_t>(d.ib.buffer().ptr()));
      e.posVbOffset = d.posVb.offset();
      e.idxVbOffset = d.idxVb.offset();
      e.wtVbOffset  = d.wtVb.offset();
      e.hasTagCam   = d.hasTagCam;
      e.bonesUsage  = d.bonesUsage;
      std::memcpy(e.pivot,  d.pivot,  sizeof(e.pivot));
      std::memcpy(e.tagCam, d.tagCam, sizeof(e.tagCam));
      std::memcpy(e.bones,  d.bones,  sizeof(e.bones));

      VkDeviceSize ibBytes = VkDeviceSize(d.indexCount) * (e.index32 ? 4u : 2u);

      e.outOffset = total; total += AlignUp(VkDeviceSize(d.vertexCount) * 12u);
      e.ibOffset  = total; total += AlignUp(ibBytes);
      e.posOffset = total; total += AlignUp(VkDeviceSize(e.rawCount) * d.posStride);
      e.idxOffset = total; total += AlignUp(VkDeviceSize(e.rawCount) * d.idxStride);
      e.wtOffset  = total; total += AlignUp(VkDeviceSize(e.rawCount) * d.wtStride);

      m_entries.push_back(e);
    }

    if (m_entries.empty() || total == 0)
      return;

    DxvkBufferCreateInfo info = { };
    info.size      = total;
    info.usage     = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages    = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    info.access    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    info.debugName = "blessed skin trace";

    m_readback = device->createBuffer(info,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!m_readback.ptr() || !m_readback->getSliceInfo().mapPtr) {
      Logger::warn("blessed: skin trace: could not create a host-visible readback buffer");
      m_entries.clear();
      m_state = State::Done;
      return;
    }

    // 2. compute writes (skinned positions) -> transfer reads
    VkMemoryBarrier2 toCopy = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    toCopy.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toCopy.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    toCopy.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    toCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo toCopyDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    toCopyDep.memoryBarrierCount = 1u;
    toCopyDep.pMemoryBarriers    = &toCopy;

    // blessed: raw
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &toCopyDep);

    DxvkResourceBufferInfo dst = m_readback->getSliceInfo();

    auto copy = [&] (const DxvkResourceBufferInfo& src, VkDeviceSize srcOffset,
                     VkDeviceSize dstOffset, VkDeviceSize size) {
      if (!size || !src.buffer)
        return;

      VkBufferCopy2 region = { VK_STRUCTURE_TYPE_BUFFER_COPY_2 };
      region.srcOffset = src.offset + srcOffset;
      region.dstOffset = dst.offset + dstOffset;
      region.size      = size;

      VkCopyBufferInfo2 copyInfo = { VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2 };
      copyInfo.srcBuffer   = src.buffer;
      copyInfo.dstBuffer   = dst.buffer;
      copyInfo.regionCount = 1u;
      copyInfo.pRegions    = &region;

      cmd->cmdCopyBuffer(DxvkCmdBuffer::ExecBuffer, &copyInfo);
    };

    size_t n = 0;
    for (const BlessedSkinTraceInput& in : inputs) {
      if (n >= m_entries.size())
        break;

      const BlessedSceneSkinnedDraw& d = *in.draw;

      if (!in.outPositions.ptr() || d.vertexCount == 0 || uint32_t(d.baseVertex) >= d.vertexCount)
        continue;

      const Entry& e = m_entries[n++];
      VkDeviceSize base = VkDeviceSize(e.baseVertex);

      copy(in.outPositions->getSliceInfo(), 0u, e.outOffset, VkDeviceSize(e.vertexCount) * 12u);
      copy(d.ib.getSliceInfo(), VkDeviceSize(e.startIndex) * (e.index32 ? 4u : 2u),
        e.ibOffset, VkDeviceSize(e.indexCount) * (e.index32 ? 4u : 2u));
      copy(d.posVb.getSliceInfo(), base * e.posStride, e.posOffset, VkDeviceSize(e.rawCount) * e.posStride);
      copy(d.idxVb.getSliceInfo(), base * e.idxStride, e.idxOffset, VkDeviceSize(e.rawCount) * e.idxStride);
      copy(d.wtVb.getSliceInfo(),  base * e.wtStride,  e.wtOffset,  VkDeviceSize(e.rawCount) * e.wtStride);

      cmd->track(in.outPositions, DxvkAccess::Read);
      cmd->track(d.ib.buffer(),    DxvkAccess::Read);
      cmd->track(d.posVb.buffer(), DxvkAccess::Read);
      cmd->track(d.idxVb.buffer(), DxvkAccess::Read);
      cmd->track(d.wtVb.buffer(),  DxvkAccess::Read);
    }

    // 3. transfer writes -> host reads, and the copies above must finish
    // before the builds that follow read the positions (read/read: no hazard)
    VkMemoryBarrier2 toHost = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    toHost.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    toHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toHost.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
    toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

    VkDependencyInfo toHostDep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    toHostDep.memoryBarrierCount = 1u;
    toHostDep.pMemoryBarriers    = &toHost;

    // blessed: raw
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &toHostDep);

    cmd->track(m_readback, DxvkAccess::Write);

    (void) ctx;
    m_state      = State::Pending;
    m_pollFrames = 0;

    Logger::info(str::format("blessed: skin trace: recorded ", m_entries.size(),
      " skinned entries, ", total, " bytes; log follows once the gpu is done"));
  }


  void BlessedSkinTrace::poll() {
    if (m_state != State::Pending)
      return;

    // non-blocking: the copy is tracked as a write, so the buffer stays
    // "in use" until the command list holding it has completed
    if (m_readback->isInUse(DxvkAccess::Write)) {
      if (++m_pollFrames == 600u)
        Logger::warn("blessed: skin trace: readback still pending after 600 frames");
      return;
    }

    writeLog();

    m_readback = nullptr;
    m_entries.clear();
    m_state = State::Done;
  }


  void BlessedSkinTrace::writeLog() {
    std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");
    std::string path = dir.empty() ? std::string("skin-trace.log")
                                   : dir + env::PlatformDirSlash + "skin-trace.log";

    if (!dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
    }

    std::ofstream f(path, std::ios::out | std::ios::trunc);

    if (!f.is_open()) {
      Logger::warn(str::format("blessed: skin trace: could not open '", path, "'"));
      return;
    }

    const uint8_t* base = reinterpret_cast<const uint8_t*>(m_readback->getSliceInfo().mapPtr);

    f << "# blessed skin trace: " << m_entries.size() << " entries\n";
    f << "# out = gpu output (blessed_skin.comp); cpu = same inputs re-skinned on the host\n";
    f << "# bbox = over the vertices this draw's index range references\n";

    for (size_t i = 0; i < m_entries.size(); i++) {
      const Entry& e = m_entries[i];
      const float* out = reinterpret_cast<const float*>(base + e.outOffset);

      // bbox over the referenced vertices only (other partitions of a shared
      // vb are skinned with this draw's palette too, and are garbage by design)
      float lo[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
      float hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
      uint32_t outOfRange = 0, nonFinite = 0, minIdx = UINT32_MAX, maxIdx = 0;

      for (uint32_t k = 0; k < e.indexCount; k++) {
        uint32_t idx;
        if (e.index32) {
          std::memcpy(&idx, base + e.ibOffset + 4u * k, 4u);
        } else {
          uint16_t i16;
          std::memcpy(&i16, base + e.ibOffset + 2u * k, 2u);
          idx = i16;
        }

        minIdx = std::min(minIdx, idx);
        maxIdx = std::max(maxIdx, idx);

        uint64_t v = uint64_t(idx) + uint64_t(e.baseVertex);
        if (v >= e.vertexCount) {
          outOfRange++;
          continue;
        }

        const float* p = &out[3u * v];
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
          nonFinite++;
          continue;
        }

        for (uint32_t c = 0; c < 3; c++) {
          lo[c] = std::min(lo[c], p[c]);
          hi[c] = std::max(hi[c], p[c]);
        }
      }

      float size[3] = { hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] };
      float center[3] = { 0.5f * (hi[0] + lo[0]), 0.5f * (hi[1] + lo[1]), 0.5f * (hi[2] + lo[2]) };

      f << "\n[" << i << "] vertices=" << e.vertexCount << " index_count=" << e.indexCount
        << " start_index=" << e.startIndex << " base_vertex=" << e.baseVertex
        << " index=" << (e.index32 ? "u32" : "u16") << " occurrence=" << e.occurrence << "\n";
      f << "  streams: pos vb=" << std::hex << e.posVb << std::dec << "+" << e.posVbOffset
        << " stride=" << e.posStride << " format=" << uint32_t(e.posFormat)
        << " | idx vb=" << std::hex << e.idxVb << std::dec << "+" << e.idxVbOffset << " stride=" << e.idxStride
        << " | wt vb=" << std::hex << e.wtVb << std::dec << "+" << e.wtVbOffset << " stride=" << e.wtStride
        << " | ib=" << std::hex << e.ib << std::dec << "\n";
      f << "  indices: min=" << minIdx << " max=" << maxIdx
        << " out_of_range=" << outOfRange << " non_finite=" << nonFinite << "\n";
      f << "  bbox: min=" << Vec(lo, 3) << " max=" << Vec(hi, 3)
        << " size=" << Vec(size, 3) << " center=" << Vec(center, 3)
        << " center_dist=" << std::sqrt(center[0] * center[0] + center[1] * center[1] + center[2] * center[2]) << "\n";
      f << "  pivot (subtracted)=" << Vec(e.pivot, 3)
        << " tag_campos=" << (e.hasTagCam ? Vec(e.tagCam, 3) : std::string("none"))
        << " bones_usage=" << e.bonesUsage << "\n";
      f << "  bone0: r0=" << Vec(&e.bones[0], 4) << " r1=" << Vec(&e.bones[4], 4)
        << " r2=" << Vec(&e.bones[8], 4) << "\n";

      for (uint32_t v = 0; v < e.rawCount; v++) {
        float pos[4];
        DecodePosition(e.posFormat, base + e.posOffset + VkDeviceSize(v) * e.posStride, pos);

        uint32_t word;
        std::memcpy(&word, base + e.idxOffset + VkDeviceSize(v) * e.idxStride, 4u);
        uint32_t ubyte[4] = { word & 0xFFu, (word >> 8) & 0xFFu, (word >> 16) & 0xFFu, (word >> 24) & 0xFFu };
        uint32_t rows[4]  = { ubyte[0] * 3u, ubyte[1] * 3u, ubyte[2] * 3u, ubyte[3] * 3u };

        uint16_t wh[4];
        std::memcpy(wh, base + e.wtOffset + VkDeviceSize(v) * e.wtStride, sizeof(wh));
        float w[4] = { HalfToFloat(wh[0]), HalfToFloat(wh[1]), HalfToFloat(wh[2]), HalfToFloat(wh[3]) };

        float cpu[3];
        CpuSkin(e.bones, pos, rows, w, e.pivot, cpu);

        const float* gpu = &out[3u * (uint64_t(e.baseVertex) + v)];

        f << "  v" << (uint64_t(e.baseVertex) + v) << ": raw_pos=" << Vec(pos, 4)
          << " rows=(" << rows[0] << ", " << rows[1] << ", " << rows[2] << ", " << rows[3] << ")"
          << " weights=" << Vec(w, 4) << " wsum=" << (w[0] + w[1] + w[2] + w[3])
          << " out=" << Vec(gpu, 3) << " cpu=" << Vec(cpu, 3) << "\n";
      }
    }

    f.flush();
    Logger::info(str::format("blessed: skin trace: wrote ", path));
  }


  BlessedSkinTimer::BlessedSkinTimer(
          DxvkDevice*             device,
          std::atomic<uint64_t>*  nsSum,
          std::atomic<uint32_t>*  samples)
  : m_device(device), m_nsSum(nsSum), m_samples(samples) {
    if (!device->properties().core.properties.limits.timestampComputeAndGraphics)
      return;

    VkQueryPoolCreateInfo queryInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = PairCount * 2u;

    if (device->vkd()->vkCreateQueryPool(device->handle(), &queryInfo, nullptr, &m_pool) != VK_SUCCESS) {
      Logger::warn("blessed: skinning: failed to create timestamp query pool, skinned_gpu_ms disabled");
      m_pool = VK_NULL_HANDLE;
    }
  }


  BlessedSkinTimer::~BlessedSkinTimer() {
    if (m_pool != VK_NULL_HANDLE)
      m_device->vkd()->vkDestroyQueryPool(m_device->handle(), m_pool, nullptr);
  }


  uint32_t BlessedSkinTimer::begin(const Rc<DxvkCommandList>& cmd) {
    if (m_pool == VK_NULL_HANDLE)
      return UINT32_MAX;

    uint32_t pair = m_next;
    uint32_t base = pair * 2u;
    m_next = (m_next + 1u) % PairCount;

    if (m_used[pair])
      collect(base);

    cmd->cmdResetQueryPool(DxvkCmdBuffer::ExecBuffer, m_pool, base, 2u);
    cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_pool, base);

    m_used[pair] = true;
    return base;
  }


  void BlessedSkinTimer::end(const Rc<DxvkCommandList>& cmd, uint32_t base) {
    if (base == UINT32_MAX)
      return;

    cmd->cmdWriteTimestamp(DxvkCmdBuffer::ExecBuffer,
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_pool, base + 1u);
  }


  void BlessedSkinTimer::collect(uint32_t base) {
    struct { uint64_t ts; uint64_t avail; } results[2] = { };

    VkResult vr = m_device->vkd()->vkGetQueryPoolResults(
      m_device->handle(), m_pool, base, 2u,
      sizeof(results), results, sizeof(results[0]),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (vr != VK_SUCCESS || !results[0].avail || !results[1].avail)
      return;

    double period = double(m_device->properties().core.properties.limits.timestampPeriod);
    uint64_t ns = uint64_t(double(results[1].ts - results[0].ts) * period);

    m_nsSum->fetch_add(ns, std::memory_order_relaxed);
    m_samples->fetch_add(1u, std::memory_order_relaxed);
  }

}
