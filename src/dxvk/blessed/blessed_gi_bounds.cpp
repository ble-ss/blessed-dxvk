// blessed: gi-bounds -- per-mesh object-space aabbs and the bounds-wide probe sample (BLESSED_GI_SAMPLE=bounds)
#include <algorithm>
#include <cmath>
#include <cstring>

#include "blessed_gi.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"

#include "../../util/log/log.h"

#include <blessed_gi_bounds.h>

namespace dxvk {

  namespace {

    // blessed: push data layout must match BlessedGiBoundsPushData in
    // shaders/blessed_gi_bounds.comp exactly (scalar layout).
    struct BlessedGiBoundsPushData {
      uint64_t vbAddress;
      uint64_t ibAddress;
      uint64_t resultAddress;
      uint32_t vbStride;
      uint32_t vbFormatCode;
      uint32_t indexTypeCode;
      uint32_t indexCount;
      uint32_t startIndex;
      int32_t  baseVertex;
      uint32_t vertexCount;
      uint32_t serial;
    };

    // one result slot: min xyz, pad, max xyz, serial
    constexpr uint32_t BoundsSlotFloats = 8u;

    // an openness this low still counts a little, so a space where every
    // probe is closed (a real interior) averages evenly instead of failing
    constexpr float OpennessFloor = 0.02f;

    // same as blessed_gi.cpp's FloorToInt (hook-cpu-2)
    inline int32_t FloorToIntB(float x) {
      int32_t i = int32_t(x);
      if (float(i) > x && i != INT32_MIN)
        i--;
      return i;
    }

    // position element size for the formats a gi record can carry, 0 otherwise
    uint32_t PositionElementSize(VkFormat fmt) {
      switch (fmt) {
        case VK_FORMAT_R32G32B32_SFLOAT:    return 12u;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16u;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8u;
        default:                            return 0u;
      }
    }

  }


  size_t BlessedGiState::BoundsKeyHash::operator () (const BoundsKey& k) const {
    size_t h = std::hash<uint64_t>()(k.vbCookie);
    auto mix = [&h] (size_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
    mix(std::hash<uint64_t>()(k.ibCookie));
    mix(k.vbOffset);
    mix(k.vbStride);
    mix(size_t(k.vbFormat));
    mix(k.ibOffset);
    mix(size_t(k.indexType));
    mix(k.indexCount);
    mix(k.startIndex);
    mix(size_t(uint32_t(k.baseVertex)));
    return h;
  }


  const float* BlessedGiState::lookupMeshBounds(const BlessedGiBoundsRequest& req, bool* pSettled) {
    bool settledDummy;
    bool& settled = pSettled ? *pSettled : settledDummy;
    settled = true;

    if (!req.vb || !req.ib)
      return nullptr;

    BoundsKey key;
    key.vbCookie   = req.vb->cookie();
    key.ibCookie   = req.ib->cookie();
    key.vbOffset   = uint32_t(req.vbOffset);
    key.vbStride   = req.vbStride;
    // same collapse as BlessedScene's CacheKey: only xyz is ever read
    key.vbFormat   = req.vbFormat == VK_FORMAT_R32G32B32A32_SFLOAT ? VK_FORMAT_R32G32B32_SFLOAT : req.vbFormat;
    key.ibOffset   = uint32_t(req.ibOffset);
    key.indexType  = req.indexType;
    key.indexCount = req.indexCount;
    key.startIndex = req.startIndex;
    key.baseVertex = req.baseVertex;

    auto it = m_bounds.find(key);

    if (it != m_bounds.end()) {
      it->second.lastUsed = m_csCacheGeneration;
      settled = it->second.state == BoundsState::Known || it->second.state == BoundsState::Failed;
      return it->second.state == BoundsState::Known ? it->second.bounds : nullptr;
    }

    // miss: queue it unless the queue is full (then it is simply asked
    // for again by a later draw)
    if (m_boundsJobs.size() >= MaxBoundsQueued) {
      settled = false;
      return nullptr;
    }

    BoundsEntry entry;
    entry.lastUsed = m_csCacheGeneration;

    // what the gpu may read has to be static and in range; anything else
    // is remembered as failed and keeps the origin sample
    uint32_t     elemSize    = PositionElementSize(req.vbFormat);
    VkDeviceSize vbWidth     = req.vb->info().size;
    VkDeviceSize ibWidth     = req.ib->info().size;
    VkDeviceSize idxSize     = req.indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u;
    uint32_t     vertexCount = 0u;

    if (elemSize && req.vbStride && req.vbOffset + elemSize <= vbWidth)
      vertexCount = uint32_t((vbWidth - req.vbOffset - elemSize) / req.vbStride + 1u);

    bool ok = !req.vb->info().blessedDynamic && !req.ib->info().blessedDynamic
      && vertexCount && req.indexCount && req.baseVertex >= 0
      && req.ibOffset + (VkDeviceSize(req.startIndex) + req.indexCount) * idxSize <= ibWidth;

    if (!ok) {
      entry.state = BoundsState::Failed;
      m_bounds.emplace(key, entry);
      return nullptr;
    }

    entry.state = BoundsState::Queued;
    m_bounds.emplace(key, entry);
    settled = false;

    BoundsJob job;
    job.key         = key;
    job.vb          = req.vb;
    job.ib          = req.ib;
    job.vbOffset    = req.vbOffset;
    job.ibOffset    = req.ibOffset;
    job.vertexCount = vertexCount;
    m_boundsJobs.push_back(std::move(job));
    return nullptr;
  }


  void BlessedGiState::dispatchBoundsJobs(DxvkContext* ctx, const Rc<DxvkCommandList>& cmd) {
    if (m_boundsResults == nullptr) {
      DxvkBufferCreateInfo info = { };
      info.size      = VkDeviceSize(MaxBoundsSlots) * BoundsSlotFloats * sizeof(float);
      info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      info.access    = VK_ACCESS_2_SHADER_WRITE_BIT;
      info.debugName = "blessed gi mesh bounds";

      // blessed: host-cached, same reasoning as the probe readback ring
      m_boundsResults = m_device->createBuffer(info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
      ctx->ensureBufferAddress(m_boundsResults);
      std::memset(m_boundsResults->getSliceInfo().mapPtr, 0, size_t(info.size));

      m_boundsFreeSlots.reserve(MaxBoundsSlots);
      for (uint32_t i = MaxBoundsSlots; i > 0u; i--)
        m_boundsFreeSlots.push_back(i - 1u);

      m_boundsLayout = m_device->createBuiltInPipelineLayout(
        DxvkPipelineLayoutFlags(), VK_SHADER_STAGE_COMPUTE_BIT,
        sizeof(BlessedGiBoundsPushData), 0u, nullptr);

      util::DxvkBuiltInShaderStage stage(blessed_gi_bounds, nullptr);
      m_boundsPipeline = m_device->createBuiltInComputePipeline(m_boundsLayout, stage);
    }

    size_t count = std::min<size_t>({ m_boundsJobs.size(), size_t(MaxBoundsPerFrame), m_boundsFreeSlots.size() });

    if (!count)
      return;

    // stable gpu addresses first, before anything is recorded
    for (size_t i = 0; i < count; i++) {
      ctx->ensureBufferAddress(m_boundsJobs[i].vb);
      ctx->ensureBufferAddress(m_boundsJobs[i].ib);
    }

    auto* results = reinterpret_cast<uint8_t*>(m_boundsResults->getSliceInfo().mapPtr);
    VkDeviceAddress resultsAddress = m_boundsResults->getSliceInfo().gpuAddress;

    cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_boundsPipeline);

    for (size_t i = 0; i < count; i++) {
      BoundsJob& job = m_boundsJobs[i];

      auto it = m_bounds.find(job.key);
      if (it == m_bounds.end())
        continue; // cannot happen: queued entries are never evicted

      uint32_t slot = m_boundsFreeSlots.back();
      m_boundsFreeSlots.pop_back();

      if (++m_boundsSerial == 0u)
        m_boundsSerial = 1u;

      // the slot is free, so the gpu is done with it: clear its serial so
      // a stale one can never match
      size_t slotBytes = size_t(slot) * BoundsSlotFloats * sizeof(float);
      std::memset(results + slotBytes, 0, BoundsSlotFloats * sizeof(float));

      BlessedGiBoundsPushData push = { };
      push.vbAddress     = job.vb->getSliceInfo().gpuAddress + job.vbOffset;
      push.ibAddress     = job.ib->getSliceInfo().gpuAddress + job.ibOffset;
      push.resultAddress = resultsAddress + slotBytes;
      push.vbStride      = job.key.vbStride;
      push.vbFormatCode  = job.key.vbFormat == VK_FORMAT_R16G16B16A16_SFLOAT ? 2u : 0u;
      push.indexTypeCode = job.key.indexType == VK_INDEX_TYPE_UINT16 ? 0u : 1u;
      push.indexCount    = job.key.indexCount;
      push.startIndex    = job.key.startIndex;
      push.baseVertex    = job.key.baseVertex;
      push.vertexCount   = job.vertexCount;
      push.serial        = m_boundsSerial;

      cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_boundsLayout,
        0u, nullptr, sizeof(push), &push);
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, 1u, 1u, 1u);

      cmd->track(job.vb, DxvkAccess::Read);
      cmd->track(job.ib, DxvkAccess::Read);

      BoundsEntry& entry = it->second;
      entry.state        = BoundsState::InFlight;
      entry.slot         = slot;
      entry.serial       = m_boundsSerial;
      entry.dispatchedAt = m_traceCount;
      m_boundsInFlight.push_back(job.key);
    }

    cmd->track(m_boundsResults, DxvkAccess::Write);
    m_boundsJobs.erase(m_boundsJobs.begin(), m_boundsJobs.begin() + count);
  }


  void BlessedGiState::pollBounds() {
    if (m_boundsResults != nullptr && !m_boundsInFlight.empty()) {
      auto* results = reinterpret_cast<const float*>(m_boundsResults->getSliceInfo().mapPtr);

      for (size_t i = 0; i < m_boundsInFlight.size(); ) {
        auto it = m_bounds.find(m_boundsInFlight[i]);

        if (it == m_bounds.end()) {
          m_boundsInFlight[i] = m_boundsInFlight.back();
          m_boundsInFlight.pop_back();
          continue;
        }

        BoundsEntry& entry = it->second;
        uint64_t age = m_traceCount - entry.dispatchedAt;

        // same fence-free margin as the probe ring: this many later traces
        // submitted before the slot is trusted, and its serial must match
        if (age <= BoundsReadbackFrames) {
          i++;
          continue;
        }

        const float* slot = results + size_t(entry.slot) * BoundsSlotFloats;
        uint32_t serial;
        std::memcpy(&serial, slot + 7, sizeof(serial));

        if (serial == entry.serial) {
          float b[6] = { slot[0], slot[1], slot[2], slot[4], slot[5], slot[6] };
          bool ok = true;

          for (uint32_t a = 0; a < 3; a++) {
            ok &= std::isfinite(b[a]) && std::isfinite(b[a + 3]) && b[a] <= b[a + 3]
               && b[a + 3] - b[a] < 1.0e6f;
          }

          if (ok) {
            std::memcpy(entry.bounds, b, sizeof(b));
            entry.state = BoundsState::Known;
            m_meshesWithBounds.fetch_add(1u, std::memory_order_relaxed);
          } else {
            entry.state = BoundsState::Failed;
          }
        } else if (age > BoundsGiveUpFrames) {
          entry.state = BoundsState::Failed;
        } else {
          i++;
          continue;
        }

        m_boundsFreeSlots.push_back(entry.slot);
        m_boundsInFlight[i] = m_boundsInFlight.back();
        m_boundsInFlight.pop_back();
      }
    }

    // forget meshes nothing has drawn in a while; queued and in-flight
    // entries stay (the job list and the in-flight list refer to them)
    if (++m_refreshesSinceEvict >= BoundsEvictInterval) {
      m_refreshesSinceEvict = 0;

      for (auto it = m_bounds.begin(); it != m_bounds.end(); ) {
        const BoundsEntry& e = it->second;
        bool settled = e.state == BoundsState::Known || e.state == BoundsState::Failed;

        if (settled && m_csCacheGeneration - e.lastUsed > BoundsEvictAfter) {
          if (e.state == BoundsState::Known)
            m_meshesWithBounds.fetch_sub(1u, std::memory_order_relaxed);
          it = m_bounds.erase(it);
        } else {
          ++it;
        }
      }
    }
  }


  float BlessedGiState::meanOpennessNow() const {
    if (!m_csCacheValid || m_probeStride <= ProbeStride)
      return 0.0f;

    size_t total = size_t(m_csCacheDims[0]) * size_t(m_csCacheDims[1]) * size_t(m_csCacheDims[2]);
    if (!total)
      return 0.0f;

    double sum = 0.0;
    for (size_t i = 0; i < total; i++)
      sum += m_csCache[i * m_probeStride + ProbeStride];

    return float(sum / double(total));
  }


  bool BlessedGiState::sampleAmbientBounds(const float world3x4[12], const float camPosAbs[3],
    const float bmin[3], const float bmax[3], float outRow[12]) const {
    if (!m_csCacheValid)
      return false;

    // blessed: the points' trilinear weights are gathered per grid cell
    // first, so a small object whose nine points share a few cells pays
    // the 12-float accumulation once per cell, not once per point. Same
    // result as averaging nine separately sampled rows.
    struct CellAcc {
      int32_t  key[3];
      uint32_t offset[8];
      float    weight[8];
    };

    CellAcc cells[9];
    uint32_t cellCount = 0;
    float    totalPointWeight = 0.0f;

    const float s = m_openness;
    const bool  hasOpenness = m_probeStride > ProbeStride;

    float centre[3], half[3];
    for (int a = 0; a < 3; a++) {
      centre[a] = 0.5f * (bmin[a] + bmax[a]);
      half[a]   = 0.5f * (bmax[a] - bmin[a]) * (1.0f - m_boundsPull);
    }

    for (uint32_t p = 0; p < 9; p++) {
      // point 0 is the centre, 1..8 the pulled-in corners
      float local[3];
      for (int a = 0; a < 3; a++) {
        float sign = p == 0 ? 0.0f : (((p - 1u) >> a) & 1u ? 1.0f : -1.0f);
        local[a] = centre[a] + sign * half[a];
      }

      float base[3];
      int32_t c0[3];
      float frac[3];

      for (int r = 0; r < 3; r++) {
        const float* row = &world3x4[4 * r];
        float world = row[0] * local[0] + row[1] * local[1] + row[2] * local[2] + row[3] + camPosAbs[r];
        base[r] = world / m_csCacheSpacing - 0.5f;
        c0[r]   = FloorToIntB(base[r]);
        frac[r] = base[r] - float(c0[r]);
      }

      // copy out what the cell cache holds: a later lookup may reuse the slot
      const CsCell& cell = lookupCsCell(c0);
      uint8_t  validMask = cell.validMask;

      CellAcc* acc = nullptr;
      for (uint32_t c = 0; c < cellCount; c++) {
        if (cells[c].key[0] == c0[0] && cells[c].key[1] == c0[1] && cells[c].key[2] == c0[2]) {
          acc = &cells[c];
          break;
        }
      }

      float cornerWeight[8];
      float weightSum = 0.0f;
      float triSum    = 0.0f;
      float openSum   = 0.0f;

      for (uint32_t corner = 0; corner < 8; corner++) {
        cornerWeight[corner] = 0.0f;

        if (!(validMask & (1u << corner)))
          continue;

        float wx = (corner & 1u) ? frac[0] : (1.0f - frac[0]);
        float wy = (corner & 2u) ? frac[1] : (1.0f - frac[1]);
        float wz = (corner & 4u) ? frac[2] : (1.0f - frac[2]);
        float tri = wx * wy * wz;

        if (tri <= 0.0f)
          continue;

        float open = hasOpenness ? m_csCache[cell.offset[corner] + ProbeStride] : 1.0f;
        float w = tri * (1.0f - s + s * std::max(open, OpennessFloor));

        cornerWeight[corner] = w;
        weightSum += w;
        triSum    += tri;
        openSum   += tri * open;
      }

      if (weightSum <= 0.0f)
        continue; // no valid probe around this point

      if (!acc) {
        acc = &cells[cellCount++];
        for (int a = 0; a < 3; a++)
          acc->key[a] = c0[a];
        for (uint32_t corner = 0; corner < 8; corner++) {
          acc->offset[corner] = cell.offset[corner];
          acc->weight[corner] = 0.0f;
        }
      }

      float pointOpen   = openSum / triSum;
      float pointWeight = 1.0f - s + s * std::max(pointOpen, OpennessFloor);
      float scale       = pointWeight / weightSum;

      for (uint32_t corner = 0; corner < 8; corner++)
        acc->weight[corner] += cornerWeight[corner] * scale;

      totalPointWeight += pointWeight;
    }

    if (totalPointWeight <= 0.0f)
      return false;

    float row[12] = { };

    for (uint32_t c = 0; c < cellCount; c++) {
      for (uint32_t corner = 0; corner < 8; corner++) {
        float w = cells[c].weight[corner];
        if (w <= 0.0f)
          continue;

        const float* sh = m_csCache.data() + cells[c].offset[corner];
        for (int i = 0; i < 12; i++)
          row[i] += sh[i] * w;
      }
    }

    float inv = 1.0f / totalPointWeight;
    for (int i = 0; i < 12; i++)
      outRow[i] = row[i] * inv;

    return true;
  }

}
