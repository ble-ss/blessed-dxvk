// blessed: cascade-cache -- per-mesh bounds registry and a draw's texel rectangle in a cascade layer (see blessed_cascade_bounds.h)
#include "blessed_cascade_bounds.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "d3d11_buffer.h"
#include "d3d11_input_layout.h"
#include "d3d11_shader.h"

#include "../dxvk/dxvk_device.h"
#include "../dxvk/blessed/blessed_cascade_cache.h"

namespace dxvk::BlessedCascadeBounds {

  namespace {

    constexpr uint32_t MaxMeshes         = 16384u;   // result slots, 64 bytes each
    constexpr uint32_t MaxRequestsFrame  = 512u;
    constexpr uint32_t MaxIndicesFrame   = 4u << 20;
    constexpr uint32_t ReadbackAge       = 8u;       // presents, past dxvk's frame latency
    constexpr uint32_t MaxScanVertices   = 65536u;
    constexpr uint32_t BoneSlot          = 10u;
    constexpr uint32_t GeometrySlot      = 2u;
    constexpr double   MarginTexels      = 2.0;

    enum class State : uint32_t { Queued, Pending, Ready, Failed };

    struct Mesh {
      State     state     = State::Queued;
      uint32_t  slot      = 0u;
      uint64_t  frame     = 0u;       // when the dispatch was emitted
      bool      hasPos    = false;
      float     min[3]    = { };
      float     max[3]    = { };
      uint32_t  minIndex  = 0u;
      uint32_t  maxIndex  = 0u;
      uint32_t  bones[3]  = { };
      BlessedCascadeBoundsRequest request;   // while queued

      // dynamic position streams: the cpu scan, once per frame
      uint64_t  scanFrame = ~0ull;
      bool      scanOk    = false;
      float     scanMin[3] = { };
      float     scanMax[3] = { };
    };

    struct Registry {
      std::unordered_map<uint64_t, Mesh> meshes;
      std::vector<uint64_t> queued;
      std::vector<uint64_t> pending;
      Rc<DxvkBuffer>        results;
      uint32_t              nextSlot = 0u;
      uint32_t              ready = 0u;
    };

    Registry& R() {
      static Registry* s_registry = new Registry();   // never destroyed: holds dxvk objects
      return *s_registry;
    }

    struct Hasher {
      uint64_t h = 0x9e3779b97f4a7c15ull;

      void u64(uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
      }
    };

    int32_t PosFormatCode(VkFormat fmt) {
      switch (fmt) {
        case VK_FORMAT_R32G32B32_SFLOAT:    return 0;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 1;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 2;
        default:                            return -1;
      }
    }

    float HalfToFloat(uint16_t h) {
      uint32_t sign = uint32_t(h & 0x8000u) << 16;
      uint32_t exp  = (h >> 10) & 0x1fu;
      uint32_t man  = h & 0x3ffu;
      uint32_t bits;

      if (exp == 0u) {
        if (man == 0u) {
          bits = sign;
        } else {
          // subnormal: renormalise
          exp = 127u - 15u + 1u;
          while (!(man & 0x400u)) { man <<= 1; exp--; }
          man &= 0x3ffu;
          bits = sign | (exp << 23) | (man << 13);
        }
      } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (man << 13);
      } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (man << 13);
      }

      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    }

    float OrderedToFloat(uint32_t o) {
      uint32_t u = (o & 0x80000000u) ? (o & 0x7fffffffu) : ~o;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    }

    // the bytes of a bound cbuffer the shader declares, or false
    bool CbBytes(const D3D11ShaderStageCbvBinding& cbv, uint32_t slot, const D3D11CommonShader* shader,
                 const uint8_t** data, uint32_t* size) {
      if (slot >= cbv.maxCount)
        return false;

      const auto& b = cbv.buffers[slot];
      D3D11Buffer* buffer = b.buffer.ptr();

      if (!buffer)
        return false;

      const uint8_t* base = reinterpret_cast<const uint8_t*>(buffer->GetMapPtr());

      if (!base)
        return false;

      uint32_t width  = buffer->Desc()->ByteWidth;
      uint32_t offset = b.constantOffset * 16u;

      if (offset >= width)
        return false;

      uint32_t bytes = width - offset;

      if (b.constantCount)
        bytes = std::min(bytes, b.constantCount * 16u);

      if (uint32_t declared = shader->BlessedCbvSize(slot))
        bytes = std::min(bytes, declared * 16u);

      *data = base + offset;
      *size = bytes;
      return true;
    }

    struct Box {
      double min[3] = {  1e300,  1e300,  1e300 };
      double max[3] = { -1e300, -1e300, -1e300 };

      bool empty() const { return min[0] > max[0]; }

      void add(const Box& o) {
        for (uint32_t i = 0; i < 3; i++) {
          min[i] = std::min(min[i], o.min[i]);
          max[i] = std::max(max[i], o.max[i]);
        }
      }
    };

    // Arvo: the aabb of a box under an affine 3x4 (rows as float4: xyz, w translation)
    Box Transform(const float* rows, const Box& b) {
      Box out;

      for (uint32_t r = 0; r < 3; r++) {
        const float* m = rows + 4 * r;
        double c = m[3], e = 0.0;

        for (uint32_t k = 0; k < 3; k++) {
          double center = 0.5 * (b.min[k] + b.max[k]);
          double extent = 0.5 * (b.max[k] - b.min[k]);
          c += double(m[k]) * center;
          e += std::abs(double(m[k])) * extent;
        }

        out.min[r] = c - e;
        out.max[r] = c + e;
      }

      return out;
    }

    Result Unknown(uint32_t* reason, uint32_t r) {
      *reason = r;
      return Result::Unknown;
    }

    // one position, format code as PosFormatCode
    void ReadPos(const uint8_t* p, int32_t format, float* out) {
      if (format == 2) {
        uint16_t h[3];
        std::memcpy(h, p, 6);
        out[0] = HalfToFloat(h[0]);
        out[1] = HalfToFloat(h[1]);
        out[2] = HalfToFloat(h[2]);
      } else {
        std::memcpy(out, p, 12);
      }
    }

  }


  const char* ReasonName(uint32_t r) {
    switch (r) {
      case ReasonNoPosition:  return "no_position";
      case ReasonNoBuffers:   return "no_buffers";
      case ReasonPending:     return "pending";
      case ReasonFailed:      return "failed";
      case ReasonNoTransform: return "no_transform";
      case ReasonScan:        return "scan";
      default:                return "none";
    }
  }


  Result DrawRect(
    const D3D11ContextState&  state,
          uint32_t            indexCount,
          uint32_t            startIndex,
          int32_t             baseVertex,
          uint64_t            frame,
    const double*             vp,
    const double*             posAdjust,
          uint32_t            width,
          uint32_t            height,
          Rect*               out,
          uint32_t*           reason) {
    *reason = ReasonNone;

    D3D11VertexShader* vs = state.vs.ptr();
    D3D11InputLayout* layout = state.ia.inputLayout.ptr();

    if (!vs || !layout || !layout->HasBlessedPosition())
      return Unknown(reason, ReasonNoPosition);

    const DxvkVertexAttribute& posAttr = layout->GetBlessedPosition();
    int32_t posFormat = PosFormatCode(posAttr.format);

    if (posFormat < 0 || posAttr.binding >= state.ia.vertexBuffers.size())
      return Unknown(reason, ReasonNoPosition);

    const auto& vbb = state.ia.vertexBuffers[posAttr.binding];
    const auto& ibb = state.ia.indexBuffer;
    D3D11Buffer* vb = vbb.buffer.ptr();
    D3D11Buffer* ib = ibb.buffer.ptr();

    if (!vb || !ib || ib->Desc()->Usage == D3D11_USAGE_DYNAMIC
     || (ibb.format != DXGI_FORMAT_R16_UINT && ibb.format != DXGI_FORMAT_R32_UINT))
      return Unknown(reason, ReasonNoBuffers);

    const D3D11CommonShader* shader = vs->GetCommonShader();
    uint32_t cbvMask = shader->GetBindingMask().cbvMask;
    bool skinned = layout->HasBlessedSkinning() && (cbvMask & (1u << BoneSlot));
    bool dynPos  = vb->Desc()->Usage == D3D11_USAGE_DYNAMIC;
    bool index32 = ibb.format == DXGI_FORMAT_R32_UINT;

    const DxvkVertexAttribute* idxAttr = skinned ? &layout->GetBlessedSkinIndices() : nullptr;
    const DxvkVertexAttribute* wtAttr  = skinned ? &layout->GetBlessedSkinWeights() : nullptr;
    const D3D11VertexBufferBinding* idxVb = nullptr;
    const D3D11VertexBufferBinding* wtVb  = nullptr;

    if (skinned) {
      if (idxAttr->binding >= state.ia.vertexBuffers.size() || wtAttr->binding >= state.ia.vertexBuffers.size())
        return Unknown(reason, ReasonNoBuffers);

      idxVb = &state.ia.vertexBuffers[idxAttr->binding];
      wtVb  = &state.ia.vertexBuffers[wtAttr->binding];

      if (!idxVb->buffer.ptr() || !wtVb->buffer.ptr()
       || idxVb->buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC
       || wtVb->buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC)
        return Unknown(reason, ReasonNoBuffers);
    }

    // ---- the mesh ----

    Hasher h;
    h.u64(ib->GetCookie());
    h.u64(uint64_t(ibb.offset) + uint64_t(startIndex) * (index32 ? 4u : 2u));
    h.u64(index32 ? 1u : 0u);
    h.u64(indexCount);
    h.u64(uint64_t(uint32_t(baseVertex)));

    if (!dynPos) {
      h.u64(vb->GetCookie());
      h.u64(vbb.offset);
      h.u64(vbb.stride);
      h.u64(posAttr.offset);
      h.u64(uint64_t(posFormat));
    } else {
      h.u64(0xd1ull);
    }

    if (skinned) {
      h.u64(idxVb->buffer->GetCookie());
      h.u64(idxVb->offset + idxAttr->offset);
      h.u64(idxVb->stride);
      h.u64(wtVb->buffer->GetCookie());
      h.u64(wtVb->offset + wtAttr->offset);
      h.u64(wtVb->stride);
    }

    Registry& reg = R();
    auto [it, inserted] = reg.meshes.try_emplace(h.h);
    Mesh& mesh = it->second;

    if (inserted) {
      if (reg.nextSlot >= MaxMeshes) {
        mesh.state = State::Failed;
        return Unknown(reason, ReasonFailed);
      }

      mesh.slot = reg.nextSlot++;

      BlessedCascadeBoundsRequest& r = mesh.request;
      r.ib         = ib->GetBuffer();
      r.ibOffset   = VkDeviceSize(ibb.offset) + VkDeviceSize(startIndex) * (index32 ? 4u : 2u);
      r.index32    = index32;
      r.indexCount = indexCount;
      r.baseVertex = baseVertex;

      if (!dynPos) {
        r.pos       = vb->GetBuffer();
        r.posOffset = VkDeviceSize(vbb.offset) + posAttr.offset;
        r.posStride = vbb.stride;
        r.posFormat = uint32_t(posFormat);
      }

      if (skinned) {
        r.idx       = idxVb->buffer->GetBuffer();
        r.idxOffset = VkDeviceSize(idxVb->offset) + idxAttr->offset;
        r.idxStride = idxVb->stride;
        r.wt        = wtVb->buffer->GetBuffer();
        r.wtOffset  = VkDeviceSize(wtVb->offset) + wtAttr->offset;
        r.wtStride  = wtVb->stride;
      }

      r.slot = mesh.slot;
      reg.queued.push_back(h.h);
      return Unknown(reason, ReasonPending);
    }

    if (mesh.state == State::Queued || mesh.state == State::Pending)
      return Unknown(reason, ReasonPending);

    if (mesh.state == State::Failed)
      return Unknown(reason, ReasonFailed);

    // ---- the object-space box ----

    Box obj;

    if (!dynPos) {
      if (!mesh.hasPos)
        return Unknown(reason, ReasonFailed);

      for (uint32_t i = 0; i < 3; i++) {
        obj.min[i] = mesh.min[i];
        obj.max[i] = mesh.max[i];
      }
    } else {
      if (mesh.scanFrame != frame) {
        mesh.scanFrame = frame;
        mesh.scanOk = false;

        const uint8_t* base = reinterpret_cast<const uint8_t*>(vb->GetMapPtr());
        int64_t first = int64_t(baseVertex) + int64_t(mesh.minIndex);
        int64_t last  = int64_t(baseVertex) + int64_t(mesh.maxIndex);
        uint32_t posSize = posFormat == 2 ? 6u : 12u;

        uint64_t endByte = uint64_t(vbb.offset) + posAttr.offset + uint64_t(std::max<int64_t>(last, 0)) * vbb.stride + posSize;

        if (base && first >= 0 && last >= first && uint64_t(last - first) < MaxScanVertices
         && endByte <= vb->Desc()->ByteWidth) {
          float mn[3] = {  1e30f,  1e30f,  1e30f };
          float mx[3] = { -1e30f, -1e30f, -1e30f };

          for (int64_t v = first; v <= last; v++) {
            float p[3];
            ReadPos(base + vbb.offset + posAttr.offset + uint64_t(v) * vbb.stride, posFormat, p);

            for (uint32_t i = 0; i < 3; i++) {
              mn[i] = std::min(mn[i], p[i]);
              mx[i] = std::max(mx[i], p[i]);
            }
          }

          bool finite = true;

          for (uint32_t i = 0; i < 3; i++)
            finite &= std::isfinite(mn[i]) && std::isfinite(mx[i]);

          if (finite) {
            std::memcpy(mesh.scanMin, mn, sizeof(mn));
            std::memcpy(mesh.scanMax, mx, sizeof(mx));
            mesh.scanOk = true;
          }
        }
      }

      if (!mesh.scanOk)
        return Unknown(reason, ReasonScan);

      for (uint32_t i = 0; i < 3; i++) {
        obj.min[i] = mesh.scanMin[i];
        obj.max[i] = mesh.scanMax[i];
      }
    }

    // ---- to cascade-relative space ----

    Box rel;
    const auto& vsCbv = state.cbv[D3D11ShaderType::eVertex];

    if (skinned) {
      const uint8_t* bones = nullptr;
      uint32_t size = 0u;

      if (!CbBytes(vsCbv, BoneSlot, shader, &bones, &size))
        return Unknown(reason, ReasonNoTransform);

      bool any = false;

      for (uint32_t b = 0; b < 80u; b++) {
        if (!(mesh.bones[b >> 5] & (1u << (b & 31u))))
          continue;

        if ((b + 1u) * 48u > size)
          return Unknown(reason, ReasonNoTransform);

        float rows[12];
        std::memcpy(rows, bones + b * 48u, sizeof(rows));
        rel.add(Transform(rows, obj));
        any = true;
      }

      if (!any)
        return Unknown(reason, ReasonFailed);

      // bone translations are absolute; the vs subtracts CameraPosAdjust
      for (uint32_t i = 0; i < 3; i++) {
        rel.min[i] -= posAdjust[i];
        rel.max[i] -= posAdjust[i];
      }
    } else if (cbvMask & (1u << GeometrySlot)) {
      const uint8_t* geom = nullptr;
      uint32_t size = 0u;

      // Utility.hlsl PerGeometry: World rows at c1-c3 (translation already
      // relative to CameraPosAdjust), TreeParams at c7
      if (!CbBytes(vsCbv, GeometrySlot, shader, &geom, &size) || size < 64u)
        return Unknown(reason, ReasonNoTransform);

      // TreeParams only when the vs reads c7: an unread register holds
      // whatever the ring memory held before (stale bytes)
      bool readsTree = shader->BlessedCbvWhole(GeometrySlot)
        ? size >= 128u
        : (shader->BlessedCbvRegs(GeometrySlot) & (1ull << 7)) && size >= 128u;

      if (readsTree) {
        float treeZ;
        std::memcpy(&treeZ, geom + 120u, 4);
        double grow = 1.1 * std::abs(double(treeZ)) * 1.7320508075688772 + 0.01;

        for (uint32_t i = 0; i < 3; i++) {
          obj.min[i] -= grow;
          obj.max[i] += grow;
        }
      }

      float rows[12];
      std::memcpy(rows, geom + 16u, sizeof(rows));
      rel = Transform(rows, obj);
    } else {
      return Unknown(reason, ReasonNoTransform);
    }

    for (uint32_t i = 0; i < 3; i++) {
      if (!std::isfinite(rel.min[i]) || !std::isfinite(rel.max[i]))
        return Unknown(reason, ReasonFailed);
    }

    // ---- to texels ----

    double tx0 = 1e300, ty0 = 1e300, tx1 = -1e300, ty1 = -1e300;

    for (uint32_t c = 0; c < 8; c++) {
      double p[3] = {
        (c & 1) ? rel.max[0] : rel.min[0],
        (c & 2) ? rel.max[1] : rel.min[1],
        (c & 4) ? rel.max[2] : rel.min[2] };

      double clip[4];

      for (uint32_t r = 0; r < 4; r++)
        clip[r] = vp[4 * r + 0] * p[0] + vp[4 * r + 1] * p[1] + vp[4 * r + 2] * p[2] + vp[4 * r + 3];

      // the sun cascades are orthographic; anything else is not ours to bound
      if (std::abs(clip[3] - 1.0) > 1e-3)
        return Unknown(reason, ReasonNoTransform);

      double tx = (clip[0] * 0.5 + 0.5) * double(width);
      double ty = (0.5 - clip[1] * 0.5) * double(height);

      tx0 = std::min(tx0, tx); tx1 = std::max(tx1, tx);
      ty0 = std::min(ty0, ty); ty1 = std::max(ty1, ty);
    }

    tx0 -= MarginTexels; ty0 -= MarginTexels;
    tx1 += MarginTexels; ty1 += MarginTexels;

    if (tx1 <= 0.0 || ty1 <= 0.0 || tx0 >= double(width) || ty0 >= double(height))
      return Result::Outside;

    out->x0 = int32_t(std::max(0.0, std::floor(tx0)));
    out->y0 = int32_t(std::max(0.0, std::floor(ty0)));
    out->x1 = int32_t(std::min(double(width),  std::ceil(tx1)));
    out->y1 = int32_t(std::min(double(height), std::ceil(ty1)));
    return Result::Rect;
  }


  bool TakeRequests(
          DxvkDevice*               device,
          BlessedCascadeBoundsArgs* args,
          uint64_t                  frame) {
    Registry& reg = R();

    if (reg.queued.empty())
      return false;

    if (reg.results == nullptr) {
      DxvkBufferCreateInfo info = { };
      info.size      = VkDeviceSize(MaxMeshes) * sizeof(BlessedCascadeBoundsResult);
      info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      info.access    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      info.debugName = "blessed cascade bounds results";

      reg.results = device->createBuffer(info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    auto* slots = reinterpret_cast<BlessedCascadeBoundsResult*>(reg.results->getSliceInfo().mapPtr);

    args->results = reg.results;
    args->requests.clear();

    uint32_t indices = 0u;
    size_t taken = 0u;

    for (; taken < reg.queued.size() && args->requests.size() < MaxRequestsFrame; taken++) {
      Mesh& mesh = reg.meshes[reg.queued[taken]];

      if (indices + mesh.request.indexCount > MaxIndicesFrame && !args->requests.empty())
        break;

      BlessedCascadeBoundsResult& s = slots[mesh.slot];
      std::memset(&s, 0, sizeof(s));

      for (uint32_t i = 0; i < 3; i++)
        s.minPos[i] = ~0u;

      s.minIndex = ~0u;

      indices += mesh.request.indexCount;
      args->requests.push_back(std::move(mesh.request));
      mesh.request = BlessedCascadeBoundsRequest();
      mesh.state = State::Pending;
      mesh.frame = frame;
      reg.pending.push_back(reg.queued[taken]);
    }

    reg.queued.erase(reg.queued.begin(), reg.queued.begin() + taken);
    return !args->requests.empty();
  }


  void ReadBack(uint64_t frame) {
    Registry& reg = R();

    if (reg.pending.empty() || reg.results == nullptr)
      return;

    auto* slots = reinterpret_cast<const BlessedCascadeBoundsResult*>(reg.results->getSliceInfo().mapPtr);
    size_t kept = 0u;

    for (size_t i = 0; i < reg.pending.size(); i++) {
      Mesh& mesh = reg.meshes[reg.pending[i]];

      if (frame < mesh.frame + ReadbackAge) {
        reg.pending[kept++] = reg.pending[i];
        continue;
      }

      BlessedCascadeBoundsResult s;
      std::memcpy(&s, &slots[mesh.slot], sizeof(s));

      if (s.minIndex > s.maxIndex) {
        mesh.state = State::Failed;
        continue;
      }

      mesh.minIndex = s.minIndex;
      mesh.maxIndex = s.maxIndex;
      std::memcpy(mesh.bones, s.boneMask, sizeof(mesh.bones));
      mesh.hasPos = true;

      for (uint32_t k = 0; k < 3; k++) {
        mesh.min[k] = OrderedToFloat(s.minPos[k]);
        mesh.max[k] = OrderedToFloat(s.maxPos[k]);
        mesh.hasPos &= s.minPos[k] != ~0u && std::isfinite(mesh.min[k]) && std::isfinite(mesh.max[k])
                    && mesh.min[k] <= mesh.max[k];
      }

      mesh.state = State::Ready;
      reg.ready++;
    }

    reg.pending.resize(kept);
  }


  void Counts(uint32_t* known, uint32_t* ready, uint32_t* pending) {
    Registry& reg = R();
    *known   = uint32_t(reg.meshes.size());
    *ready   = reg.ready;
    *pending = uint32_t(reg.pending.size() + reg.queued.size());
  }

}
