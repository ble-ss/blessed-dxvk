// blessed: gi v1 -- per-tlas-instance geometry + albedo record, shared between
// blessed_scene.cpp (writer, one entry per instance pushed to the tlas build,
// same order/index) and blessed_gi_trace.comp (reader, indexed by
// rayQueryGetIntersectionInstanceIdEXT -- the hardware instance array index,
// NOT instanceCustomIndex). Layout must match BlessedGiInstanceInfo in
// blessed_gi_trace.comp exactly (scalar layout, tightly packed).
#pragma once

#include <cstdint>

namespace dxvk {

  // vbFormatCode / indexTypeCode: small enums so the shader can branch on a
  // plain uint rather than carrying VkFormat/VkIndexType constants across
  // the host/device boundary.
  enum class BlessedGiVertexFormat : uint32_t {
    Rgb32f  = 0, // VK_FORMAT_R32G32B32_SFLOAT
    Rgba32f = 1, // VK_FORMAT_R32G32B32A32_SFLOAT
    Rgba16f = 2, // VK_FORMAT_R16G16B16A16_SFLOAT
  };

  struct BlessedGiInstanceInfo {
    uint64_t vbAddress            = 0;  // vb base address, vbOffset already folded in -- 4-byte
                                         // aligned already (the same address the blas build's own
                                         // VkAccelerationStructureGeometryTrianglesDataKHR uses)
    // blessed: rounded DOWN to a 4-byte boundary (a uint16 index buffer's ib
    // offset is only guaranteed 2-byte aligned -- see BlessedSceneCaptureDraw's
    // own alignment check -- so this address alone may not be safe to read
    // as a buffer_reference-of-uint declared 4-byte aligned). Add
    // ibByteShift (0 or 2) to get the true byte offset of index 0 from this
    // rounded base -- see blessed_gi_trace.comp's fetchIndex.
    uint64_t ibAddress            = 0;
    uint32_t vbStride             = 0;
    uint32_t vbFormatCode         = 0;  // BlessedGiVertexFormat
    uint32_t indexTypeCode        = 0;  // 0 = uint16, 1 = uint32
    uint32_t primitiveOffsetBytes = 0;  // key.startIndex * index element size
    int32_t  baseVertex           = 0;  // key.baseVertex (== rangeInfo.firstVertex at build time)
    uint32_t albedoId             = 0;  // index into the albedo table; 0 = unknown/default
    uint32_t ibByteShift          = 0;  // 0 or 2 -- see ibAddress's own comment above
  };

}
