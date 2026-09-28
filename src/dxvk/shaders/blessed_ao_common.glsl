// blessed: BLESSED_AO=rt -- constants, push data and helpers shared by the three ao passes

// blessed: global sampler heap, fixed set/binding convention (see
// dxvk_mipgen.comp, DxvkPipelineLayoutFlag::UsesSamplerHeap)
layout(set = 0, binding = 0)
uniform sampler s_samplers[];

// blessed: matches BlessedAoConstants in blessed_ao.cpp byte for byte
// (scalar layout, no vec3 padding)
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer BlessedAoConstants {
  mat4  invViewProj;     // CameraViewProjInverse, raw row-major bytes (so v * m below)
  mat4  prevViewProj;    // the history's view-projection, same convention
  vec3  camDelta;        // camPosAdjust(now) - camPosAdjust(tlas build)
  vec3  reprojDelta;     // camPosAdjust(now) - camPosAdjust(history)
  float farDepthValue;
  float radius;
  float strength;
  uint  rays;
  uint  frameIndex;
  uint  historyValid;
  uint  fullWidth;
  uint  fullHeight;
  uint  traceWidth;
  uint  traceHeight;
  uint  traceScale;      // 1 or 2
  uint  debugMode;       // BlessedAoDebug
};

layout(push_constant, scalar) uniform BlessedAoPush {
  uint64_t constantsAddress;
  uint64_t tlasAddress;
  uint     samplerIndex;
  uint     tlasValid;
  uint     passArg;       // blur: 0 horizontal (reads the accumulation), 1 vertical
} pc;

// blessed: BlessedAoDebug in blessed_ao.h
const uint BLESSED_AO_DEBUG_NONE   = 0u;
const uint BLESSED_AO_DEBUG_BLACK  = 1u;
const uint BLESSED_AO_DEBUG_WHITE  = 2u;
const uint BLESSED_AO_DEBUG_RAW    = 3u;
const uint BLESSED_AO_DEBUG_NOBLUR = 4u;
const uint BLESSED_AO_DEBUG_HIST   = 5u;
const uint BLESSED_AO_DEBUG_NORMAL = 6u;
const uint BLESSED_AO_DEBUG_REPROJ = 7u;

// blessed: camera-relative position of full-res pixel c at depth d. d3d
// viewport conventions (ndc.y = 1 - 2v); row-major cbuffer bytes read into
// a column-major mat4 land transposed, so skyrim's mul(M, v) is v * m
vec3 unproject(mat4 invViewProj, ivec2 c, float d, vec2 size) {
  vec2 uv = (vec2(c) + 0.5) / size;
  vec4 h = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0) * invViewProj;
  return h.xyz / h.w;
}

// blessed: per-pixel, per-frame hash -> [0,1)
float blessedAoHash(uvec2 p, uint salt) {
  uint h = p.x * 374761393u + p.y * 668265263u + salt * 2654435761u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h ^= h >> 16u;
  return float(h) * (1.0 / 4294967296.0);
}

// blessed: the accumulation's x packs the history length with the ao
float unpackAo(float x) {
  return x - 2.0 * floor(x * 0.5);
}
