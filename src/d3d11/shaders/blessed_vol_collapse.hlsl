// blessed: vol-collapse, the volumetric z-integration chain (cs.1c4ebb62, one dispatch per slice) as one dispatch
//
// compiled by fxc into src/d3d11/blessed_vol_collapse_dxbc.h (see
// blessed-shaders/build-verify-dxbc.sh).
//
// vanilla dispatch k (k = k1 .. k1 + n - 1) reads src[k - 1] and src[k] and
// writes dst[k - 1] = src[k - 1], dst[k] = src[k - 1] + src[k], with src and
// dst swapping every dispatch. every thread touches only its own (x, y)
// column, so one thread per column can replay all n dispatches in order.
// the values go through the textures themselves (typed uav loads and
// stores), so every store rounds to the storage format exactly as the
// chain's own store does, and every later load sees what the chain's later
// dispatch would have read.

cbuffer params : register(b0) {
  uint g_k1;      // k of the chain's first dispatch
  uint g_count;   // dispatches in the chain
  uint2 g_pad;
};

RWTexture3D<float> g_first  : register(u0);  // src of the first dispatch
RWTexture3D<float> g_second : register(u1);  // dst of the first dispatch

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  for (uint m = 0; m < g_count; m++) {
    uint k = g_k1 + m;
    uint3 lo = uint3(id.xy, k - 1u);
    uint3 hi = uint3(id.xy, k);

    if ((m & 1u) == 0u) {
      float v0 = g_first[lo];
      float v1 = g_first[hi];
      g_second[lo] = v0;
      g_second[hi] = v0 + v1;
    } else {
      float v0 = g_second[lo];
      float v1 = g_second[hi];
      g_first[lo] = v0;
      g_first[hi] = v0 + v1;
    }
  }
}
