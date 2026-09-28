// blessed: shader-replace, cs.cf1e1a21 (volumetric lighting horizontal edge-aware blur) in 480-thread groups
//
// vanilla: 990-thread groups (31 warps; one group per ampere sm, 65%
// occupancy, and the whole sm idles at the barrier), each thread one fetch
// pair and one output, 15-texel aprons. here: the same 960-texel tile per
// group (the game's dispatch count is unchanged), 480 threads, each loads
// up to 3 apron/tile texels and writes 2 outputs. math is vanilla's, same
// constants and the same operation order.

cbuffer cb0 : register(b0) {
  float4 c0;  // xy: texel size
  float4 c1;  // xy: uv clamp
};

SamplerState s0 : register(s0);
SamplerState s1 : register(s1);
Texture2D<float4> t0 : register(t0);  // x: the value blurred
Texture2D<float4> t1 : register(t1);  // x: the edge key (depth)
RWTexture2D<float4> u0 : register(u0);

#define TILE    960
#define APRON   15
#define LOADS   (TILE + 2 * APRON)
#ifndef THREADS
#define THREADS 480
#endif
#define LOAD_STEPS ((LOADS + THREADS - 1) / THREADS)
#define OUT_STEPS  ((TILE + THREADS - 1) / THREADS)

groupshared float g0[LOADS];
groupshared float g1[LOADS];

[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tig : SV_GroupThreadID) {
  [unroll]
  for (uint k = 0; k < LOAD_STEPS; k++) {
    uint j = tig.x + k * THREADS;

    if (j < LOADS) {
      int px = int(gid.x) * TILE + int(j) - APRON;
      float2 uv = min((float2(px, int(gid.y)) + 0.5f) * c0.xy, c1.xy);
      g0[j] = t0.SampleLevel(s0, uv, 0.0f).x;
      g1[j] = t1.SampleLevel(s1, uv, 0.0f).x;
    }
  }

  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint m = 0; m < OUT_STEPS; m++) {
    uint i = tig.x + m * THREADS;

    if (i < TILE) {
      uint j = i + APRON;
      float a = g0[j];
      float b = g1[j];

      float e = b * 5.0f - g1[j - 12];
      e = e - g1[j - 6];
      e = e - b;
      e = e - g1[j + 6];
      e = e - g1[j + 12];

      float o = a;

      if (!(asfloat(0x3b03126fu) < abs(e))) {
        float s = g0[j - 6] * asfloat(0x3e577b39u);
        s = g0[j - 12] * asfloat(0x3e36ae7du) + s;
        s = a * asfloat(0x3e63ac93u) + s;
        s = g0[j + 6] * asfloat(0x3e577b39u) + s;
        o = g0[j + 12] * asfloat(0x3e36ae7du) + s;
      }

      u0[uint2(gid.x * TILE + i, gid.y)] = o.xxxx;
    }
  }
}
