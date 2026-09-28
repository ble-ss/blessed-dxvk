// blessed: shader-replace, cs.dbb99d2e (volumetric lighting vertical edge-aware blur), blur3: compact dispatch
//
// blur2 (../cs.dbb99d2e371c9076e2476c023672a740.hlsl) keeps the vanilla
// dispatch shape (one group per texel column, COLS=4 of every 4 doing
// work, the other 3 returning at gid.x % COLS != 0) so a plain shader
// swap needs no dispatch change. this variant drops that early return
// and instead expects the dispatch itself to be cut to width/COLS groups
// in x (see blessed_dispatch_rewrite.h): group x now directly indexes
// the COLS-wide column band it owns (col = gid.x*COLS + tig.x), so every
// launched group does real work. same TILE/APRON/LOADS layout, same
// float op order and constants as vanilla and as blur2 -- only which
// groups get launched, and what column each computes, changes.

cbuffer cb0 : register(b0) {
  float4 c0;  // xy: texel size
  float4 c1;  // xy: uv clamp
};

SamplerState s0 : register(s0);
SamplerState s1 : register(s1);
Texture2D<float4> t0 : register(t0);  // x: the value blurred
Texture2D<float4> t1 : register(t1);  // x: the edge key (depth)
RWTexture2D<float4> u0 : register(u0);

#define TILE    540
#define APRON   15
#define LOADS   (TILE + 2 * APRON)
#ifndef COLS
#define COLS    4
#endif
#ifndef THREADS
#define THREADS 128
#endif
#define LOAD_STEPS ((LOADS + THREADS - 1) / THREADS)
#define OUT_STEPS  ((TILE + THREADS - 1) / THREADS)

groupshared float g0[COLS][LOADS];
groupshared float g1[COLS][LOADS];

[numthreads(COLS, THREADS, 1)]
void main(uint3 gid : SV_GroupID, uint3 tig : SV_GroupThreadID) {
  uint width, height;
  u0.GetDimensions(width, height);

  uint col = gid.x * COLS + tig.x;

  [unroll]
  for (uint k = 0; k < LOAD_STEPS; k++) {
    uint j = tig.y + k * THREADS;

    if (j < LOADS) {
      int py = int(gid.y) * TILE + int(j) - APRON;
      float2 uv = min((float2(int(col), py) + 0.5f) * c0.xy, c1.xy);
      g0[tig.x][j] = t0.SampleLevel(s0, uv, 0.0f).x;
      g1[tig.x][j] = t1.SampleLevel(s1, uv, 0.0f).x;
    }
  }

  GroupMemoryBarrierWithGroupSync();

  if (col >= width)
    return;

  [unroll]
  for (uint m = 0; m < OUT_STEPS; m++) {
    uint i = tig.y + m * THREADS;

    if (i < TILE) {
      uint j = i + APRON;
      float a = g0[tig.x][j];
      float b = g1[tig.x][j];

      float e = b * 5.0f - g1[tig.x][j - 12];
      e = e - g1[tig.x][j - 6];
      e = e - b;
      e = e - g1[tig.x][j + 6];
      e = e - g1[tig.x][j + 12];

      float o = a;

      if (!(asfloat(0x3b03126fu) < abs(e))) {
        float s = g0[tig.x][j - 6] * asfloat(0x3e577b39u);
        s = g0[tig.x][j - 12] * asfloat(0x3e36ae7du) + s;
        s = a * asfloat(0x3e63ac93u) + s;
        s = g0[tig.x][j + 6] * asfloat(0x3e577b39u) + s;
        o = g0[tig.x][j + 12] * asfloat(0x3e36ae7du) + s;
      }

      u0[uint2(col, gid.y * TILE + i)] = o.xxxx;
    }
  }
}
