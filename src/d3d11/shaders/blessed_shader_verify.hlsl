// blessed: shader-replace verifier, diffs the vanilla twin's target against the replacement twin's
//
// compiled by fxc into src/d3d11/blessed_shader_verify_dxbc.h (see
// blessed-shaders/build-verify-dxbc.sh), once as-is (2d targets) and once
// with /D VERIFY_3D=1 (3d uavs).
//
// one record of 8 uints per target, at byte offset g_record * 32:
//   0  max over pixels of the max channel |a - b| (float bits; nan sorts above inf)
//   4  pixels whose raw bits differ in any channel
//   8  pixels touched (a or b nonzero in any channel)
//   12 sum of min(max channel diff, 1) * 2^20 over pixels, low word
//   16 same sum, high word
//   20 ~(linear index) of the first differing pixel (max of ~i = min of i; 0 = none)
//   24 pixels where either side is nan or inf
//   28 unused

cbuffer params : register(b0) {
  uint g_record;
  uint3 g_pad;
};

#ifdef VERIFY_3D
Texture3D<float4> g_a : register(t0);
Texture3D<float4> g_b : register(t1);
#else
Texture2D<float4> g_a : register(t0);
Texture2D<float4> g_b : register(t1);
#endif

RWByteAddressBuffer g_out : register(u0);

groupshared uint gs_max;
groupshared uint gs_diff;
groupshared uint gs_touched;
groupshared uint gs_sumLo;
groupshared uint gs_sumHi;
groupshared uint gs_first;
groupshared uint gs_bad;

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
  if (gi == 0u) {
    gs_max = 0u;
    gs_diff = 0u;
    gs_touched = 0u;
    gs_sumLo = 0u;
    gs_sumHi = 0u;
    gs_first = 0u;
    gs_bad = 0u;
  }

  GroupMemoryBarrierWithGroupSync();

  uint w, h, d;
#ifdef VERIFY_3D
  g_a.GetDimensions(w, h, d);
  bool inside = tid.x < w && tid.y < h && tid.z < d;
  float4 a = inside ? g_a.Load(int4(tid, 0)) : 0.0f;
  float4 b = inside ? g_b.Load(int4(tid, 0)) : 0.0f;
#else
  g_a.GetDimensions(w, h);
  d = 1u;
  bool inside = tid.x < w && tid.y < h;
  float4 a = inside ? g_a.Load(int3(tid.xy, 0)) : 0.0f;
  float4 b = inside ? g_b.Load(int3(tid.xy, 0)) : 0.0f;
#endif

  uint4 ua = asuint(a);
  uint4 ub = asuint(b);

  if (inside && (any(ua != 0u) || any(ub != 0u))) {
    uint dummy;
    InterlockedAdd(gs_touched, 1u, dummy);

    // exponent all ones: nan or inf
    bool bad = any((ua & 0x7f800000u) == 0x7f800000u)
            || any((ub & 0x7f800000u) == 0x7f800000u);

    if (bad)
      InterlockedAdd(gs_bad, 1u, dummy);

    if (any(ua != ub)) {
      float4 diff = abs(a - b);
      float md = max(max(diff.x, diff.y), max(diff.z, diff.w));

      if (bad)
        md = asfloat(0x7f800000u);

      uint q = uint(min(md, 1.0f) * 1048576.0f);
      uint old;
      InterlockedAdd(gs_sumLo, q, old);

      if (old + q < old)
        InterlockedAdd(gs_sumHi, 1u, dummy);

      InterlockedAdd(gs_diff, 1u, dummy);
      InterlockedMax(gs_max, asuint(md), dummy);
      InterlockedMax(gs_first, ~((tid.z * h + tid.y) * w + tid.x), dummy);
    }
  }

  GroupMemoryBarrierWithGroupSync();

  if (gi == 0u && gs_touched != 0u) {
    uint base = g_record * 32u;
    uint old, dummy;
    g_out.InterlockedMax(base + 0u, gs_max, dummy);
    g_out.InterlockedAdd(base + 4u, gs_diff, dummy);
    g_out.InterlockedAdd(base + 8u, gs_touched, dummy);
    g_out.InterlockedAdd(base + 12u, gs_sumLo, old);

    if (old + gs_sumLo < old)
      g_out.InterlockedAdd(base + 16u, 1u, dummy);

    if (gs_sumHi != 0u)
      g_out.InterlockedAdd(base + 16u, gs_sumHi, dummy);

    g_out.InterlockedMax(base + 20u, gs_first, dummy);
    g_out.InterlockedAdd(base + 24u, gs_bad, dummy);
  }
}
