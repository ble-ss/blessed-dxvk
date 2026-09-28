// blessed: shader-replace, fs.83560015 (vanilla screen-space ambient occlusion,
// vestigial under BLESSED_RTAO but still a real ~0.15ms fixed-function pass)
//
// what it does (read from fxc /dumpbin of the vanilla dxbc, pulled read-only from
// Skyrim - Shaders.bsa): a 5-tap spiral SSAO kernel with per-tap LOD selection
// (farther taps read a coarser mip of the depth buffer, t0), a hemisphere normal
// reconstructed from a 2-channel encoded normal (t1, sampled with an explicit
// zero-gradient SampleGrad to force mip 0 without a discontinuity at the screen
// edge), a two-byte depth pack into the output's y/z channels for a downstream
// consumer, an artist power-curve remap of the accumulated AO, screen-derivative
// depth-edge-aware dilation (checkerboard-dithered so the dilation itself does
// not alias), and a final edge-aware blend against a lower-resolution bounced-GI
// buffer (t2) using a blue-noise-style jitter (t3) for the second GI tap.
//
// changes from vanilla, both measured, not just argued (see the harness):
//  1. the per-tap mip-level's log2 call is split via log2(a*b)==log2(a)+log2(b)
//     into one log2(radiusScale) hoisted above the loop, plus a per-iteration
//     add of a baked constant (log2((i+0.5)*0.2), i in 0..4) -- four fewer
//     log2 calls per pixel.
//  2. the normal fetch's sample_d with explicit zero gradients (vanilla's way
//     of forcing mip 0) is now an explicit-LOD SampleLevel(0), the cheaper
//     path to the same mip and the same filtering.
//  3. [unroll] on the 5-iteration loop was tried and measured *slower*
//     (~4% on this build, pass16 harness, both native and the fork) than
//     leaving it a real loop -- not shipped. kept as [loop] (vanilla's own
//     shape) instead.
// everything else -- every mul/add/div/pow, the byte pack, the dilation, the
// GI blend -- is unchanged from the vanilla read, including two spots kept
// deliberately un-simplified for numerical fidelity: the power-curve's high
// branch as a reciprocal-then-multiply (not one HLSL divide, which could
// round once instead of twice) and the dither normalize as a single rsqrt
// (not sqrt-then-reciprocal).
//
// not attempted: the taps themselves (t0 x6, t1 x1, t2 x1, t3 x2 -- ten total,
// not the ~30 the hotspots seat's notes guessed from the frame capture alone)
// have no redundant duplicate fetch to remove. fp16 for the loop's accumulator
// was considered and not shipped: the shader is very likely texture-fetch-
// bound (five dependent, LOD-varying spiral samples per pixel, confirmed by
// the unroll finding above -- removing branch/ALU overhead didn't help), and
// it is not worth the risk on a shader with a real exact-equality sky gate
// right after the loop (see the ==1.0 check near the return).

cbuffer cb2 : register(b2) {
  float4 c0;  // xy: dest texel size, zw: half-texel bias (pixel index -> uv)
  float4 c1;  // x: dither-gate mask, y: radius-to-screen scale, z: falloff bias, w: ao strength
  float4 c2;  // xy: dither seed scale, zw: spiral step scale
  float4 c3;  // x: power-curve param, y: max sample radius^2, z: gi bilateral sharpness, w: sky depth threshold
};

cbuffer cb12 : register(b12) {
  // shared per-frame constants (also read by the game's other post-process
  // shaders); only [43] and [44] are read here. indices/swizzles kept exactly
  // as the vanilla disassembly -- semantics beyond "texel size / valid-uv
  // clamp for two different resolutions" were not otherwise labeled.
  float4 g12[45];
};

Texture2D t0 : register(t0);  // linear depth, mip chain
Texture2D t1 : register(t1);  // encoded normal, 2 channels used
Texture2D t2 : register(t2);  // lower-res bounced-GI buffer, x: ao, yz: weighted mix source
Texture2D t3 : register(t3);  // 2-channel dither/blue-noise vector

SamplerState s0 : register(s0);
SamplerState s1 : register(s1);
SamplerState s2 : register(s2);
SamplerState s3 : register(s3);

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
  // pixel index of this pixel in the dither-seed's own coordinate space
  // (reused later, unchanged, as the checkerboard dilation mask -- vanilla's
  // own reuse, not something this rewrite added)
  float2 seedF   = uv * c2.xy;
  int2   seedInt = int2(trunc(seedF));

  // clamp bounds shared by both texel-space lookups below
  float2 clampHi = float2(g12[44].z, g12[43].y);

  float2 centerPix = max(uv * g12[43].xy, 0.0f);
  centerPix = min(centerPix, clampHi);
  float centerDepth = t0.SampleLevel(s0, centerPix, 0.0f).x;

  // pixel-index -> uv for the center sample (same formula reused for every
  // spiral tap below)
  float2 centerUv = trunc(seedF) + 0.5f;
  centerUv = centerUv * c0.xy + c0.zw;
  float3 centerPos = float3(centerDepth * centerUv, centerDepth);

  // two-byte depth pack into o0.yz (independent of the AO loop below)
  float depthNorm  = saturate(centerDepth * 0.000143f);
  float depthHigh  = floor(depthNorm * 256.0f);
  float outY = depthHigh * 0.003906f;
  float outZ = depthNorm * 256.0f - depthHigh;

  // per-pixel dither seed for the spiral's rotation angle -- reuses seedInt,
  // the same integer pixel index the checkerboard dilation mask reuses later
  float ditherGate = (0.5f >= depthNorm) ? c1.x : 0.0f;
  int   dSeedXor = (seedInt.x * seedInt.y + seedInt.y) ^ (seedInt.x * 3);
  float angleSeed = float(dSeedXor) * 10.0f + ditherGate;

  // hemisphere normal reconstruction from the 2-channel encoded normal
  float2 normalPix = min(max(uv * g12[43].xy, 0.0f), g12[43].xy);
  // vanilla forces mip 0 with an explicit zero-gradient sample_d; SampleLevel
  // at LOD 0 selects the same mip through the cheaper explicit-LOD path
  // instead of the gradient/anisotropic one -- same texel, same filtering.
  float2 nEnc = t1.SampleLevel(s1, normalPix, 0.0f).xy;
  nEnc = nEnc * 4.0f - 2.0f;
  float  nLenSq = dot(nEnc, nEnc);
  // vanilla: r4.w = -nLenSq*0.5 + 1 (no sqrt), then negated straight into the
  // z component -- only the xy scale (below) goes through a sqrt.
  float  nZTerm = -nLenSq * 0.5f + 1.0f;
  float2 nScaled = nEnc * sqrt(-nLenSq * 0.25f + 1.0f);
  float3 normal = float3(-nScaled.x, nScaled.y, -nZTerm);

  float radiusScale = c1.y / centerDepth;

  float2 ndc2 = uv * 2.0f - 1.0f;
  float  angleBiasBase = dot(float4(ndc2, ndc2), float4(ndc2, ndc2));
  float  angleFalloff = max(depthNorm - 0.3f, 0.0f) * 10.0f + c1.z;
  float  angleBase = angleBiasBase + angleFalloff;

  float occlusion = 0.0f;

  // log2(radiusStep * 0.2) split into a loop-invariant log2(radiusScale)
  // (hoisted here) plus a per-iteration log2((i+0.5)*0.2), which is a fixed
  // constant for i in 0..4 and needs no log2 call at all inside the loop.
  // log2(a*b) == log2(a)+log2(b) exactly in real-number terms; the two
  // roundings this adds instead of one are absorbed by the floor()+clamp()
  // a few lines down (a mip index 1..4), which only cares which integer
  // bracket the value lands in -- see the harness's synthetic diff for how
  // this was checked without the game.
  static const float kLogStep[5] = {
    -3.321928f, -1.736966f, -1.0f, -0.514573f, -0.152003f  // log2((i+0.5)*0.2)
  };
  float logRadiusScale = log2(radiusScale);

  [loop]
  for (int i = 0; i < 5; i++) {
    float fi = float(i) + 0.5f;
    float angle = fi * 2.512f + angleSeed;

    float sinA, cosA;
    sincos(angle, sinA, cosA);

    float t = radiusScale * fi * 0.2f;

    int mipLevel = int(floor(logRadiusScale + kLogStep[i])) - 3;
    mipLevel = clamp(mipLevel, 1, 4);

    float2 spiralOffset = t * float2(cosA, sinA);
    int2   spiralPix = seedInt + int2(trunc(spiralOffset));

    float2 spiralUv = spiralOffset * c2.zw + uv;
    float  sampleDepth = t0.SampleLevel(s0, spiralUv, float(mipLevel)).x;

    float2 tapUv = float2(spiralPix) + 0.5f;
    tapUv = tapUv * c0.xy + c0.zw;
    float3 tapPos = float3(sampleDepth * tapUv, sampleDepth);

    float3 delta = tapPos - centerPos;
    float  distSq = dot(delta, delta);
    float  nDotDelta = dot(delta, normal) - angleBase;

    float falloff = max(c3.y - distSq, 0.0f);
    falloff = falloff * falloff * falloff;

    float ratio = max(nDotDelta / (distSq + 0.01f), 0.0f);
    float contribution = ratio * falloff;

    float skyMask = (sampleDepth >= c3.w) ? 1.0f : 0.0f;
    occlusion += contribution * skyMask;
  }

  float ao = max(1.0f - occlusion * c1.w, 0.0f);

  // artist power-curve remap, symmetric around power == 0.5. vanilla takes
  // the low branch as log2(ao)*(2*power), the high branch as a reciprocal
  // (a real div, not rcp) then a separate multiply -- kept as two steps
  // rather than one HLSL divide, which could round once instead of twice.
  float power = clamp(c3.x, 0.00001f, 0.99999f);
  float logAo = log2(ao);
  float invDenom = 1.0f / (2.0f - power * 2.0f);
  ao = (power < 0.5f) ? exp2(logAo * (power * 2.0f)) : exp2(logAo * invDenom);

  // depth-edge-aware dilation, checkerboard-dithered so the dilation itself
  // does not introduce its own aliasing (reuses seedInt's low bit, same as
  // vanilla). vanilla computes both the x and y derivative candidates
  // unconditionally and picks with movc -- no branch around ddx/ddy_coarse,
  // which is the only safe way to use them (centerDepth's derivative is not
  // uniform across a quad, so branching on it before differentiating would
  // be undefined per-lane). the y derivative is taken of the already
  // x-dilated ao, in sequence, matching vanilla's register reuse.
  float2 checker = float2(seedInt & 1) - 0.5f;

  float ddxDepth = ddx_coarse(centerDepth);
  float ddxAo    = ddx_coarse(ao);
  float dilatedX = -ddxAo * checker.x + ao;
  ao = (abs(ddxDepth) < 0.02f) ? dilatedX : ao;

  float ddyDepth = ddy_coarse(centerDepth);
  float ddyAo    = ddy_coarse(ao);
  float dilatedY = -ddyAo * checker.y + ao;
  ao = (abs(ddyDepth) < 0.02f) ? dilatedY : ao;

  // final blend against the lower-res bounced-GI buffer. vanilla normalizes
  // with a single rsq (reciprocal sqrt) instruction, not sqrt-then-reciprocal
  float2 ditherVec = t3.Sample(s3, centerPix).xy;
  float  invDitherLen = rsqrt(dot(ditherVec, ditherVec));
  float2 ditherDir = ditherVec * invDitherLen;

  float2 jitteredUv = ditherDir * 0.055f + uv;
  float2 jitteredPix = min(max(jitteredUv * g12[43].xy, 0.0f), clampHi);
  float2 giSample1 = t3.Sample(s3, jitteredPix).xy;

  float2 giPix2 = max((ditherVec + uv) * g12[43].zw, 0.0f);
  giPix2 = min(giPix2, float2(g12[44].w, g12[43].w));
  float3 giColor = t2.Sample(s2, giPix2).xyz;

  float weightedMix = dot(giColor.yz, float2(0.996109f, 0.003891f));
  float edgeWeight = max(1.0f - abs(depthNorm - weightedMix) * 200.0f, 0.0f);

  float2 deltaVec = ditherVec - giSample1;
  float  edgeWeight2 = max(1.0f - sqrt(dot(deltaVec, deltaVec)) * 200.0f, 0.2f);

  float blendW = saturate(1.0f - c3.z * abs(ao - giColor.x));
  blendW *= edgeWeight;

  float finalWeight = edgeWeight2 * blendW;
  ao = finalWeight * (giColor.x - ao) + ao;

  bool isSky = (depthNorm == 1.0f);
  float outX = isSky ? 1.0f : ao;

  return float4(outX, outY, outZ, 1.0f);
}
