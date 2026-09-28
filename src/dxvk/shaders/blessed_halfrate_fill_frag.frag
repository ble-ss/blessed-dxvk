// blessed: half-rate far field -- prefill the main lit pass's targets from the far layer (off frames)
#version 460

#extension GL_EXT_samplerless_texture_functions : require

// blessed: drawn twice per off frame, before the main lit pass's first draw
// (see docs/research/halfrate-far.md in the main repo):
//
// mode 0 (draw A, no depth write): pixels the prepass covered. Their
//   position is exact from this frame's depth; fetch the layer there.
// mode 1 (draw B, depth test ALWAYS + write): pixels the prepass left
//   empty. Walk this pixel's ray against the layer's own depth, then write
//   the layer's colour and that depth, so the sky dome later in the pass
//   stays behind the far field as it would behind real lod.
//
// Every draw that is not skipped still draws on top of this, so the fill
// never has to know which pixels a fresh draw will cover.

layout(std140, set = 0, binding = 0) uniform Params {
  mat4  invViewProj;       // this frame's CameraViewProjInverse (row-major bytes, v * M)
  mat4  viewProj;          // its inverse
  mat4  layerViewProj;     // the layer frame's view-projection
  mat4  layerInvViewProj;  // the layer frame's CameraViewProjInverse
  vec4  camDelta;          // xyz: camPosNow - camPosLayer (pos in layer = pos now + camDelta)
  vec4  params;            // x captureDist, y same-surface tolerance, z clear depth, w mode
  ivec4 size;              // xy target extent
} p;

layout(set = 0, binding = 1) uniform texture2D s_depth;   // this frame's prepass depth (our copy)
layout(set = 0, binding = 2) uniform texture2D s_col;
layout(set = 0, binding = 3) uniform texture2D s_rt2;
layout(set = 0, binding = 4) uniform texture2D s_rt3;
layout(set = 0, binding = 5) uniform texture2D s_ldepth;  // layer depth, or a mark

layout(location = 0) out vec4 o_rt0;
layout(location = 1) out vec2 o_mv;
layout(location = 2) out vec4 o_rt2;
layout(location = 3) out vec2 o_rt3;

const float MARK_SKY  = 2.0;
const float MARK_NEAR = -1.0;

bool isSurface(float ld) {
  return ld >= 0.0 && ld < p.params.z;
}

vec3 unproject(vec2 ndcXY, float depth, mat4 inv) {
  vec4 h = vec4(ndcXY, depth, 1.0) * inv;
  return h.xyz / h.w;
}

vec2 ndcOfTexel(ivec2 c) {
  vec2 uv = (vec2(c) + 0.5) / vec2(p.size.xy);
  return vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// posNow is camera-relative in this frame. Returns false only when the point
// is behind the layer's camera; points off the layer's screen clamp to its
// edge (the strip a turn uncovers gets the edge texels, for one frame).
bool toLayer(vec3 posNow, out ivec2 lc, out vec2 ndcPrev) {
  vec4 clip = vec4(posNow + p.camDelta.xyz, 1.0) * p.layerViewProj;
  lc = ivec2(0);
  ndcPrev = vec2(0.0);

  if (clip.w <= 0.0)
    return false;

  ndcPrev = clip.xy / clip.w;
  vec2 uv = vec2(ndcPrev.x * 0.5 + 0.5, (1.0 - ndcPrev.y) * 0.5);
  lc = clamp(ivec2(uv * vec2(p.size.xy)), ivec2(0), p.size.xy - 1);
  return true;
}

// a layer texel's surface, camera-relative in this frame
vec3 layerPos(ivec2 lc, float ld) {
  return unproject(ndcOfTexel(lc), ld, p.layerInvViewProj) - p.camDelta.xyz;
}

// the nearest surface texel along a short cross, for pixels the layer
// never saw (disocclusions); -1 when there is none within 8 texels
ivec2 dilate(ivec2 lc) {
  const ivec2 taps[12] = ivec2[12](
    ivec2( 2, 0), ivec2(-2, 0), ivec2(0,  2), ivec2(0, -2),
    ivec2( 4, 0), ivec2(-4, 0), ivec2(0,  4), ivec2(0, -4),
    ivec2( 8, 0), ivec2(-8, 0), ivec2(0,  8), ivec2(0, -8));

  for (int i = 0; i < 12; i++) {
    ivec2 t = clamp(lc + taps[i], ivec2(0), p.size.xy - 1);
    if (isSurface(texelFetch(s_ldepth, t, 0).r))
      return t;
  }

  return ivec2(-1);
}

void emit(ivec2 lc, vec2 ndcPix, vec2 ndcPrev) {
  o_rt0 = texelFetch(s_col, lc, 0);
  o_rt2 = texelFetch(s_rt2, lc, 0);
  o_rt3 = texelFetch(s_rt3, lc, 0).rg;
  // skyrim's GetSSMotionVector: (-0.5, 0.5) * (ndc now - ndc previous)
  o_mv  = vec2(-0.5, 0.5) * (ndcPix - ndcPrev);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 ndcPix = ndcOfTexel(coord);
  float depth = texelFetch(s_depth, coord, 0).r;
  bool empty = depth >= p.params.z;

  gl_FragDepth = depth;

  if (p.params.w < 0.5) {
    // draw A: the prepass covered this pixel
    if (empty)
      discard;

    vec3 pos = unproject(ndcPix, depth, p.invViewProj);
    float dist = length(pos);

    if (dist < p.params.x)
      discard;

    ivec2 lc;
    vec2 ndcPrev;

    if (!toLayer(pos, lc, ndcPrev))
      discard;

    float ld = texelFetch(s_ldepth, lc, 0).r;

    if (isSurface(ld)) {
      float layerDist = length(layerPos(lc, ld));

      if (layerDist < dist * (1.0 - p.params.y)) {
        // the layer holds something in front of this surface (a skipped
        // lod hill before a prepassed tree): refine once along our ray
        vec3 front = normalize(pos) * layerDist;
        ivec2 lc2;
        vec2 ndcPrev2;

        if (toLayer(front, lc2, ndcPrev2) && isSurface(texelFetch(s_ldepth, lc2, 0).r)) {
          emit(lc2, ndcPix, ndcPrev2);
          return;
        }
      }

      if (layerDist <= dist * (1.0 + p.params.y)) {
        emit(lc, ndcPix, ndcPrev);
        return;
      }
    }

    // the layer never saw this surface: borrow a neighbour
    ivec2 t = dilate(lc);

    if (t.x < 0)
      discard;

    emit(t, ndcPix, ndcPrev);
    return;
  }

  // draw B: the prepass left this pixel empty (non-prepassed lod, or sky)
  if (!empty)
    discard;

  // any depth gives a point on this pixel's ray (the camera is at the
  // origin); 0.5 stays finite whatever the far plane is. Start beyond the
  // furthest lod (150k units in the whiterun frame): the first fetch is
  // then a pure-rotation reprojection.
  vec3 dir = normalize(unproject(ndcPix, 0.5, p.invViewProj));
  float t = 400000.0;

  ivec2 lc = ivec2(0);
  vec2 ndcPrev = vec2(0.0);
  bool found = false;

  for (int i = 0; i < 3; i++) {
    if (!toLayer(dir * t, lc, ndcPrev))
      discard;

    float ld = texelFetch(s_ldepth, lc, 0).r;

    if (!isSurface(ld)) {
      // sky or never seen: leave it to the sky dome
      if (!found)
        discard;
      break;
    }

    t = max(dot(layerPos(lc, ld), dir), 1.0);
    found = true;
  }

  if (!toLayer(dir * t, lc, ndcPrev) || !isSurface(texelFetch(s_ldepth, lc, 0).r))
    discard;

  vec4 clip = vec4(dir * t, 1.0) * p.viewProj;
  gl_FragDepth = clamp(clip.z / clip.w, 0.0, 1.0);
  emit(lc, ndcPix, ndcPrev);
}
