// blessed: cascade-cache -- writes the cached static depth; the pipeline's LESS_OR_EQUAL test keeps min(live, cached)
#version 450

#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D s_cached;

void main() {
  gl_FragDepth = texelFetch(s_cached, ivec2(gl_FragCoord.xy), 0).r;
}
