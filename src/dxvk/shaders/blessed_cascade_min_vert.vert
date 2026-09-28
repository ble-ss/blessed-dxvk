// blessed: cascade-cache -- fullscreen triangle for the cached-depth composite
#version 450

void main() {
  vec2 coord = vec2(
    float(gl_VertexIndex & 1) * 2.0f,
    float(gl_VertexIndex & 2));

  gl_Position = vec4(-1.0f + 2.0f * coord, 0.0f, 1.0f);
}
