#version 330 core

out vec2 v_uv;

// Only one big triangle
const vec2 POS[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    vec2 p = POS[gl_VertexID];
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv = 0.5 * (p + 1.0); 
}
