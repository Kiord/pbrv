#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec3 in_tangent;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat3 u_normal_matrix;

out VS_OUT {
    vec3 worldPos;
    vec3 worldNormal;
    vec3 worldTangent;
    vec3 worldBitangent;
    vec2 uv;
} vs_out;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    vs_out.worldPos = world_pos.xyz;

    vec3 N = normalize(u_normal_matrix * in_normal);
    vec3 T = normalize(u_normal_matrix * in_tangent);
    // construct bitangent
    vec3 B = normalize(cross(N, T));

    vs_out.worldNormal   = N;
    vs_out.worldTangent  = T;
    vs_out.worldBitangent = B;
    vs_out.uv = in_uv;

    gl_Position = u_projection * u_view * world_pos;
}
