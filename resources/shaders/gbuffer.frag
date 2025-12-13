#version 330 core

in VS_OUT {
    vec3 worldPos;
    vec3 worldNormal;
    vec3 worldTangent;
    vec3 worldBitangent;
    vec2 uv;
} fs_in;

layout(location = 0) out vec4 gPosition;    
layout(location = 1) out vec4 gNormal;      
layout(location = 2) out vec4 gAlbedo;      
layout(location = 3) out vec4 gRMAOS;    
layout(location = 4) out vec4 gEmissive;    

uniform float u_time;

uniform vec3 u_albedo;
uniform sampler2D u_albedo_map; 
uniform bool u_use_albedo_map;



uniform sampler2D u_normal_map;
uniform bool      u_use_normal_map;

uniform float u_roughness;
uniform sampler2D u_roughness_map;
uniform bool      u_use_roughness_map;

uniform float u_metalness;
uniform sampler2D u_metalness_map;
uniform bool      u_use_metalness_map;

uniform vec3 u_emissive;
uniform sampler2D u_emissive_map;
uniform bool      u_use_emissive_map;

uniform float u_specular;
uniform sampler2D u_specular_map;
uniform bool      u_use_specular_map;

uniform sampler2D u_ao_map;
uniform bool      u_use_ao_map;

void main() {
    vec3 N_geom = normalize(fs_in.worldNormal);

    vec3 T = normalize(fs_in.worldTangent);
    vec3 B = normalize(fs_in.worldBitangent);
    mat3 TBN = mat3(T, B, N_geom);

    vec3 N = N_geom;
    if (u_use_normal_map) {
        vec3 n_tan = texture(u_normal_map, fs_in.uv).rgb;
        n_tan = n_tan * 2.0 - 1.0;

        //float t = float(sin(u_time) > 0.0);
        //N = t * normalize(TBN * n_tan) + (1-t) * N;
        N = normalize(TBN * n_tan);
    }

    gPosition = vec4(fs_in.worldPos, 1.0);
    gNormal   = vec4(N, 1.0);

    // --- Albedo ---
    vec3 albedo = u_albedo;
    if (u_use_albedo_map) {
        albedo = texture(u_albedo_map, fs_in.uv).rgb;  // map overwrites uniform
    }
    gAlbedo = vec4(albedo, 1.0);

    // --- Roughness / metallic / AO packed into gRMAOS ---
    float roughness   = u_roughness;
    float metalness = u_metalness;
    vec3 emissive = u_emissive;
    float specular = u_specular;
    float ao          = 1.0;

    if (u_use_roughness_map)
        roughness = texture(u_roughness_map, fs_in.uv).r;
    if (u_use_metalness_map)
        metalness = texture(u_metalness_map, fs_in.uv).r;
    if (u_use_specular_map)
        specular = texture(u_specular_map, fs_in.uv).r;
    if (u_use_ao_map)
        ao = texture(u_ao_map, fs_in.uv).r;

    gRMAOS = vec4(roughness, metalness, ao, specular);

    if (u_use_emissive_map)
        emissive = texture(u_emissive_map, fs_in.uv).rgb;
    gEmissive = vec4(emissive, 1);

}
