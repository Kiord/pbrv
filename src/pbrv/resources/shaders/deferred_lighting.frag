#version 330 core

#include shaders/procedural_environment.glsl 

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

in vec2 v_uv;

out vec4 fragColor;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gRMAOS;
uniform sampler2D gEmissive;
uniform sampler2D u_ssao;

uniform vec3 u_env_color;

uniform bool u_use_env_map;
uniform samplerCube u_background_env;
uniform samplerCube u_irradiance_env;
uniform samplerCube u_specular_env;
uniform float u_env_lod;
uniform int u_num_specular_mips;

uniform bool u_use_procedural_environment=false;
uniform bool u_use_procedural_sun=false;


uniform bool u_use_ssao;


uniform vec3 u_viewPos;
uniform mat4 u_invView;
uniform mat4 u_invProj;
uniform mat3 u_envRotation;

uniform float u_time;

uniform float u_specularTint = 0.0;

uniform bool u_use_point_light;
uniform vec3 u_pointLightPos;
uniform vec3 u_pointLightColor;

uniform bool  u_use_dir_light;
uniform vec3  u_dirLightDir =vec3(0,-1,0);
uniform vec3  u_dirLightColor;


uniform sampler2DShadow u_shadowMap;
uniform mat4  u_lightViewProj;

// const float GAMMA = 2.2;
const vec3 LUMINANCE_PERCEPTION = vec3(0.2126, 0.7152, 0.0722);

vec3 procedural_sun_radiance(vec3 rd);
vec3 procedural_environment(vec3 ro, vec3 rd);
vec3 procedural_environment_ggx(vec3 ro, vec3 rd, float roughness);
vec3 procedural_environment_lambert(vec3 ro, vec3 rd);

  
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}  

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
               * pow(1.0 - cosTheta, 5.0);
}

vec3 get_world_dir_from_uv(vec2 uv)
{
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clip = vec4(ndc, 1.0, 1.0);
    vec4 view = u_invProj * clip;
    view /= view.w;
    vec3 viewDir = normalize(view.xyz);
    vec4 worldDir4 = u_invView * vec4(viewDir, 0.0);
    return normalize(worldDir4.xyz);
}

vec3 evalSpecularBRDF(vec3 N, vec3 V, vec3 L, float roughness, vec3 F0, out vec3 F)
{
    vec3 H = normalize(V + L);

    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    F         = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;

    return numerator / denominator;
}

vec3 evalDiffuseBRDF(vec3 albedo, float metallic, vec3 F)
{
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    return kD * albedo / PI;
}



float shadowVisibility(vec3 worldPos, vec3 N, vec3 L)
{
    vec4 lightClip = u_lightViewProj * vec4(worldPos, 1.0);
    vec3 projCoords = (lightClip.xyz / lightClip.w) * 0.5 + 0.5;

    // Outside the shadow map
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0)
        return 1.0;

    vec2 shadowMapSize = vec2(4096);
    vec2 texel = 1.0 / shadowMapSize;

    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        vec2 uv = projCoords.xy + vec2(x, y) * texel;
        sum += texture(u_shadowMap, vec3(uv, projCoords.z)); // returns 0..1 compare result
    }

    return sum / 9.0;
}

vec3 evaluatePunctualLightingBRDF(
    vec3 radiance,
    vec3 position,
    vec3 worldPos,
    vec3 N,
    vec3 V,
    vec3 albedo,
    float roughness,
    float metallic,
    vec3 F0
){
    vec3 L = normalize(position - worldPos);

    float distance    = length(position - worldPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 effective_radiance   = radiance * attenuation;

    

    float NdotL = max(dot(N, L), 0.0);

    vec3 F;
    vec3 specularBRDF = evalSpecularBRDF(N, V, L, roughness, F0, F);
    vec3 diffuseBRDF  = evalDiffuseBRDF(albedo, metallic, F);

    return (diffuseBRDF + specularBRDF) * effective_radiance * NdotL;
}

vec3 evaluateDirectionalLightingBRDF(
    vec3 radiance,
    vec3 direction,
    vec3 worldPos,
    vec3 N,
    vec3 V,
    vec3 albedo,
    float roughness,
    float metallic,
    vec3 F0
){
    vec3 L = normalize(-direction);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        return vec3(0.0);
    }

    vec3 F;
    vec3 specularBRDF = evalSpecularBRDF(N, V, L, roughness, F0, F);
    vec3 diffuseBRDF  = evalDiffuseBRDF(albedo, metallic, F);

    float vis = shadowVisibility(worldPos, N, L);
    return (diffuseBRDF + specularBRDF) * radiance * NdotL * vis;
}

vec3 envSamplingDirection(vec3 dirWorld){
    return normalize(u_envRotation * dirWorld);
}

vec3 evaluateIBLBRDF(
    vec3 N,
    vec3 V,
    vec3 albedo,
    float roughness,
    float metallic,
    float ao,
    vec3 F0
){

    float NdotV = max(dot(N, V), 0.0);

    vec3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);

    vec3 diffuseBRDF_ibl = evalDiffuseBRDF(albedo, metallic, F_ibl);

    vec3 irradiance_env = u_env_color;
    vec3 specular_env = u_env_color;
    if (u_use_env_map){
        irradiance_env = texture(u_irradiance_env, envSamplingDirection(N)).rgb;
        float lod = roughness * float(u_num_specular_mips - 1);
        vec3 R = reflect(-V, N);
        specular_env = textureLod(u_specular_env, envSamplingDirection(R), lod).rgb;
    }else if (u_use_procedural_environment){
        vec3 sunDir = envSamplingDirection(normalize(-u_dirLightDir));
        irradiance_env = procedural_environment_lambert(envSamplingDirection(N), sunDir);
        vec3 R = reflect(-V, N);
        specular_env = procedural_environment_ggx(envSamplingDirection(R), sunDir, roughness).rgb;
    }

    vec3 diffuseIBL = diffuseBRDF_ibl * irradiance_env;
    vec3 specIBL = specular_env * F_ibl;

    float specWeight = roughness * roughness;
    float specAO = mix(1.0, ao, specWeight);

    return ao * diffuseIBL + specAO * specIBL;
}

vec3 get_background_color(){
    if (u_use_env_map){
        vec3 bg = vec3(0.0);
        vec3 viewDir = get_world_dir_from_uv(v_uv);
        viewDir = envSamplingDirection(viewDir);
        if (u_env_lod == 0.0)
            return texture(u_background_env, viewDir).rgb;
        else if (u_env_lod == u_num_specular_mips)
            return  texture(u_irradiance_env, viewDir).rgb / PI;
        else
           return textureLod(u_specular_env, viewDir, u_env_lod).rgb;
        return bg;
    }else if (u_use_procedural_environment){
        vec3 viewDir = envSamplingDirection(get_world_dir_from_uv(v_uv));
        vec3 sunDir = envSamplingDirection(normalize(-u_dirLightDir));
        return procedural_environment(viewDir, sunDir);
    }
    return u_env_color;
}


void main()
{
    vec4 worldPos4 = texture(gPosition, v_uv).rgba;

    if (worldPos4.a < 0.5) {
        fragColor = vec4(get_background_color(), 1);
        return;
    }

    vec3 worldPos = worldPos4.xyz;

    vec3 viewDir = normalize(worldPos - u_viewPos);
    vec3 N      = normalize(texture(gNormal, v_uv).rgb);
    vec3 albedo = texture(gAlbedo, v_uv).rgb;
    vec4 rmaos   = texture(gRMAOS, v_uv);

    float roughness = clamp(rmaos.r, 0.04, 1.0);
    float metallic  = clamp(rmaos.g, 0.0, 1.0);
    float specular  = clamp(rmaos.a, 0.0, 1.0);
    float ao        = clamp(rmaos.b, 0.0, 1.0);


    vec3 V = -viewDir;

    if (u_use_ssao) {
        ao = texture(u_ssao, v_uv).r;
    }
    //ao = 1.0;
    
    float luminance = dot(albedo, LUMINANCE_PERCEPTION);
    vec3 Ctint = luminance > 0.0 ? albedo / luminance : vec3(1.0);
    vec3 dielectricF0 =  0.08 * specular * mix(vec3(1.0), Ctint, u_specularTint);
    vec3 F0 = mix(dielectricF0, albedo, metallic);

    // Punctual lighting
    vec3 Lo_direct = vec3(0.0);
    if (u_use_point_light && dot(u_pointLightColor, u_pointLightColor) > 0.0) {
        Lo_direct += evaluatePunctualLightingBRDF(
            u_pointLightColor,
            u_pointLightPos,
            worldPos, N, V,
            albedo, roughness, metallic, F0
        );
    }
    
    // Directional light + shadow
    vec3 dirLightColor = u_dirLightColor;
    if (u_use_procedural_sun){
        dirLightColor = procedural_sun_radiance(-u_dirLightDir);
    }
    if (u_use_dir_light && dot(dirLightColor, dirLightColor) > 0.0) {
        Lo_direct += evaluateDirectionalLightingBRDF(
            dirLightColor,
            u_dirLightDir,
            worldPos, N, V,
            albedo, roughness, metallic, F0
        );
    }

    // IBL
    
    vec3 Lo_ibl = evaluateIBLBRDF(
        N, V,
        albedo, roughness, metallic, ao, F0
    );

    vec3 color =  Lo_direct + Lo_ibl;

    color += texture(gEmissive, v_uv).rgb;


    fragColor = vec4(color,1.0);
    //fragColor.rgb = texture(gAlbedo, v_uv).rgb;
}