#version 330 core

in vec2 v_uv;

out vec4 fragColor;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gRMAOS;
uniform sampler2D u_ssao;

uniform bool u_use_env;
uniform samplerCube u_background_env;
uniform samplerCube u_irradiance_env;
uniform samplerCube u_specular_env;
uniform int u_num_specular_mips;

uniform int u_tone_mapping_id;
uniform float u_exposure;

uniform bool u_use_ssao;

uniform vec3 u_lightPos;
uniform vec3 u_lightColor;
uniform vec3 u_viewPos;
uniform mat4 u_invView;
uniform mat4 u_invProj;

uniform float u_time;

uniform float u_specularTint = 0.0;

const float GAMMA = 2.2;
const vec3 LUMINANCE_PERCEPTION = vec3(0.2126, 0.7152, 0.0722);




vec3 linear_to_srgb(vec3 x) {
    return pow(clamp(x, 0.0, 1.0), vec3(1.0 / GAMMA));
}

vec3 tonemap_exposure(vec3 hdr, float exposure) {
    vec3 ldr = hdr * exposure;
    return ldr;  // still linear, do gamma after
}

// Or with gamma baked in:
vec3 tonemap_exposure_srgb(vec3 hdr, float exposure) {
    vec3 ldr = hdr * exposure;
    return linear_to_srgb(ldr);
}

vec3 tonemap_reinhard(vec3 hdr, float exposure) {
    vec3 x = hdr * exposure;
    return x / (vec3(1.0) + x);  // linear
}

vec3 tonemap_reinhard_srgb(vec3 hdr, float exposure) {
    return linear_to_srgb(tonemap_reinhard(hdr, exposure));
}

vec3 uncharted2_tonemap(vec3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F;
}

vec3 tonemap_uncharted2(vec3 hdr, float exposure) {
    const float W = 11.2; // white point used in Hable’s paper

    vec3 x = hdr * exposure;
    vec3 curr = uncharted2_tonemap(x);
    vec3 whiteScale = 1.0 / uncharted2_tonemap(vec3(W));
    return curr * whiteScale; // linear
}

vec3 tonemap_uncharted2_srgb(vec3 hdr, float exposure) {
    return linear_to_srgb(tonemap_uncharted2(hdr, exposure));
}

vec3 tonemap_aces(vec3 hdr, float exposure) {
    vec3 x = hdr * exposure;

    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    vec3 mapped = (x*(a*x + b)) / (x*(c*x + d) + e);
    return clamp(mapped, 0.0, 1.0); // linear
}

vec3 tonemap_aces_srgb(vec3 hdr, float exposure) {
    return linear_to_srgb(tonemap_aces(hdr, exposure));
}

vec3 tonemap(vec3 col){
    col = min(col, vec3(100));
    if (u_tone_mapping_id == 0)
        return tonemap_exposure_srgb(col, u_exposure);
    if (u_tone_mapping_id == 1)
        return tonemap_aces_srgb(col, u_exposure);
    if (u_tone_mapping_id == 2)
        return tonemap_reinhard_srgb(col, u_exposure);
    if (u_tone_mapping_id == 3)
        return tonemap_uncharted2_srgb(col, u_exposure);
    return col;
}


const float PI = 3.14159265359;
  
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

vec3 evaluateDirectLightingBRDF(
    vec3 worldPos,
    vec3 N,
    vec3 V,
    vec3 albedo,
    float roughness,
    float metallic,
    vec3 F0
){
    vec3 L = normalize(u_lightPos - worldPos);

    float distance    = length(u_lightPos - worldPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance     = u_lightColor * attenuation;

    

    float NdotL = max(dot(N, L), 0.0);

    vec3 F;
    vec3 specularBRDF = evalSpecularBRDF(N, V, L, roughness, F0, F);
    vec3 diffuseBRDF  = evalDiffuseBRDF(albedo, metallic, F);

    // Lo_direct
    return (diffuseBRDF + specularBRDF) * radiance * NdotL;
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
    if (!u_use_env) {
        return vec3(0.0);
    }

    float NdotV = max(dot(N, V), 0.0);

    vec3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);

    vec3 diffuseBRDF_ibl = evalDiffuseBRDF(albedo, metallic, F_ibl);

    vec3 irradiance = texture(u_irradiance_env, N).rgb;// / PI;
    //vec3 irradiance = textureLod(u_specular_env, N, 9).rgb * PI;
    vec3 diffuseIBL = diffuseBRDF_ibl * irradiance;// / PI;

    vec3 R = reflect(-V, N);

    float lod = roughness * float(u_num_specular_mips - 1);
    vec3 prefilteredColor = textureLod(u_specular_env, R, lod).rgb;

    vec3 specIBL = prefilteredColor * F_ibl;

    float specWeight = roughness * roughness;
    float specAO = mix(1.0, ao, specWeight);

    return ao * diffuseIBL + specAO * specIBL;
}

void main()
{
    vec3 worldPos = texture(gPosition, v_uv).rgb;
    vec3 viewDir = get_world_dir_from_uv(v_uv);
    if (worldPos.x > 2.0) {
        if (u_use_env){
            vec3 bg  = texture(u_background_env, viewDir).rgb;
            //vec3 bg  = texture(u_irradiance_env, viewDir).rgb / PI;
            //vec3 bg  = textureLod(u_specular_env, viewDir, 4).rgb;
            bg = tonemap(bg);
            fragColor = vec4(bg, 1.0);
        }
        return;
    }

    vec3 N      = normalize(texture(gNormal, v_uv).rgb);
    vec3 albedo = texture(gAlbedo, v_uv).rgb;
    vec4 rmaos   = texture(gRMAOS, v_uv);

    float roughness = clamp(rmaos.r, 0.04, 1.0);
    float metallic  = clamp(rmaos.g, 0.0, 1.0);
    float specular  = clamp(rmaos.a, 0.0, 1.0);
    float ao        = clamp(rmaos.b, 0.0, 1.0);

    N = normalize(N);

    // V is surface -> camera
    vec3 V = -viewDir;

    // -------- Direct lighting (BRDF) --------
    if (u_use_ssao) {
        ao = texture(u_ssao, v_uv).r;
    }
    //ao = 1.0;
    
    float luminance = dot(albedo, LUMINANCE_PERCEPTION);
    vec3 Ctint = luminance > 0.0 ? albedo / luminance : vec3(1.0);
    vec3 dielectricF0 =  0.08 * specular * mix(vec3(1.0), Ctint, u_specularTint);
    vec3 F0 = mix(dielectricF0, albedo, metallic);

    vec3 Lo_direct = evaluateDirectLightingBRDF(
        worldPos, N, V,
        albedo, roughness, metallic, F0
    );

    // -------- IBL (BRDF-based) --------
    vec3 Lo_ibl = evaluateIBLBRDF(
        N, V,
        albedo, roughness, metallic, ao, F0
    );

    vec3 color = Lo_direct + Lo_ibl;
    
    color = tonemap(color);

    fragColor = vec4(color,1.0);
    //fragColor.rgb=vec3(roughness);
}