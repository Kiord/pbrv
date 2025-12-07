#version 330 core

in vec2 v_uv;

out vec4 fragColor;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gRMAO;
uniform sampler2D u_ssao;

uniform bool u_use_env;
uniform samplerCube u_irradiance_env;
uniform samplerCube u_specular_env;
uniform int u_num_specular_mips;

uniform bool u_use_ssao;

uniform vec3 u_lightPos;
uniform vec3 u_lightColor;
uniform vec3 u_viewPos;
uniform mat4 u_invView;
uniform mat4 u_invProj;

uniform float u_time;


const float A = 0.15;//ShoulderStrength
const float B = 0.50;//LinearStrength
const float C = 0.10;//LinearAngle
const float D = 0.20;//ToeStrength
const float E = 0.02;
const float F = 0.30;
const float W = 10.2;

vec3 Uncharted2Tonemap(vec3 x){
   	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 ACESFilm(vec3 x ){
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp(vec3(0.),vec3(1.),(x*(a*x+b))/(x*(c*x+d)+e));
}

vec3 ExposureCorrect(vec3 col, float linfac, float logfac){
	return linfac*(1.0 - exp(col*logfac));
}

vec3 LinearToGamma(vec3 linRGB){
    linRGB = max(linRGB, vec3(0.));
    return max(1.055 * pow(linRGB, vec3(0.416666667)) - 0.055, vec3(0.));
}


vec3 ACESFilmicToneMapping(vec3 col){
	vec3 curr = Uncharted2Tonemap(col);
    const float ExposureBias = 2.0;
	curr *= ExposureBias;
    curr /= Uncharted2Tonemap(vec3(W));
    return LinearToGamma(curr);
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
    out vec3 F0_out,
    out vec3 F_out
){
    vec3 L = normalize(u_lightPos - worldPos);

    float distance    = length(u_lightPos - worldPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance     = u_lightColor * attenuation;

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    float NdotL = max(dot(N, L), 0.0);

    vec3 specularBRDF = evalSpecularBRDF(N, V, L, roughness, F0, F_out);
    vec3 diffuseBRDF  = evalDiffuseBRDF(albedo, metallic, F_out);

    F0_out = F0;

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

    // Use view-dependent Fresnel for IBL
    vec3 F_ibl = fresnelSchlick(NdotV, F0);

    // Reuse the diffuse BRDF helper for irradiance
    vec3 diffuseBRDF_ibl = evalDiffuseBRDF(albedo, metallic, F_ibl);

    // Diffuse IBL: irradiance (E) * diffuse BRDF
    vec3 irradiance = texture(u_irradiance_env, N).rgb / PI;
    vec3 diffuseIBL = diffuseBRDF_ibl * irradiance;

    // Specular IBL: prefiltered env + Fresnel
    vec3 R = reflect(-V, N);

    // NOTE: set this to num_mips(specular_cube) - 1
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

    if (worldPos == vec3(0.0)) {
        if (u_use_env){
            vec3 bg  = texture(u_specular_env, viewDir).rgb;
            fragColor = vec4(bg, 1.0);
        }
        return;
    }

    vec3 N      = texture(gNormal, v_uv).rgb;
    vec3 albedo = texture(gAlbedo, v_uv).rgb;
    vec4 rmao   = texture(gRMAO, v_uv);

    float roughness = clamp(rmao.r, 0.04, 1.0);
    float metallic  = clamp(rmao.g, 0.0, 1.0);
    float ao        = clamp(rmao.b, 0.0, 1.0);

    N = normalize(N);

    // V is surface -> camera
    vec3 V = -viewDir;

    // -------- Direct lighting (BRDF) --------
    if (u_use_ssao) {
        ao = texture(u_ssao, v_uv).r;
    }

    vec3 F0;
    vec3 F_direct; // not used outside, but needed by evalSpecularBRDF
    vec3 Lo_direct = evaluateDirectLightingBRDF(
        worldPos, N, V,
        albedo, roughness, metallic,
        F0, F_direct
    );

    // -------- IBL (BRDF-based) --------
    vec3 Lo_ibl = evaluateIBLBRDF(
        N, V,
        albedo, roughness, metallic, ao,
        F0
    );

    // -------- Ambient / SSAO --------

    vec3 color = Lo_direct + Lo_ibl;

    color = ExposureCorrect(color, 2.1, -0.8);
    color = ACESFilmicToneMapping(color);

    fragColor = vec4(color, 1.0);
}