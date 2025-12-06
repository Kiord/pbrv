#version 330 core

in vec2 v_uv;

out vec4 fragColor;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gRMAO;
uniform sampler2D u_ssao;

uniform bool      u_use_ssao;

uniform vec3 u_lightPos;
uniform vec3 u_lightColor;
uniform vec3 u_viewPos;

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

void main()
{
    vec3 worldPos = texture(gPosition, v_uv).rgb;
    vec3 N        = texture(gNormal,   v_uv).rgb;
    vec3 albedo   = texture(gAlbedo,   v_uv).rgb;
    vec4 rmao     = texture(gRMAO,     v_uv);

    // Early out if nothing was drawn here
    if (worldPos == vec3(0.0)) {
        fragColor = vec4(0.0);
        return;
    }

    // Decode material params from gRMAO
    float roughness = clamp(rmao.r, 0.04, 1.0);
    float metallic  = clamp(rmao.g, 0.0, 1.0);
    float ao        = clamp(rmao.b, 0.0, 1.0);



    // Assume albedo is stored in sRGB, convert to linear
    //albedo = pow(albedo, vec3(2.2));

    // Re-normalize normal
    N = normalize(N);

    vec3 V = normalize(u_viewPos  - worldPos);
    vec3 L = normalize(u_lightPos - worldPos);
    vec3 H = normalize(V + L);

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    float distance    = length(u_lightPos - worldPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance     = u_lightColor * attenuation;        
    
    // cook-torrance brdf
    float NDF = DistributionGGX(N, H, roughness);        
    float G   = GeometrySmith(N, V, L, roughness);      
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;	  
    
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular     = numerator / denominator;  
        
    // add to outgoing radiance Lo
    float NdotL = max(dot(N, L), 0.0);                
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL; 

    if (u_use_ssao) {
        ao = texture(u_ssao, v_uv).r;
    }

    vec3 ambient = vec3(0.005) * albedo * ao;
    vec3 color = ambient + Lo;
	
    color = ExposureCorrect(color, 2.1, -0.8);
    color = ACESFilmicToneMapping(color);
    
    fragColor = vec4(color, 1.0);
}