#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_hdr;    // linear HDR
uniform sampler2D u_bloom;  // blurred bloom (linear)

uniform float u_bloom_intensity; // e.g. 0.05 .. 0.5

uniform int u_tone_mapping_id;
uniform float u_exposure;

const float GAMMA = 2.2;
const float MAX_LUMINANCE = 100.0;


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
    col = min(col, vec3(MAX_LUMINANCE));
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

void main()
{
    vec3 hdr = texture(u_hdr, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;

    vec3 color = hdr + bloom * u_bloom_intensity;
 
    color = tonemap(color);

    fragColor = vec4(color, 1.0);
}
