#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

float saturate(float x) { return clamp(x, 0.0, 1.0); }
float sqr(float x) { return x * x; }

const vec3  SUN_COLOR     = vec3(1.0, 0.95, 0.85);
const vec3 SUN_DUSK_COLOR = vec3(0.74, 0.33, 0.17);
const float SUN_INTENSITY = 20.0;
const float SUN_RADIUS    = 0.04;   // radians
const float SUN_HALO      = 0.5;

const vec3 DAY_SKY_TOP     = vec3(0.03, 0.07, 0.18);
const vec3 DAY_SKY_HORIZON = vec3(0.55, 0.65, 0.80);
const vec3 DAY_GROUND      = vec3(0.06, 0.05, 0.045);

const vec3 NIGHT_SKY_TOP     = vec3(0.005, 0.010, 0.030);
const vec3 NIGHT_SKY_HORIZON = vec3(0.015, 0.020, 0.040);
const vec3 NIGHT_GROUND      = vec3(0.004, 0.004, 0.005);

const vec3 DUSK_HORIZON = vec3(0.95, 0.45, 0.15);
const vec3 DUSK_TOP     = vec3(0.25, 0.10, 0.18);
const float DUSK_STRENGTH = 0.35;

const float START_NIGHT_SUN_Y = 0.65;
const float FULL_NIGHT_SUN_Y  = -0.05;

const float LAMBERT_HORIZON_W  = 0.75;
const float LAMBERT_SIGMA_LOBE = 0.90;

// clamp prefiltered seam sharpness
float base_horizon_width() { return 0.03; }

// day/night
float sun_up_factor(vec3 sunDir)
{
    return smoothstep(FULL_NIGHT_SUN_Y, START_NIGHT_SUN_Y, sunDir.y);
}

// Warm dusk
float dusk_factor(vec3 sunDir, float sunUp)
{
    float nearH = 1.0 - saturate(abs(sunDir.y) / 0.25); // horizon band
    nearH = nearH * nearH * (3.0 - 2.0 * nearH);

    float mid = sunUp * (1.0 - sunUp) * 4.0; // 0 at 0/1, max at 0.5
    return DUSK_STRENGTH * nearH * mid;
}

// cut by ground
float sky_weight_unblurred(float y)
{
    return smoothstep(-0.02, 0.04, y);
}

// sky/ground
vec3 sky_color_from_y(float y, float sunUp, float dusk)
{
    float t = saturate(y * 0.5 + 0.5); // -1..1 -> 0..1

    vec3 skyH = mix(NIGHT_SKY_HORIZON, DAY_SKY_HORIZON, sunUp);
    vec3 skyT = mix(NIGHT_SKY_TOP,     DAY_SKY_TOP,     sunUp);

    float ySky = saturate(y); // only above horizon
    float horizonW = 1.0 - smoothstep(0.0, 0.6, ySky);
    skyH = mix(skyH, DUSK_HORIZON, dusk * (0.75 + 0.25 * horizonW));
    skyT = mix(skyT, DUSK_TOP,     dusk * 0.35);

    vec3 sky = mix(skyH, skyT, pow(t, 1.6));

    // haze 
    float hazeAmt = mix(0.01, 0.08, sunUp);
    sky += hazeAmt * skyH * (1.0 - smoothstep(0.0, 0.25, abs(y)));

    return sky;
}

vec3 ground_color_from_y(float y, float sunUp, float dusk)
{
    float g = saturate(1.0 + y);

    vec3 day   = mix(DAY_GROUND   * 0.35, DAY_GROUND   * 1.15, pow(g, 0.6));
    vec3 night = mix(NIGHT_GROUND * 0.35, NIGHT_GROUND * 1.15, pow(g, 0.6));

    vec3 ground = mix(night, day, sunUp);

    float nearH = 1.0 - smoothstep(0.0, 0.25, abs(y));
    ground += dusk * nearH * 0.06 * DUSK_HORIZON;

    return ground;
}

vec3 sky_ground_unblurred(float y, float sunUp, float dusk)
{
    vec3 sky    = sky_color_from_y(y, sunUp, dusk);
    vec3 ground = ground_color_from_y(y, sunUp, dusk);

    float h = sky_weight_unblurred(y);
    return mix(ground, sky, h);
}

vec3 sky_ground_prefiltered(vec3 dir, float w, float sunUp, float dusk)
{
    float skyW = smoothstep(-w, w, dir.y);

    float skyY    = clamp(dir.y + 0.35 * w,  0.0, 1.0);
    float groundY = clamp(dir.y - 0.35 * w, -1.0, 0.0);

    vec3 sky    = sky_color_from_y(skyY, sunUp, dusk);
    vec3 ground = ground_color_from_y(groundY, sunUp, dusk);

    return mix(ground, sky, skyW);
}

// sun + scatter 
float sun_halo(vec3 V, vec3 S)
{
    float mu = max(dot(V, S), 0.0);
    return SUN_HALO * (pow(mu, 4.0) + 0.15 * pow(mu, 24.0));
}

float sun_disk(vec3 V, vec3 S, float radius)
{
    float mu = clamp(dot(V, S), -1.0, 1.0);
    float a  = acos(mu);
    float aa = 0.005;
    return 1.0 - smoothstep(radius - aa, radius + aa, a);
}

vec3 sun_luminance(float sunUpVis){
    vec3 sun_color = mix(SUN_DUSK_COLOR, SUN_COLOR, vec3(sunUpVis));
    return  sun_color * (SUN_INTENSITY * sunUpVis);
}

vec3 sun_radiance(vec3 V, vec3 S, float sunUpVis)
{
    float disk = sun_disk(V, S, max(SUN_RADIUS, 1e-6));
    float halo = sun_halo(V, S);
    vec3 Lsun = sun_luminance(sunUpVis);
    return Lsun * (disk + 0.02 * halo);
}

vec3 sun_scatter(vec3 V, vec3 S, float sunUpVis)
{
    float mu = clamp(dot(V, S), -1.0, 1.0);

    float rayleigh = 0.75 * (1.0 + mu * mu);
    float mie      = pow(max(mu, 0.0), 16.0);

    float horizon = 1.0 - saturate(abs(V.y));
    horizon *= horizon;

    vec3 warm = vec3(1.00, 0.70, 0.35);
    vec3 cool = vec3(0.30, 0.55, 1.00);
    vec3 col  = mix(cool, warm, saturate(0.35 + 0.65 * max(S.y, 0.0)));

    float s = (0.05 * rayleigh + 0.12 * mie) * (0.25 + 0.75 * horizon);
    return col * (SUN_INTENSITY * 0.10) * s * sunUpVis;
}

// Blurred sun
float gauss_angle(float theta, float sigma)
{
    sigma = max(sigma, 1e-6);
    return exp(-0.5 * sqr(theta / sigma));
}

float sigma_from_radius_halfmax(float radius)
{
    radius = max(radius, 1e-6);
    return radius / sqrt(2.0 * log(2.0));
}

vec3 sun_prefiltered(vec3 V, vec3 S, float sigmaLobe, float sunUp)
{
    float mu = clamp(dot(V, S), -1.0, 1.0);
    float theta = acos(mu);

    float sigmaSun = sigma_from_radius_halfmax(SUN_RADIUS);
    float sigmaEff = sqrt(sigmaSun * sigmaSun + sigmaLobe * sigmaLobe);
    float amp = (sigmaSun * sigmaSun) / (sigmaEff * sigmaEff);

    vec3 Lsun = sun_luminance(sunUp);

    float core = amp * gauss_angle(theta, sigmaEff);
    float halo = sun_halo(V, S);
    return Lsun * (core + 0.02 * halo);
}

// GGX
float ggx_prefilter_blend(float roughness)
{
    return smoothstep(0.0, 0.05, saturate(roughness));
}

float ggx_horizon_width(float roughness)
{
    float a = max(roughness * roughness, 1e-4);
    return sin(atan(1.5 * a));
}

float ggx_sigma_lobe(float roughness)
{
    float a = max(roughness * roughness, 1e-4);
    return 0.55 * a;
}

// Starrs
float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p)
{
    float n = hash12(p);
    return vec2(n, hash12(p + 17.17));
}

vec3 starfield(vec3 V, float sunUp, float skyW)
{
    float night = sqr(saturate(1.0 - sunUp));
    if (night <= 0.0) return vec3(0.0);

    float lon = atan(V.z, V.x);
    float lat = asin(clamp(V.y, -1.0, 1.0));
    vec2 uv = vec2(lon / (2.0 * PI) + 0.5, lat / PI + 0.5);

    vec3 col = vec3(0.0);

    float density = 380.0;
    vec2 g = uv * density;
    vec2 cell = floor(g);
    vec2 f = fract(g);

    float pick = step(0.9978, hash12(cell)); // sparse
    vec2  rnd  = hash22(cell + 13.7);
    vec2  d    = f - rnd;

    float r2 = dot(d, d);
    float star = exp(-r2 * 3200.0) * pick;

    float tint = hash12(cell + 7.1);
    vec3  c    = mix(vec3(0.70, 0.80, 1.00), vec3(1.00, 0.90, 0.70), tint);

    float amp = 4.0 + 14.0 * hash12(cell + 1.3);

    col += c * (star * amp);

    return col * night * skyW;
}

//  API

vec3 procedural_sun_radiance(vec3 rd){
    vec3 S = normalize(rd);
    float sunUp = sun_up_factor(S);
    return sun_luminance(sunUp);
}

vec3 procedural_environment(vec3 ro, vec3 rd)
{
    vec3 V = normalize(ro);
    vec3 S = normalize(rd);

    float sunUp = sun_up_factor(S);
    float dusk  = dusk_factor(S, sunUp);

    float skyW  = sky_weight_unblurred(V.y); // cuts sun/scatter/stars by ground
    float sunUpVis = sunUp * skyW;

    vec3 env = sky_ground_unblurred(V.y, sunUp, dusk);

    env += sun_scatter(V, S, sunUpVis);
    env += sun_radiance(V, S, sunUpVis);

    // stars only here
    env += starfield(V, sunUp, skyW);

    return env;
}

vec3 procedural_environment_ggx(vec3 ro, vec3 rd, float roughness)
{
    vec3 V = normalize(ro);
    vec3 S = normalize(rd);
    roughness = saturate(roughness);

    float sunUp = sun_up_factor(S);
    float dusk  = dusk_factor(S, sunUp);

    // exact match at roughness=0
    vec3 env0 = sky_ground_unblurred(V.y, sunUp, dusk);
    float skyW0 = sky_weight_unblurred(V.y);
    env0 += sun_scatter(V, S, sunUp * skyW0);
    env0 += sun_radiance(V, S, sunUp * skyW0);

    // blurred version
    float w = max(ggx_horizon_width(roughness), base_horizon_width());
    vec3 env1 = sky_ground_prefiltered(V, w, sunUp, dusk);

    float skyW1 = smoothstep(-w, w, V.y); // consistent with prefilter horizon
    env1 += sun_scatter(V, S, sunUp * skyW1);
    env1 += sun_prefiltered(V, S, ggx_sigma_lobe(roughness), sunUp); // already includes sunUp

    float t = ggx_prefilter_blend(roughness);
    return mix(env0, env1, t);
}

vec3 procedural_environment_lambert(vec3 ro, vec3 rd)
{
    vec3 V = normalize(ro);
    vec3 S = normalize(rd);

    float sunUp = sun_up_factor(S);
    float dusk  = dusk_factor(S, sunUp);

    float w = max(LAMBERT_HORIZON_W, base_horizon_width());
    vec3 env = sky_ground_prefiltered(V, w, sunUp, dusk);

    float skyW = smoothstep(-w, w, V.y);
    env += sun_scatter(V, S, sunUp * skyW);
    env += sun_prefiltered(V, S, LAMBERT_SIGMA_LOBE, sunUp);

    return env;
}
