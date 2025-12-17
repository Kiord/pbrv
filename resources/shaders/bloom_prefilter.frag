#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_hdr;
uniform sampler2D u_emissive;

uniform float u_emissive_boost;
uniform float u_hdr_boost;


const vec3 REL_LUMINACE = vec3(0.2126, 0.7152, 0.0722);
const float MAX_BLOOM_LUMINANCE = 20.0;

void main()
{
    vec3 hdr = texture(u_hdr, v_uv).rgb * u_hdr_boost;
    vec3 emi = texture(u_emissive, v_uv).rgb * u_emissive_boost;

    float hdr_luminance = dot(hdr, REL_LUMINACE);

    vec3 bloom_hdr = hdr *  hdr_luminance;
    bloom_hdr = clamp(bloom_hdr, 0.0, MAX_BLOOM_LUMINANCE);

    fragColor = vec4(bloom_hdr + emi, 1.0);
}
