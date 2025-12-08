#version 330 core

in vec2 v_uv;
out float aoOut;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D u_ssao_noise;

uniform vec3 u_ssao_samples[32];
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec2 u_ssao_noise_scale;
uniform float u_ssao_radius;
uniform int   u_ssao_sample_count;
uniform float u_ssao_intensity;

void main()
{
    vec3 worldPos    = texture(gPosition, v_uv).xyz;
    vec3 worldNormal = texture(gNormal,   v_uv).xyz;

    if (worldPos == vec3(0.0)) {
        aoOut = 1.0;
        return;
    }

    vec3 posView    = (u_view * vec4(worldPos,    1.0)).xyz;
    vec3 normalView = normalize((u_view * vec4(worldNormal, 0.0)).xyz);

    vec3 rand = texture(u_ssao_noise, v_uv * u_ssao_noise_scale).xyz;
    vec3 tangent   = normalize(rand - normalView * dot(rand, normalView));
    vec3 bitangent = cross(normalView, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normalView);

    float occlusion = 0.0;

    for (int i = 0; i < u_ssao_sample_count; ++i)
    {
        vec3 sampleVecView = TBN * u_ssao_samples[i];
        vec3 samplePosView = posView + sampleVecView * u_ssao_radius;

        vec4 offsetClip = u_projection * vec4(samplePosView, 1.0);
        offsetClip.xyz /= offsetClip.w;
        vec2 offsetUV = offsetClip.xy * 0.5 + 0.5;

        if (offsetUV.x < 0.0 || offsetUV.x > 1.0 ||
            offsetUV.y < 0.0 || offsetUV.y > 1.0)
            continue;

        vec3 sampleWorld = texture(gPosition, offsetUV).xyz;
        if (sampleWorld.x == 100) 
            continue;
        vec3 sampleView  = (u_view * vec4(sampleWorld, 1.0)).xyz;

        float rangeCheck = smoothstep(0.0, 1.0,
                                      u_ssao_radius / abs(posView.z - sampleView.z));

        if (sampleView.z >= samplePosView.z)
            occlusion += rangeCheck;
    }

    float ao = 1.0 - (occlusion / float(u_ssao_sample_count));
    ao = pow(ao, u_ssao_intensity);

    aoOut = ao;
}
