#version 330 core

in vec2 v_uv;
out float aoBlur;

uniform sampler2D u_ssao;    
uniform sampler2D gPosition; 
uniform sampler2D gNormal;   

uniform float u_blur_depth_sigma;
uniform float u_blur_normal_sigma;

void main()
{
    vec3 centerPos = texture(gPosition, v_uv).xyz;
    vec3 centerN   = normalize(texture(gNormal,   v_uv).xyz);
    float centerAO = texture(u_ssao,    v_uv).r;

    if (centerPos.x == 100) {
        aoBlur = 1.0;
        return;
    }

    float sum = 0.0;
    float wsum = 0.0;

    vec2 texel = 1.0 / vec2(textureSize(u_ssao, 0));

    const int R = 2;

    for (int x = -R; x <= R; ++x) {
        for (int y = -R; y <= R; ++y) {
            vec2 offset = vec2(x, y) * texel;
            vec2 uv = v_uv + offset;

            float aoSample = texture(u_ssao, uv).r;
            vec3 posSample = texture(gPosition, uv).xyz;
            vec3 nSample   = normalize(texture(gNormal,   uv).xyz);

            if (posSample.x == 100)
                continue;

            float posDiff = length(centerPos - posSample);
            float nDiff   = 1.0 - max(dot(centerN, nSample), 0.0);

            float w_depth  = exp(-posDiff * u_blur_depth_sigma);
            float w_normal = exp(-nDiff  * u_blur_normal_sigma);

            float w = w_depth * w_normal;

            sum  += aoSample * w;
            wsum += w;
        }
    }

    if (wsum > 0.0)
        aoBlur = sum / wsum;
    else
        aoBlur = centerAO;

    //aoBlur = centerAO;
}   
