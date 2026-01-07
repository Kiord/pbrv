#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_src;
uniform vec2 u_direction; 
uniform vec2 u_texel_size;

void main()
{
    vec2 off = u_direction * u_texel_size;

    vec3 c = vec3(0.0);

    c += texture(u_src, v_uv - 4.0 * off).rgb * 0.0162162162;
    c += texture(u_src, v_uv - 3.0 * off).rgb * 0.0540540541;
    c += texture(u_src, v_uv - 2.0 * off).rgb * 0.1216216216;
    c += texture(u_src, v_uv - 1.0 * off).rgb * 0.1945945946;
    c += texture(u_src, v_uv).rgb             * 0.2270270270;
    c += texture(u_src, v_uv + 1.0 * off).rgb * 0.1945945946;
    c += texture(u_src, v_uv + 2.0 * off).rgb * 0.1216216216;
    c += texture(u_src, v_uv + 3.0 * off).rgb * 0.0540540541;
    c += texture(u_src, v_uv + 4.0 * off).rgb * 0.0162162162;
    
    fragColor = vec4(c, 1.0);
}
