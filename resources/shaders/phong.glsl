#version 330

in vec3 v_fragPos;
in vec3 v_normal;
in vec2 v_uv;

out vec4 fragColor;

uniform vec3 u_lightPos;
uniform vec3 u_viewPos;

uniform vec3 u_ambientColor;
uniform vec3 u_diffuseColor;
uniform vec3 u_specularColor;
uniform float u_shininess;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 lightDir = normalize(u_lightPos - v_fragPos);
    vec3 viewDir = normalize(u_viewPos - v_fragPos);

    // ambient
    vec3 ambient = u_ambientColor * u_diffuseColor;

    // diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * u_diffuseColor;

    // specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), u_shininess);
    vec3 specular = spec * u_specularColor;

    vec3 color = ambient + diffuse + specular + vec3(v_uv, 1);
    fragColor = vec4(color, 1.0);
}
