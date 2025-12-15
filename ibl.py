from dataclasses import dataclass
from typing import Optional
import math

import moderngl
from moderngl import Context, Texture, TextureCube, ComputeShader
from sun_extraction import SunExtraction
import numpy as np
from constants import EPSILON

from utils import safe_set_uniform

@dataclass
class PrefilterSettings:
    background_size: int = 1024
    num_mips: Optional[int] = None
    
    specular0_size: int = 512
    specular_sample_count: int = 1024

    irradiance_size: int = 32
    irradiance_sample_count: int = 1024*16


def _set_exclusion_uniforms(cs: ComputeShader, to_exclude_sun: Optional[SunExtraction]) -> None:

    if to_exclude_sun is None:
        safe_set_uniform(cs, "u_exclude_enable", False)
        return
    
    d = np.asarray(to_exclude_sun.direction, dtype=np.float32)
    d = d / (np.linalg.norm(d) + EPSILON)
    
    safe_set_uniform(cs, "u_exclude_enable", True)
    safe_set_uniform(cs, "u_exclude_dir", d)
    safe_set_uniform(cs, "u_exclude_cos", to_exclude_sun.exclude_cos)
    safe_set_uniform(cs, "u_exclude_feather", to_exclude_sun.feather)

class EnvironmentMapPrecomputer:

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._pano_to_cube_cs: Optional[ComputeShader] = None
        self._prefilter_cs: Optional[ComputeShader] = None
        self._irradiance_cs: Optional[ComputeShader] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def panorama_to_cubemap(self,  panorama:Texture, cube_size:int, release=True):
        self._ensure_shaders()

        cube = self.ctx.texture_cube(
            (cube_size, cube_size),
            components=4,   # RGBA16F to match layout(rgba16f)
            dtype="f2",
        )

        self._dispatch_panorama_to_cubemap(panorama, cube, cube_size)
        if release:
            panorama.release()

        return cube

    def __call__(self, background_cube:Texture|TextureCube, 
                settings: PrefilterSettings | None = None,
                to_exclude_sun: Optional[SunExtraction] = None,
                release=True):
        if settings is None:
            settings = PrefilterSettings()
        spec0_size = settings.specular0_size
        bkg_size = settings.background_size

        self._ensure_shaders()

        if isinstance(background_cube, Texture):
            background_cube = self.panorama_to_cubemap(background_cube, bkg_size, release)


        irr_size = settings.irradiance_size

        irradiance_cube = self.ctx.texture_cube(
            (irr_size, irr_size),
            components=4,   # RGBA16F
            dtype="f2",
        )

        irradiance_cube.filter = (moderngl.LINEAR, moderngl.LINEAR)
        irradiance_cube.repeat_x = True
        irradiance_cube.repeat_y = True

        self._dispatch_irradiance(
            src_env=background_cube,
            dst_irradiance=irradiance_cube,
            size=irr_size,
            sample_count=settings.irradiance_sample_count,
            to_exclude_sun=to_exclude_sun,
        )

        # Specular cube map

        max_mips =  settings.num_mips or int(math.floor(math.log2(spec0_size))) + 1
        max_mips = max(1, max_mips)

        specular_cube = self.ctx.texture_cube(
            (spec0_size, spec0_size),
            components=4,
            dtype="f2",
        )

        specular_cube.build_mipmaps()
        specular_cube.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        # specular_cube.repeat_x = True
        # specular_cube.repeat_y = True

        self._dispatch_specular_prefilter(
            src_env=background_cube,
            dst_prefiltered=specular_cube,
            size=spec0_size,
            max_mips=max_mips,
            sample_count=settings.specular_sample_count,
            to_exclude_sun=None, # to_exclude_sun,# None for now...
        )

        self.ctx.finish()
        

        return background_cube, irradiance_cube, specular_cube, max_mips


    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_shaders(self) -> None:
        if self._pano_to_cube_cs is None:
            self._pano_to_cube_cs = self.ctx.compute_shader(_PANORAMA_TO_CUBE_CS)
        if self._irradiance_cs is None:
            self._irradiance_cs = self.ctx.compute_shader(_IRRADIANCE_CS)
        if self._prefilter_cs is None:
            self._prefilter_cs = self.ctx.compute_shader(_SPECULAR_PREFILTER_CS)

    def _dispatch_panorama_to_cubemap(
        self,
        pano: Texture,
        cube: TextureCube,
        face_size: int,
    ) -> None:
        """
        Write panorama into cubemap mip level 0 using a compute shader.
        """
        cs = self._pano_to_cube_cs

        # Set uniforms
        cs["u_face_size"].value = face_size
        cs["u_panorama"].value = 0  # texture unit 0

        pano.use(location=0)

        # Bind cubemap level 0 as imageCube (write-only)
        cube.bind_to_image(0, read=False, write=True, level=0)

        local_size = 8  # matches layout in compute shader
        groups_x = (face_size + local_size - 1) // local_size
        groups_y = (face_size + local_size - 1) // local_size
        groups_z = 6  # 6 faces

        cs.run(groups_x, groups_y, groups_z)

    def _dispatch_irradiance(
        self,
        src_env: TextureCube,
        dst_irradiance: TextureCube,
        size: int,
        sample_count: int,
        to_exclude_sun: Optional[SunExtraction]
    ) -> None:
        cs = self._irradiance_cs

        src_env.use(location=0)
        cs["u_env_map"].value = 0
        cs["u_face_size"].value = size
        cs["u_sample_count"].value = int(sample_count)
        _set_exclusion_uniforms(cs, to_exclude_sun)

        local_size = 8
        groups_x = (size + local_size - 1) // local_size
        groups_y = (size + local_size - 1) // local_size
        groups_z = 6

        dst_irradiance.bind_to_image(0, read=False, write=True, level=0)
        cs.run(groups_x, groups_y, groups_z)

    def _dispatch_specular_prefilter(
        self,
        src_env: TextureCube,
        dst_prefiltered: TextureCube,
        size: int,
        max_mips: int,
        sample_count: int,
        to_exclude_sun: Optional[SunExtraction]
    ) -> None:
        cs = self._prefilter_cs

        src_env.use(location=0)
        cs["u_env_map"].value = 0
        cs["u_sample_count"].value = int(sample_count)
        _set_exclusion_uniforms(cs, to_exclude_sun)

        local_size = 8  # matches layout in compute shader

        for level in range(max_mips):
            mip_size = max(1, size >> level)
            roughness = 0.0 if max_mips == 1 else level / float(max_mips - 1)

            cs["u_face_size"].value = mip_size
            cs["u_roughness"].value = float(roughness)

            groups_x = (mip_size + local_size - 1) // local_size
            groups_y = (mip_size + local_size - 1) // local_size
            groups_z = 6  # faces

            dst_prefiltered.bind_to_image(0, read=False, write=True, level=level)

            cs.run(groups_x, groups_y, groups_z)


_PANORAMA_TO_CUBE_CS = """
#version 430

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform imageCube u_out_cube;

uniform sampler2D u_panorama;
uniform int u_face_size;

const float PI = 3.14159265358979323846;

vec2 sample_spherical_map(vec3 v) {
    v = normalize(v);
    float lon = atan(v.z, v.x);
    float lat = asin(clamp(v.y, -1.0, 1.0));

    float u = lon / (2.0 * PI) + 0.5;
    float v_tex = 0.5 - lat / PI;
    return vec2(u, v_tex);
}

vec3 face_uv_to_dir(uint face, vec2 uv) {
    // uv in [0, 1]
    vec2 st = uv * 2.0 - 1.0;   // [-1, 1]
    float s = st.x;
    float t = st.y;

    if (face == 0u) {          // +X (right)
        return normalize(vec3( 1.0, -t, -s));
    } else if (face == 1u) {   // -X (left)
        return normalize(vec3(-1.0, -t,  s));
    } else if (face == 2u) {   // +Y (top)
        return normalize(vec3( s,  1.0,  t));
    } else if (face == 3u) {   // -Y (bottom)
        return normalize(vec3( s, -1.0, -t));
    } else if (face == 4u) {   // +Z (front)
        return normalize(vec3( s, -t,  1.0));
    } else {                   // -Z (back)
        return normalize(vec3(-s, -t, -1.0));
    }
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    int x = gid.x;
    int y = gid.y;
    int face = gid.z;

    if (x >= u_face_size || y >= u_face_size || face >= 6) {
        return;
    }

    vec2 uv = (vec2(x, y) + vec2(0.5)) / float(u_face_size);
    vec3 dir = face_uv_to_dir(uint(face), uv);

    vec2 pano_uv = sample_spherical_map(dir);
    vec3 color = texture(u_panorama, pano_uv).rgb;

    imageStore(u_out_cube, ivec3(x, y, face), vec4(color, 1.0));
}
"""


_SPECULAR_PREFILTER_CS = """
#version 430

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform imageCube u_out_cube;

uniform samplerCube u_env_map;

uniform float u_roughness;
uniform int   u_sample_count;
uniform int   u_face_size;

uniform bool   u_exclude_enable;
uniform vec3  u_exclude_dir;       // normalized
uniform float u_exclude_cos;       // cos(theta)
uniform float u_exclude_feather;   // e.g. 0.02

const float PI = 3.14159265358979323846;

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u)  | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u)  | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u)  | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u)  | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint i, uint N) {
    return vec2(
        float(i) / float(N),
        radical_inverse_vdc(i)
    );
}

vec3 importance_sample_ggx(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;

    float phi = 2.0 * PI * Xi.x;
    float cos_theta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    vec3 H;
    H.x = cos(phi) * sin_theta;
    H.y = sin(phi) * sin_theta;
    H.z = cos_theta;

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sample_vec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sample_vec);
}

vec3 face_uv_to_dir(uint face, vec2 uv) {
    // uv in [0, 1]
    vec2 st = uv * 2.0 - 1.0;   // [-1, 1]
    float s = st.x;
    float t = st.y;

    if (face == 0u) {          // +X (right)
        return normalize(vec3( 1.0, -t, -s));
    } else if (face == 1u) {   // -X (left)
        return normalize(vec3(-1.0, -t,  s));
    } else if (face == 2u) {   // +Y (top)
        return normalize(vec3( s,  1.0,  t));
    } else if (face == 3u) {   // -Y (bottom)
        return normalize(vec3( s, -1.0, -t));
    } else if (face == 4u) {   // +Z (front)
        return normalize(vec3( s, -t,  1.0));
    } else {                   // -Z (back)
        return normalize(vec3(-s, -t, -1.0));
    }
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    int x    = gid.x;
    int y    = gid.y;
    int face = gid.z;

    if (x >= u_face_size || y >= u_face_size || face >= 6) {
        return;
    }

    vec2 uv = (vec2(x, y) + vec2(0.5)) / float(u_face_size);

    vec3 N = face_uv_to_dir(uint(face), uv);
    vec3 V = N;

    vec3 prefiltered = vec3(0.0);
    float total_weight = 0.0;

    uint sample_count = uint(max(u_sample_count, 1));

    for (uint i = 0u; i < sample_count; ++i) {
        vec2 Xi = hammersley(i, sample_count);
        vec3 H  = importance_sample_ggx(Xi, N, u_roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            vec3 sample_color = textureLod(u_env_map, L, 0.0).rgb;

            float keep = 1.0;
            if (u_exclude_enable) {
                float c = dot(L, u_exclude_dir);
                // c close to 1 means "towards the sun"
                float t = smoothstep(u_exclude_cos, min(1.0, u_exclude_cos + u_exclude_feather), c);
                keep = 1.0 - t;
            }
            sample_color *= keep;

            prefiltered += sample_color * NdotL;
            total_weight += NdotL;
        }
    }

    if (total_weight > 0.0) {
        prefiltered /= total_weight;
    }

    imageStore(u_out_cube, ivec3(x, y, face), vec4(prefiltered, 1.0));
}
"""


_IRRADIANCE_CS = """
#version 430

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform imageCube u_out_irradiance;

uniform samplerCube u_env_map;

uniform int u_face_size;
uniform int u_sample_count;

uniform bool   u_exclude_enable;
uniform vec3  u_exclude_dir;       // normalized
uniform float u_exclude_cos;       // cos(theta)
uniform float u_exclude_feather;   // e.g. 0.02

const float PI = 3.14159265358979323846;

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u)  | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u)  | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u)  | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u)  | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint i, uint N) {
    return vec2(
        float(i) / float(N),
        radical_inverse_vdc(i)
    );
}

vec3 face_uv_to_dir(uint face, vec2 uv) {
    // uv in [0, 1]
    vec2 st = uv * 2.0 - 1.0;   // [-1, 1]
    float s = st.x;
    float t = st.y;

    if (face == 0u) {          // +X (right)
        return normalize(vec3( 1.0, -t, -s));
    } else if (face == 1u) {   // -X (left)
        return normalize(vec3(-1.0, -t,  s));
    } else if (face == 2u) {   // +Y (top)
        return normalize(vec3( s,  1.0,  t));
    } else if (face == 3u) {   // -Y (bottom)
        return normalize(vec3( s, -1.0, -t));
    } else if (face == 4u) {   // +Z (front)
        return normalize(vec3( s, -t,  1.0));
    } else {                   // -Z (back)
        return normalize(vec3(-s, -t, -1.0));
    }
}

// cosine-weighted hemisphere sampling around N
vec3 sample_hemisphere_cosine(vec2 Xi, vec3 N) {
    float r = sqrt(Xi.x);
    float phi = 2.0 * PI * Xi.y;

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - x * x - y * y));

    vec3 H = vec3(x, y, z);

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 L = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(L);
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    int x    = gid.x;
    int y    = gid.y;
    int face = gid.z;

    if (x >= u_face_size || y >= u_face_size || face >= 6) {
        return;
    }

    vec2 uv = (vec2(x, y) + vec2(0.5)) / float(u_face_size);

    vec3 N = face_uv_to_dir(uint(face), uv);

    vec3 irradiance = vec3(0.0);

    uint sample_count = uint(max(u_sample_count, 1));

    for (uint i = 0u; i < sample_count; ++i) {
        vec2 Xi = hammersley(i, sample_count);
        vec3 L = sample_hemisphere_cosine(Xi, N);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            vec3 sample_color = textureLod(u_env_map, L, 0.0).rgb;
            float keep = 1.0;
            if (u_exclude_enable) {
                float c = dot(L, u_exclude_dir);
                // c close to 1 means "towards the sun"
                float t = smoothstep(u_exclude_cos, min(1.0, u_exclude_cos + u_exclude_feather), c);
                keep = 1.0 - t;
            }
            sample_color *= keep;
            
            // cosine-weighted sampling already includes cos(theta),
            // so irradiance ≈ π * average(L)
            irradiance += sample_color;
        }
    }

    irradiance *= PI / float(sample_count);

    imageStore(u_out_irradiance, ivec3(x, y, face), vec4(irradiance, 1.0));
}
"""