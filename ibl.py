from dataclasses import dataclass
from typing import Optional
import math

import moderngl
from moderngl import Context, Texture, TextureCube, ComputeShader


@dataclass
class PrefilterSettings:
    background_size: int = 1024
    num_mips: Optional[int] = None
    
    specular0_size: int = 512
    specular_sample_count: int = 1024

    irradiance_size: int = 32
    irradiance_sample_count: int = 1024*16


class EnvironmentMapPrecomputer:

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._pano_to_cube_cs: Optional[ComputeShader] = None
        self._prefilter_cs: Optional[ComputeShader] = None
        self._irradiance_cs: Optional[moderngl.ComputeShader] = None

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
        # cube.build_mipmaps()
        # cube.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # cube.repeat_x = True
        # cube.repeat_y = True

        self._dispatch_panorama_to_cubemap(panorama, cube, cube_size)
        if release:
            panorama.release()

        return cube

    def __call__(self, background_cube:Texture|TextureCube, 
                settings: PrefilterSettings | None = None,
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
    ) -> None:
        cs = self._irradiance_cs

        src_env.use(location=0)
        cs["u_env_map"].value = 0
        cs["u_face_size"].value = size
        cs["u_sample_count"].value = int(sample_count)

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
    ) -> None:
        cs = self._prefilter_cs

        src_env.use(location=0)
        cs["u_env_map"].value = 0
        cs["u_sample_count"].value = int(sample_count)

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
            // cosine-weighted sampling already includes cos(theta),
            // so irradiance ≈ π * average(L)
            irradiance += sample_color;
        }
    }

    irradiance *= PI / float(sample_count);

    imageStore(u_out_irradiance, ivec3(x, y, face), vec4(irradiance, 1.0));
}
"""
# if __name__ == '__main__':
#     from pathlib import Path
#     import math

#     import numpy as np
#     from PIL import Image
#     import moderngl

#     cube_map_path = Path('resources/cubemaps/learnopengl')
#     face_names = ["right", "left", "top", "bottom", "front", "back"]

#     # ------------------------------------------------------------------
#     # Utility: tonemap + debug print
#     # ------------------------------------------------------------------

#     def _hdr_to_uint8(arr: np.ndarray, label: str) -> np.ndarray:
#         """
#         - Replace NaN / Inf
#         - Print min / max for debugging
#         - Normalize to [0, 1] so we can *see* something in PNG
#         - Apply simple gamma and convert to uint8
#         """
#         arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

#         if arr.ndim == 3 and arr.shape[2] > 3:
#             arr = arr[..., :3]

#         vmin = float(arr.min())
#         vmax = float(arr.max())
#         print(f"[{label}] range = {vmin:.6g} .. {vmax:.6g}")

#         # Avoid division by zero; if the tex is completely black, just return black
#         if vmax > 1e-8:
#             arr = arr / vmax

#         # Clamp + gamma to approximate sRGB
#         arr = np.clip(arr, 0.0, 1.0)
#         #arr = np.power(arr, 1.0 / 2.2)
        

#         img = (arr * 255.0 + 0.5).astype(np.uint8)
#         return img

#     # ------------------------------------------------------------------
#     # Load LearnOpenGL-style cubemap
#     # ------------------------------------------------------------------

#     def load_learnopengl_cubemap(ctx: moderngl.Context, folder: Path) -> TextureCube:
#         exts = [".hdr", ".exr", ".png", ".jpg", ".jpeg"]
#         images = []

#         for name in face_names:
#             path = None
#             for ext in exts:
#                 candidate = folder / f"{name}{ext}"
#                 if candidate.exists():
#                     path = candidate
#                     break
#             if path is None:
#                 raise FileNotFoundError(
#                     f"Could not find face '{name}' in {folder} (tried {exts})"
#                 )

#             img = Image.open(path).convert("RGB")
#             arr = np.asarray(img, dtype=np.float32) / 255.0
#             images.append(arr)

#         h, w, c = images[0].shape
#         for name, arr in zip(face_names, images):
#             if arr.shape != images[0].shape:
#                 raise ValueError(
#                     f"Face {name} has size {arr.shape}, expected {(h, w, c)}"
#                 )

#         cube = ctx.texture_cube((w, h), components=3, dtype="f2")
#         cube.repeat_x = True
#         cube.repeat_y = True
#         cube.filter = (moderngl.LINEAR, moderngl.LINEAR)

#         for face_idx, arr in enumerate(images):
#             data = arr.astype("float16").tobytes()
#             cube.write(face=face_idx, data=data)

#         cube.build_mipmaps()
#         return cube

#     # ------------------------------------------------------------------
#     # Save one level of a cubemap
#     # ------------------------------------------------------------------

#     def save_cubemap_level(
#         tex: TextureCube,
#         base_size: int,
#         level: int,
#         out_dir: Path,
#         label_prefix: str,
#     ) -> None:
#         out_dir.mkdir(parents=True, exist_ok=True)

#         size = max(1, base_size >> level)
#         w = h = size

#         for face_idx, name in enumerate(face_names):
#             raw = tex.read(face=face_idx, alignment=1)
#             arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
#             arr = arr.reshape((h, w, tex.components))

#             img = _hdr_to_uint8(arr, f"{label_prefix}[level={level}, face={name}]")
#             Image.fromarray(img, mode="RGB").save(out_dir / f"{name}.png")

#     # ------------------------------------------------------------------
#     # Save specular mips by sampling each LOD into a 2D texture
#     # ------------------------------------------------------------------

#     def save_specular_mips(
#         ctx: moderngl.Context,
#         tex: TextureCube,
#         base_size: int,
#         max_mips: int,
#         root_dir: Path,
#     ) -> None:
#         root_dir.mkdir(parents=True, exist_ok=True)

#         vs_src = """
#         #version 330
#         out vec2 v_uv;
#         const vec2 POS[3] = vec2[3](
#             vec2(-1.0, -1.0),
#             vec2( 3.0, -1.0),
#             vec2(-1.0,  3.0)
#         );
#         const vec2 UV[3] = vec2[3](
#             vec2(0.0, 0.0),
#             vec2(2.0, 0.0),
#             vec2(0.0, 2.0)
#         );
#         void main() {
#             gl_Position = vec4(POS[gl_VertexID], 0.0, 1.0);
#             v_uv = UV[gl_VertexID];
#         }
#         """

#         fs_src = """
#         #version 330
#         in vec2 v_uv;
#         out vec4 fragColor;

#         uniform samplerCube u_env;
#         uniform int u_face;
#         uniform float u_lod;

#         vec3 face_uv_to_dir(uint face, vec2 uv) {
#             // uv in [0, 1]
#             vec2 st = uv * 2.0 - 1.0;   // [-1, 1]
#             float s = st.x;
#             float t = st.y;

#             if (face == 0u) {          // +X (right)
#                 return normalize(vec3( 1.0, -t, -s));
#             } else if (face == 1u) {   // -X (left)
#                 return normalize(vec3(-1.0, -t,  s));
#             } else if (face == 2u) {   // +Y (top)
#                 return normalize(vec3( s,  1.0,  t));
#             } else if (face == 3u) {   // -Y (bottom)
#                 return normalize(vec3( s, -1.0, -t));
#             } else if (face == 4u) {   // +Z (front)
#                 return normalize(vec3( s, -t,  1.0));
#             } else {                   // -Z (back)
#                 return normalize(vec3(-s, -t, -1.0));
#             }
#         }

#         void main() {
#             vec2 uv = v_uv;
#             vec3 dir = face_uv_to_dir(uint(u_face), uv);
#             vec3 c = textureLod(u_env, dir, u_lod).rgb;
#             fragColor = vec4(c, 1.0);
#         }
#         """

#         prog = ctx.program(vertex_shader=vs_src, fragment_shader=fs_src)
#         vao = ctx.vertex_array(prog, [])
#         tex.use(location=0)
#         prog["u_env"].value = 0

#         for level in range(max_mips):
#             size = max(1, base_size >> level)
#             level_dir = root_dir / str(level)
#             level_dir.mkdir(parents=True, exist_ok=True)

#             rt_tex = ctx.texture((size, size), components=3, dtype="f2")
#             fbo = ctx.framebuffer(color_attachments=[rt_tex])

#             for face_idx, name in enumerate(face_names):
#                 fbo.use()
#                 ctx.viewport = (0, 0, size, size)
#                 ctx.disable(moderngl.DEPTH_TEST)
#                 fbo.clear(0.0, 0.0, 0.0, 1.0)

#                 prog["u_face"].value = face_idx
#                 prog["u_lod"].value = float(level)

#                 vao.render(mode=moderngl.TRIANGLES, vertices=3)

#                 raw = rt_tex.read(alignment=1)
#                 arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
#                 arr = arr.reshape((size, size, rt_tex.components))

#                 img = _hdr_to_uint8(arr, f"specular[level={level}, face={name}]")
#                 Image.fromarray(img, mode="RGB").save(level_dir / f"{name}.png")

#             fbo.release()
#             rt_tex.release()

#     # ------------------------------------------------------------------
#     # Context + IBL compute
#     # ------------------------------------------------------------------

#     ctx = moderngl.create_standalone_context(require=430)

#     print(f"Loading base cubemap from: {cube_map_path}")
#     base_cube = load_learnopengl_cubemap(ctx, cube_map_path)

#     base_size = base_cube.size[0]
#     settings = PrefilterSettings(
#         cube_size=base_size,
#         num_mips=None,               # auto
#         specular_sample_count=256,
#         irradiance_size=32,
#         irradiance_sample_count=128,
#     )

#     precomp = EnvironmentMapPrecomputer(ctx)
#     base, irradiance, specular = precomp.compute(base_cube, settings)

#     # ------------------------------------------------------------------
#     # Save base cubemap (to verify loading is OK)
#     # ------------------------------------------------------------------

#     base_dir = cube_map_path / "base"
#     print(f"Saving base cubemap to: {base_dir}")
#     save_cubemap_level(base, base_size, level=0, out_dir=base_dir, label_prefix="base")

#     # ------------------------------------------------------------------
#     # Save irradiance cubemap (single level)
#     # ------------------------------------------------------------------

#     irr_dir = cube_map_path / "irradiance"
#     print(f"Saving irradiance cubemap to: {irr_dir}")
#     save_cubemap_level(
#         irradiance,
#         settings.irradiance_size,
#         level=0,
#         out_dir=irr_dir,
#         label_prefix="irradiance",
#     )

#     # ------------------------------------------------------------------
#     # Save specular prefiltered cubemap mips
#     # ------------------------------------------------------------------

#     spec_dir = cube_map_path / "specular"
#     max_mips = settings.num_mips or int(math.floor(math.log2(base_size))) + 1
#     print(f"Saving specular prefiltered cubemap mips to: {spec_dir}")
#     print(f"Specular base size: {base_size}, mips: {max_mips}")
#     save_specular_mips(ctx, specular, base_size, max_mips, spec_dir)

#     print("Done.")
