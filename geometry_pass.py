from dataclasses import dataclass
from typing import Optional, Tuple

import moderngl
from moderngl import Context, Texture, Program, VertexArray, Buffer
import numpy as np

from constants import TexUnit
from utils import Pass, safe_set_uniform
from gbuffer import GBuffer
from scene import Scene


@dataclass
class GeometryConfig:
    pass


class GeometryPass(Pass):

    def __init__(
        self,
        ctx: Context,
        load_program_fn,
        scene: Scene,
        vbo: Buffer,
        ibo: Buffer,
        config: Optional[GeometryConfig] = None,
    ):
        super().__init__(ctx, load_program_fn)
        self.scene = scene
        self.vbo = vbo
        self.ibo = ibo
        self.cfg = config or GeometryConfig()

        self.prog: Optional[Program] = None
        self.vao: Optional[VertexArray] = None

        self.albedo_tex = None
        self.normal_tex = None
        self.roughness_tex = None
        self.metallic_tex = None
        self.ao_tex = None

        self.use_albedo_tex = False
        self.use_normal_tex = False
        self.use_roughness_tex = False
        self.use_metallic_tex = False
        self.use_ao_tex = False 

        self.reload_shaders()
        self._load_material_textures()


    def reload_shaders(self) -> None:
        if isinstance(self.prog, Program):
            self.prog.release()

        self.prog = self.load_program_fn(
            vertex_shader="shaders/gbuffer.vert",
            fragment_shader="shaders/gbuffer.frag",
        )

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (
                    self.vbo,
                    "3f 3f 2f 3f",
                    "in_position",
                    "in_normal",
                    "in_uv",
                    "in_tangent",
                )
            ],
            self.ibo,
        )

        self._setup_material_samplers()

    def _setup_material_samplers(self) -> None:
        safe_set_uniform(self.prog, "u_albedo_map", TexUnit.ALBEDO_MAP)
        safe_set_uniform(self.prog, "u_normal_map", TexUnit.NORMAL_MAP)
        safe_set_uniform(self.prog, "u_roughness_map", TexUnit.ROUGHNESS_MAP)
        safe_set_uniform(self.prog, "u_metalness_map", TexUnit.METALNESS_MAP)
        safe_set_uniform(self.prog, "u_ao_map", TexUnit.AO_MAP)


    def _load_material_textures(self) -> None:
        mat = self.scene.material

        self.albedo_tex, self.use_albedo_tex = self._make_texture2d(mat.albedo_map, 3)
        self.normal_tex, self.use_normal_tex = self._make_texture2d(mat.normal_map, 3)
        self.roughness_tex, self.use_roughness_tex = self._make_texture2d(mat.roughness_map, 1)
        self.metallic_tex, self.use_metallic_tex = self._make_texture2d(mat.metalness_map, 1)
        self.ao_tex, self.use_ao_tex = self._make_texture2d(mat.ambient_occlusion_map, 1)

    def _make_texture2d(
        self,
        img: Optional[np.ndarray],
        channels: int = 3,
    ) -> Tuple[Texture, bool]:
        to_use = img is not None
        if img is None:
            img = np.ones((1, 1, channels), dtype="f4")
        else:
            img = np.ascontiguousarray(img.astype("f4"))

        h, w = img.shape[:2]
        components = 1 if img.ndim == 2 else img.shape[2]

        tex = self.ctx.texture(
            (w, h),
            components=components,
            data=img.tobytes(),
            dtype="f4",
        )
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        return tex, to_use

    def _update_material_uniforms(self) -> None:
        safe_set_uniform(self.prog, "u_use_albedo_map", self.use_albedo_tex)
        safe_set_uniform(self.prog, "u_use_normal_map", self.use_normal_tex)
        safe_set_uniform(self.prog, "u_use_roughness_map", self.use_roughness_tex)
        safe_set_uniform(self.prog, "u_use_metalness_map", self.use_metallic_tex)
        safe_set_uniform(self.prog, "u_use_ao_map", self.use_ao_tex)
        safe_set_uniform(self.prog, "u_albedo", self.scene.material.albedo)
        safe_set_uniform(self.prog, "u_roughness", self.scene.material.roughness)
        safe_set_uniform(self.prog, "u_metalness", self.scene.material.metalness)


    def render(
        self,
        gbuffer: GBuffer,
        model_matrix: np.ndarray,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        time_value: float,
    ) -> None:
        gbuffer.fbo.use()
        self.ctx.viewport = (0, 0, gbuffer.width, gbuffer.height)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.prog["u_model"].write(np.asarray(model_matrix, dtype="f4").tobytes())
        self.prog["u_view"].write(np.asarray(view_matrix, dtype="f4").tobytes())
        self.prog["u_projection"].write(np.asarray(projection_matrix, dtype="f4").tobytes())
        normal_matrix = np.linalg.inv(model_matrix).T[:3, :3]
        self.prog["u_normal_matrix"].write(np.asarray(normal_matrix, dtype="f4").tobytes())

        self._update_material_uniforms()
        safe_set_uniform(self.prog, "u_time", time_value)

        self.albedo_tex.use(location=TexUnit.ALBEDO_MAP)
        self.normal_tex.use(location=TexUnit.NORMAL_MAP)
        self.roughness_tex.use(location=TexUnit.ROUGHNESS_MAP)
        self.metallic_tex.use(location=TexUnit.METALNESS_MAP)
        self.ao_tex.use(location=TexUnit.AO_MAP)

        self.vao.render()
