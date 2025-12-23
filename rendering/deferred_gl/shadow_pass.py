from dataclasses import dataclass
from typing import Optional

import moderngl
import numpy as np
from moderngl import Buffer, Context, Framebuffer, Program, Texture, VertexArray

from pyrr import matrix44, vector3, matrix33

from core.constants import EPSILON, UP
from core.scene import DirectionalLight

from rendering.deferred_gl.utils import RenderPass, safe_set_uniform
from rendering.deferred_gl.gl_scene import GLMesh

@dataclass
class ShadowSettings:
    size: int = 4096        
    margin: float = 0.05    
    cull_face: str = "front" 
    min_near: float = 0.05  


class ShadowPass(RenderPass):

    def __init__(
        self,
        ctx: Context,
        load_program_fn,
        mesh: GLMesh,
        settings: Optional[ShadowSettings] = None,
    ) -> None:
        super().__init__(ctx, load_program_fn)
        self.settings = settings or ShadowSettings()

        s = int(self.settings.size)
        self.depth_tex: Texture = self.ctx.depth_texture((s, s))
        self.depth_tex.compare_func = '<='
        self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_tex.repeat_x = False
        self.depth_tex.repeat_y = False

        self.fbo: Framebuffer = self.ctx.framebuffer(depth_attachment=self.depth_tex)

        self.prog = self.load_program_fn(
            vertex_shader='shaders/shadow.vert',
            fragment_shader='shaders/empty.frag',
        )

        self.vao: VertexArray = self.ctx.vertex_array(
            self.prog,
            [
                (mesh.vbo, "3f 32x", "in_position")
            ],
            mesh.ibo,
        )

    def resize(self, w, h):
        pass

    def render(self, model_matrix: np.ndarray, dir_light: DirectionalLight, env_matrix:Optional[np.ndarray]=None) -> DirectionalLight:
        light_dir = np.asarray(dir_light.direction, dtype=np.float32)
        light_dir = light_dir / (np.linalg.norm(light_dir) + EPSILON)

        if env_matrix is not None:
            light_dir = matrix33.apply_to_vector(env_matrix.T, light_dir)
        light_view_proj = self._compute_light_view_proj(light_dir)

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.settings.size, self.settings.size)

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.cull_face = self.settings.cull_face
        
        self.fbo.clear(depth=1.0)

        safe_set_uniform(self.prog, "u_model", model_matrix)
        safe_set_uniform(self.prog, "u_lightViewProj", light_view_proj)

        self.vao.render()


        return DirectionalLight(color=dir_light.color, direction=light_dir, view_proj=light_view_proj)

    def _compute_light_view_proj(self, light_dir: np.ndarray) -> np.ndarray:
        R = float(np.sqrt(3.0) * (1.0 + self.settings.margin))  

        up = UP
        if abs(float(vector3.dot(up, light_dir))) > 0.99:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        dist = 2.0 * R
        eye = - light_dir * dist

        view = matrix44.create_look_at(eye, (0,0,0), up, dtype=np.float32)

        near = max(float(self.settings.min_near), dist - R)
        far  = dist + R

        proj = matrix44.create_orthogonal_projection(-R, R, -R, R, near, far, dtype=np.float32)
        return matrix44.multiply(view, proj)

    def release(self):
        self.depth_tex.release()
        self.fbo.release()
        self.vao.release()