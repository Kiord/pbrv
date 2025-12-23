from dataclasses import dataclass
import moderngl
from moderngl import Program, Texture, Framebuffer, VertexArray
import numpy as np
from core.constants import EPSILON
from typing import Callable, Optional

from rendering.deferred_gl.utils import safe_set_uniform, RenderPass, TexUnit

@dataclass
class SSAOConfig:
    kernel_size: int = 32
    noise_dim: int = 4
    radius: float = 0.1
    intensity: float = 1.0
    blur_depth_sigma: float = 4.0
    blur_normal_sigma: float = 32.0


class SSAOPass(RenderPass):
    def __init__(self, ctx: moderngl.Context, load_program_fn:Callable, config: Optional[SSAOConfig]=None):
        super().__init__(ctx, load_program_fn)
        if config is None:
            config = SSAOConfig()

        self.cfg = config 
        self._init_kernel_and_noise()
        
        self.prog:Optional[Program] = None
        self.blur_prog:Optional[Program] = None
        self.vao:Optional[VertexArray] = None
        self.blur_vao:Optional[VertexArray] = None
        self.reload_shaders()
        
        self.tex: Optional[Texture] = None
        self.blur_tex: Optional[Texture] = None
        self.fbo:Optional[Framebuffer] = None
        self.blur_fbo:Optional[Framebuffer] = None


    def resize(self, width, height):
        if self.tex is not None:
            self.tex.release()
        if self.blur_tex is not None:
            self.blur_tex.release()
        if self.fbo is not None:
            self.fbo.release()
        if self.blur_fbo is not None:
            self.blur_fbo.release()


        half_w = max(1, width // 2)
        half_h = max(1, height // 2)

        self.tex = self.ctx.texture((half_w, half_h), 1, dtype='f1')
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])

        self.blur_tex = self.ctx.texture((half_w, half_h), 1, dtype='f1')
        self.blur_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_tex])

    def reload_shaders(self):
        if self.prog is not None:
            self.prog.release()
        if self.blur_prog is not None:
            self.blur_prog.release()
        if self.vao is not None:
            self.vao.release()
        if self.blur_vao is not None:
            self.blur_vao.release()

        self.prog = self.load_program_fn(
            vertex_shader='shaders/screen.vert',
            fragment_shader='shaders/ssao.frag',
        )
        self.blur_prog = self.load_program_fn(
            vertex_shader='shaders/screen.vert',
            fragment_shader='shaders/ssao_blur.frag',
        )
        
        self.vao      = self.ctx.vertex_array(self.prog, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])


        safe_set_uniform(self.prog, 'gPosition', TexUnit.GBUFFER_POSITION)
        safe_set_uniform(self.prog, 'gNormal', TexUnit.GBUFFER_NORMAL) 
        safe_set_uniform(self.prog, 'u_ssao_noise', TexUnit.SSAO_NOISE)

        safe_set_uniform(self.blur_prog, 'u_ssao',    TexUnit.SSAO)
        safe_set_uniform(self.blur_prog, 'gPosition', TexUnit.GBUFFER_POSITION)
        safe_set_uniform(self.blur_prog, 'gNormal',   TexUnit.GBUFFER_NORMAL)

    def render(self, g_position:moderngl.Texture, g_normal:moderngl.Texture, view, proj):
        # half-res viewport
        w, h = self.tex.size
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)  # AO=1 (no occlusion) base

        # G-buffer (full res) as input
        g_position.use(location=TexUnit.GBUFFER_POSITION)
        g_normal.use(location=TexUnit.GBUFFER_NORMAL)
        self.noise_tex.use(location=TexUnit.SSAO_NOISE)

        # matrices
        self.prog['u_view'].write(np.asarray(view, dtype='f4').tobytes())
        self.prog['u_projection'].write(np.asarray(proj, dtype='f4').tobytes())

        noise_scale = (
            w / float(self.cfg.noise_dim),
            h / float(self.cfg.noise_dim),
        )

        safe_set_uniform(self.prog, 'u_ssao_noise_scale',  noise_scale)
        safe_set_uniform(self.prog, 'u_ssao_radius',       self.cfg.radius)
        safe_set_uniform(self.prog, 'u_ssao_sample_count', self.cfg.kernel_size)
        safe_set_uniform(self.prog, 'u_ssao_intensity',    self.cfg.intensity)

        # upload kernel array
        self.prog['u_ssao_samples'].write(self.kernel.astype('f4').tobytes())

        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)

    def blur(self, g_position:moderngl.Texture, g_normal:moderngl.Texture):
        w, h = self.blur_tex.size
        self.blur_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        # raw SSAO as input, full-res G-buffer for edge awareness
        self.tex.use(location=TexUnit.SSAO)
        g_position.use(location=TexUnit.GBUFFER_POSITION)
        g_normal.use(location=TexUnit.GBUFFER_NORMAL)


        safe_set_uniform(self.blur_prog, 'u_blur_depth_sigma',  self.cfg.blur_depth_sigma)
        safe_set_uniform(self.blur_prog, 'u_blur_normal_sigma', self.cfg.blur_normal_sigma)

        self.blur_vao.render(mode=moderngl.TRIANGLES, vertices=3)

    @property
    def output_texture(self):
        return self.blur_tex
    

    def _init_kernel_and_noise(self):

        samples = np.empty((self.cfg.kernel_size, 3), dtype=np.float32)
        samples[:, :2] = np.random.uniform(-1.0, 1.0, size=(self.cfg.kernel_size, 2))
        samples[:, 2] = np.random.uniform(0.0, 1.0, size=(self.cfg.kernel_size,))

        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms[norms < EPSILON] = 1.0
        samples = samples / norms

        scale = np.linspace(0, 1, self.cfg.kernel_size, dtype=np.float32)
        scale = 0.1 + 0.9 * (scale * scale)  # bias towards the center
        self.kernel = samples * scale[:, None]

        noise_hw = (self.cfg.noise_dim, self.cfg.noise_dim)
        noise = np.zeros((*noise_hw , 3), dtype=np.float32)
        noise[:, :, :2] = np.random.uniform(-1.0, 1.0, size=(*noise_hw, 2))

        self.noise_tex = self.ctx.texture(
            noise_hw,
            components=3,
            data=noise.astype('f4').tobytes(),
            dtype='f4',
        )
        self.noise_tex.repeat_x = True
        self.noise_tex.repeat_y = True
        self.noise_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)


    def release(self):
        self.tex.release() 
        self.fbo.release()
        self.blur_tex.release()
        self.blur_fbo.release()
        self.noise_tex.release()
        self.fbo.release()
        self.vao.release()
        self.blur_vao.release()
        self.fbo.release()
        self.blur_fbo.release()
        self.prog.release()
        self.blur_prog.release()