from typing import Optional, Tuple

import moderngl
from moderngl import Context, Program, VertexArray, Texture, TextureCube, Framebuffer
import numpy as np

from constants import TexUnit, TONE_MAPPING_IDS
from utils import RenderPass, safe_set_uniform
from gbuffer import GBuffer
from scene import EnvMap, PointLight, DirectionalLight
from sun_extraction import SunExtraction
from ibl import EnvironmentMapPrecomputer

class LightingPass(RenderPass):
    def __init__(
        self,
        ctx: Context,
        load_program_fn,
        envmap:Optional[EnvMap],
        to_exclude_sun:Optional[SunExtraction]
    ):
        super().__init__(ctx, load_program_fn)
        
        self.background_tex:Optional[TextureCube] = None
        self.irradiance_tex:Optional[TextureCube] = None
        self.specular_tex:Optional[TextureCube] = None
        self.num_specular_mips = 0
        if envmap is not None:
            precomp = EnvironmentMapPrecomputer(self.ctx)
            env_tex = envmap.to_gl(self.ctx)
            (
                self.background_tex,
                self.irradiance_tex,
                self.specular_tex,
                self.num_specular_mips,
            ) = precomp(env_tex, release=True, to_exclude_sun=to_exclude_sun)

        self.prog: Optional[Program] = None
        self.vao: Optional[VertexArray] = None

        self.tex: Optional[Texture] = None
        self.fbo: Optional[Framebuffer] = None

        self.reload_shaders()
    
    @property
    def output_texture(self) -> Optional[Texture]:
        return self.tex
    
    def release(self) -> None:
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None

        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
        if self.tex is not None:
            self.tex.release()
            self.tex = None

        for t in (self.background_tex, self.irradiance_tex, self.specular_tex):
            if t is not None:
                t.release()
        self.background_tex = None
        self.irradiance_tex = None
        self.specular_tex = None
        self.num_specular_mips = 0

    def reload_shaders(self) -> None:
        if self.prog is not None:
            self.prog.release()
        if self.vao is not None:
            self.vao.release()

        self.prog = self.load_program_fn(
            vertex_shader="shaders/deferred_lighting.vert",
            fragment_shader="shaders/deferred_lighting.frag",
        )
        self.vao = self.ctx.vertex_array(self.prog, [])

        safe_set_uniform(self.prog, "gPosition", TexUnit.GBUFFER_POSITION)
        safe_set_uniform(self.prog, "gNormal", TexUnit.GBUFFER_NORMAL)
        safe_set_uniform(self.prog, "gAlbedo", TexUnit.GBUFFER_ALBEDO)
        safe_set_uniform(self.prog, "gRMAOS", TexUnit.GBUFFER_RMAOS)
        safe_set_uniform(self.prog, "gEmissive", TexUnit.GBUFFER_EMISSIVE)
        safe_set_uniform(self.prog, "u_ssao", TexUnit.SSAO_BLUR)
        safe_set_uniform(self.prog, "u_background_env", TexUnit.ENV_BACKGROUND)
        safe_set_uniform(self.prog, "u_irradiance_env", TexUnit.ENV_IRRADIANCE) 
        safe_set_uniform(self.prog, "u_specular_env", TexUnit.ENV_SPECULAR)
        safe_set_uniform(self.prog, "u_shadowMap", TexUnit.SHADOW_MAP)

    
    def resize(self, width: int, height: int) -> None:
        # if width <= 0 or height <= 0:
        #     return

        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
        if self.tex is not None:
            self.tex.release()
            self.tex = None

        self.tex = self.ctx.texture((width, height), components=4, dtype="f2")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = False
        self.tex.repeat_y = False

        self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])   

    def render(
        self,
        gbuffer: GBuffer,
        ssao_tex: Texture,
        shadow_tex: Texture,
        point_light: Optional[PointLight],
        dir_light: Optional[DirectionalLight],
        eye_pos: np.ndarray,
        inv_view: np.ndarray,
        inv_proj: np.ndarray,
        env_matrix:np.ndarray,
        env_lod_factor:float,
        use_ssao: bool,
        specular_tint:float,
        time_value: float,
        window_size: Tuple[int, int],
    ) -> None:
        
        self.fbo.use()
        w, h = window_size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.02, 0.02, 0.02, 1.0)

        gbuffer.position.use(location=TexUnit.GBUFFER_POSITION)
        gbuffer.normal.use(location=TexUnit.GBUFFER_NORMAL)
        gbuffer.albedo.use(location=TexUnit.GBUFFER_ALBEDO)
        gbuffer.rmaos.use(location=TexUnit.GBUFFER_RMAOS)
        gbuffer.emissive.use(location=TexUnit.GBUFFER_EMISSIVE)
     

        safe_set_uniform(self.prog, "u_viewPos", tuple(eye_pos))

        safe_set_uniform(self.prog, "u_specularTint", min(1, max(0, specular_tint)))
        safe_set_uniform(self.prog, "u_time", time_value)
        
        safe_set_uniform(self.prog, "u_use_ssao", use_ssao)
        if use_ssao:
            ssao_tex.use(location=TexUnit.SSAO_BLUR)

        use_env = self.irradiance_tex is not None and self.specular_tex is not None
        safe_set_uniform(self.prog, "u_use_env", use_env)
        if use_env:
            self.background_tex.use(location=TexUnit.ENV_BACKGROUND)
            self.irradiance_tex.use(location=TexUnit.ENV_IRRADIANCE)
            self.specular_tex.use(location=TexUnit.ENV_SPECULAR)
            safe_set_uniform(self.prog, "u_invView", inv_view.astype("f4"))
            safe_set_uniform(self.prog, "u_invProj", inv_proj.astype("f4"))
            safe_set_uniform(self.prog, "u_envRotation", env_matrix.astype("f4"))
            safe_set_uniform(self.prog, "u_num_specular_mips", self.num_specular_mips)
            safe_set_uniform(self.prog, "u_env_lod", self.num_specular_mips * env_lod_factor)

        use_point_light = point_light is not None
        safe_set_uniform(self.prog, "u_use_point_light", use_point_light)
        if use_point_light:
            safe_set_uniform(self.prog, "u_pointLightPos", point_light.position)
            safe_set_uniform(self.prog, "u_pointLightColor", point_light.color)

        use_dir_light = dir_light is not None
        safe_set_uniform(self.prog, "u_use_dir_light", use_dir_light)
        if use_dir_light:
            shadow_tex.use(location=TexUnit.SHADOW_MAP)
            safe_set_uniform(self.prog, "u_dirLightDir", dir_light.direction)
            safe_set_uniform(self.prog, "u_dirLightColor", dir_light.color)
            safe_set_uniform(self.prog, "u_lightViewProj", dir_light.view_proj.astype("f4"))
            safe_set_uniform(self.prog, "u_shadow_bias", 0.012)
            safe_set_uniform(self.prog, "u_shadow_strength", 1.0)

        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)
