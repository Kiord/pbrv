from typing import Optional, Tuple

import moderngl
from moderngl import Context, Program, VertexArray, Texture, TextureCube
import numpy as np

from constants import TexUnit, TONE_MAPPING_IDS
from utils import Pass, safe_set_uniform
from gbuffer import GBuffer
from scene import EnvMap, PointLight, DirectionalLight
from ibl import EnvironmentMapPrecomputer

class LightingPass(Pass):
    def __init__(
        self,
        ctx: Context,
        load_program_fn,
        envmap:Optional[EnvMap],
    ):
        super().__init__(ctx, load_program_fn)
        
        self.background_tex:Optional[TextureCube] = None
        self.irradiance_tex:Optional[TextureCube] = None
        self.specular_tex:Optional[TextureCube] = None
        self.num_specular_mips = 0
        if envmap is not None:
            precomp = EnvironmentMapPrecomputer(self.ctx)
            env_tex = envmap.to_gl(self.ctx)
            self.background_tex, self.irradiance_tex, self.specular_tex, self.num_specular_mips = precomp(env_tex, release=True)

        self.prog: Optional[Program] = None
        self.vao: Optional[VertexArray] = None

        self.reload_shaders()
        
    def reload_shaders(self) -> None:
        if isinstance(self.prog, Program):
            self.prog.release()

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

        

    def render(
        self,
        gbuffer: GBuffer,
        ssao_tex: Texture,
        shadow_tex: Texture,
        point_light: Optional[PointLight],
        dir_light: Optional[DirectionalLight],
        eye_pos: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        env_matrix:np.ndarray,
        env_lod_factor:float,
        use_ssao: bool,
        specular_tint:float,
        tone_mapping: str,
        exposure: float,
        time_value: float,
        window_size: Tuple[int, int],
    ) -> None:
        self.ctx.screen.use()
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
        tone_mapping_id = TONE_MAPPING_IDS.get(tone_mapping, 0)
        safe_set_uniform(self.prog, "u_tone_mapping_id", tone_mapping_id)
        safe_set_uniform(self.prog, "u_exposure", exposure)
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
            safe_set_uniform(self.prog, "u_invView", np.linalg.inv(view_matrix).astype("f4"))
            safe_set_uniform(self.prog, "u_invProj", np.linalg.inv(proj_matrix).astype("f4"))
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
