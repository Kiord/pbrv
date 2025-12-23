from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import moderngl
from moderngl import Context, Program, Texture, Framebuffer, VertexArray

from core.constants import TONE_MAPPING_IDS

from rendering.deferred_gl.utils import RenderPass, safe_set_uniform

@dataclass
class BloomConfig:
    downsample: int = 2  

    emissive_boost: float = 5.0
    hdr_boost: float = 0.25

    blur_iterations: int = 3
    
    # Composite
    intensity: float = 0.15
    exposure: float = 1.0
    gamma: float = 2.2


class PostProcessingPass(RenderPass):
    """
      1) prefilter (bright-pass from HDR + emissive boost) -> seed tex (half-res)
      2) ping-pong blur (separable) -> bloom tex (half-res)
      3) composite (hdr + bloom) + tonemap/gamma -> screen
    """

    def __init__(
        self,
        ctx: Context,
        load_program_fn: Callable[..., Program],
        config: Optional[BloomConfig] = None,
    ) -> None:
        super().__init__(ctx, load_program_fn)
        self.cfg = config or BloomConfig()

        self.prefilter_prog: Optional[Program] = None
        self.blur_prog: Optional[Program] = None
        self.composite_prog: Optional[Program] = None

        self.prefilter_vao: Optional[VertexArray] = None
        self.blur_vao: Optional[VertexArray] = None
        self.composite_vao: Optional[VertexArray] = None

        self.seed_tex: Optional[Texture] = None
        self.seed_fbo: Optional[Framebuffer] = None

        self.pingpong_tex: Tuple[Optional[Texture]] = [None, None]
        self.pingpong_fbo: Tuple[Optional[Framebuffer]] = [None, None]

        self._bloom_size: Tuple[int, int] = (0, 0)

        self.reload_shaders()

    def release(self) -> None:
        for prog in (self.prefilter_prog, self.blur_prog, self.composite_prog):
            if prog is not None:
                prog.release()

        for vao in (self.prefilter_vao, self.blur_vao, self.composite_vao):
            if vao is not None:
                vao.release()

        for fbo in (self.seed_fbo, self.pingpong_fbo[0], self.pingpong_fbo[1]):
            if fbo is not None:
                fbo.release()

        for tex in (self.seed_tex, self.pingpong_tex[0], self.pingpong_tex[1]):
            if tex is not None:
                tex.release()

        self.prefilter_prog = self.blur_prog = self.composite_prog = None
        self.prefilter_vao = self.blur_vao = self.composite_vao = None
        self.seed_tex = None
        self.seed_fbo = None
        self.pingpong_tex = (None, None)
        self.pingpong_fbo = (None, None)
        self._bloom_size = (0, 0)

    def reload_shaders(self) -> None:
        for prog in (self.prefilter_prog, self.blur_prog, self.composite_prog):
            if prog is not None:
                prog.release()
        for vao in (self.prefilter_vao, self.blur_vao, self.composite_vao):
            if vao is not None:
                vao.release()

        self.prefilter_prog = self.load_program_fn(
            vertex_shader='shaders/screen.vert', 
            fragment_shader='shaders/bloom_prefilter.frag')
        self.blur_prog = self.load_program_fn(
            vertex_shader='shaders/screen.vert', 
            fragment_shader='shaders/bloom_blur.frag')
        self.composite_prog = self.load_program_fn(
            vertex_shader='shaders/screen.vert', 
            fragment_shader='shaders/composite.frag')


        self.prefilter_vao = self.ctx.vertex_array(self.prefilter_prog, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])
        self.composite_vao = self.ctx.vertex_array(self.composite_prog, [])

        safe_set_uniform(self.prefilter_prog, "u_hdr", 0)
        safe_set_uniform(self.prefilter_prog, "u_emissive", 1)

        safe_set_uniform(self.blur_prog, "u_src", 0)

        safe_set_uniform(self.composite_prog, "u_hdr", 0)
        safe_set_uniform(self.composite_prog, "u_bloom", 1)


    def _make_tex_rgba16f(self, size: Tuple[int, int]) -> Texture:
        tex = self.ctx.texture(size=size, components=4, dtype="f2")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def resize(self, width: int, height: int) -> None:
        ds = max(1, int(self.cfg.downsample))
        bw = max(1, width // ds)
        bh = max(1, height // ds)

        if (bw, bh) == self._bloom_size and self.seed_tex is not None:
            return

        for fbo in (self.seed_fbo, self.pingpong_fbo[0], self.pingpong_fbo[1]):
            if fbo is not None:
                fbo.release()
        for tex in (self.seed_tex, self.pingpong_tex[0], self.pingpong_tex[1]):
            if tex is not None:
                tex.release()

        self._bloom_size = (bw, bh)

        self.seed_tex = self._make_tex_rgba16f((bw, bh))
        self.seed_fbo = self.ctx.framebuffer(color_attachments=[self.seed_tex])

        self.pingpong_tex = (self._make_tex_rgba16f((bw, bh)), self._make_tex_rgba16f((bw, bh)))
        self.pingpong_fbo = (self.ctx.framebuffer(color_attachments=[self.pingpong_tex[0]]),
                             self.ctx.framebuffer(color_attachments=[self.pingpong_tex[1]]))

    @property
    def bloom_texture(self) -> Optional[Texture]:
        return self.pingpong_tex[1]

    def render(
        self,
        input_tex: Texture,
        emissive_tex: Texture,
        tone_mapping: str,
        exposure: float,
        time_value:float,
        window_size: Tuple[int, int],
        *,
        emissive_boost: Optional[float] = None,
        hdr_boost: Optional[float] = None,
        blur_iterations: Optional[int] = None,
        intensity: Optional[float] = None,
        
    ) -> None:

        ebo = self.cfg.emissive_boost if emissive_boost is None else float(emissive_boost)
        hbo = self.cfg.hdr_boost if hdr_boost is None else float(hdr_boost)
        iters = self.cfg.blur_iterations if blur_iterations is None else int(blur_iterations)
        inten = self.cfg.intensity if intensity is None else float(intensity)
        exp = self.cfg.exposure if exposure is None else float(exposure)

        bw, bh = self._bloom_size

        # Common state
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        #  Prefilter to seed tex (half-res)
        self.seed_fbo.use()
        self.ctx.viewport = (0, 0, bw, bh)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        input_tex.use(location=0)
        emissive_tex.use(location=1)

        safe_set_uniform(self.prefilter_prog, "u_emissive_boost", ebo)
        safe_set_uniform(self.prefilter_prog, "u_hdr_boost", hbo)

        self.prefilter_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        # Blur ping pong
        src_tex = self.seed_tex
        texel_size = (1.0 / float(bw), 1.0 / float(bh))
        safe_set_uniform(self.blur_prog, "u_texel_size", texel_size)

        for _ in range(max(1, iters)):
            # Horizontal
            self.pingpong_fbo[0].use()
            self.ctx.viewport = (0, 0, bw, bh)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)

            src_tex.use(location=0)
            safe_set_uniform(self.blur_prog, "u_direction", (1.0, 0.0))
            self.blur_vao.render(mode=moderngl.TRIANGLES, vertices=3)

            # Vertical
            self.pingpong_fbo[1].use()
            self.ctx.viewport = (0, 0, bw, bh)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)

            self.pingpong_tex[0].use(location=0)
            safe_set_uniform(self.blur_prog, "u_direction", (0.0, 1.0))
            self.blur_vao.render(mode=moderngl.TRIANGLES, vertices=3)

            src_tex = self.pingpong_tex[1]

        # Composite to screen + tone map/gamma
        self.ctx.screen.use()
        w, h = window_size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        input_tex.use(location=0)
        self.pingpong_tex[1].use(location=1)

        safe_set_uniform(self.composite_prog, "u_bloom_intensity", inten)
        safe_set_uniform(self.composite_prog, "u_exposure", exp)
        #safe_set_uniform(self.composite_prog, "u_gamma", gam)

        tone_mapping_id = TONE_MAPPING_IDS.get(tone_mapping, 0)
        safe_set_uniform(self.composite_prog, "u_tone_mapping_id", tone_mapping_id)
        safe_set_uniform(self.composite_prog, "u_exposure", exposure)
        safe_set_uniform(self.composite_prog, "u_time", time_value)

        self.composite_vao.render(mode=moderngl.TRIANGLES, vertices=3)
