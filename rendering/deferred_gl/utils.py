from typing import Protocol, Any, Optional, Callable
from moderngl import Program, Context
import numpy as np

class RenderPass(Protocol):
    def __init__(self, ctx: Context, load_program_fn:Optional[Callable[..., Program]]=None):
        self.ctx = ctx
        if load_program_fn is None:
            load_program_fn = ctx.program
        self.load_program_fn = load_program_fn
    def resize(self, w:int, h:int) -> None: ...
    def reload_shaders(self) -> None: ...
    def release(self) -> None: ...
    def render(self) -> None: ...


def safe_set_uniform(prog:Program, name: str, value: Any):
    if name in prog:
        if isinstance(value, np.ndarray):
            prog[name].write(value.astype(np.float32).tobytes())
        else:
            prog[name].value = value
        return

class TexUnit:
    GBUFFER_POSITION = 0
    GBUFFER_NORMAL   = 1
    GBUFFER_ALBEDO   = 2
    GBUFFER_RMAOS    = 3
    GBUFFER_EMISSIVE = 4

    SSAO_NOISE       = 5
    SSAO             = 6
    SSAO_BLUR        = 7

    SHADOW_MAP       = 8

    ALBEDO_MAP       = 9
    NORMAL_MAP       = 10
    ROUGHNESS_MAP    = 11
    METALLIC_MAP     = 12
    EMISSIVE_MAP     = 13
    SPECULAR_MAP     = 14
    AO_MAP           = 15

    ENV_BACKGROUND   = 16
    ENV_IRRADIANCE   = 17
    ENV_SPECULAR     = 18
