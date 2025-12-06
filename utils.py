from typing import Any, Callable, Optional
from moderngl import Program, Context

def safe_set_uniform(prog:Program, name: str, value: Any):
    if name in prog:
        prog[name].value = value


class Pass:
    def __init__(self, ctx: Context, load_program_fn:Optional[Callable[..., Program]]=None):
        self.ctx = ctx
        if load_program_fn is None:
            load_program_fn = ctx.program
        self.load_program_fn = load_program_fn