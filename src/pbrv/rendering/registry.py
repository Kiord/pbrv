from typing import Callable 
from moderngl import Context
from pbrv.rendering.deferred_gl.renderer import DeferredGLRenderer

def deferred_gl_renderer_factory(ctx: Context, load_program_fn:Callable):
    return DeferredGLRenderer(ctx, load_program_fn)

REGISTRY = {"deferred_gl": staticmethod(deferred_gl_renderer_factory)}