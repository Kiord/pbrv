from rendering.deferred_gl.renderer import deferred_gl_renderer_factory

REGISTRY = {"deferred_gl": deferred_gl_renderer_factory}