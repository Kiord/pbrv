# rendering/deferred_gl/renderer.py

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from moderngl import Context

from pbrv.core.scene import Scene, DirectionalLight

from pbrv.rendering.api import Renderer, FrameState
from pbrv.rendering.deferred_gl.gbuffer import GBuffer
from pbrv.rendering.deferred_gl.geometry_pass import GeometryPass
from pbrv.rendering.deferred_gl.shadow_pass import ShadowPass
from pbrv.rendering.deferred_gl.ssao_pass import SSAOPass
from pbrv.rendering.deferred_gl.lighting_pass import LightingPass
from pbrv.rendering.deferred_gl.post_pass import PostProcessingPass
from pbrv.rendering.deferred_gl.gl_scene import GLMesh, upload_mesh


class DeferredGLRenderer(Renderer):
    name: str = "deferred_gl"

    def __init__(self, ctx: Context, load_program_fn: Callable[..., object]) -> None:
        self.ctx = ctx
        self.load_program_fn = load_program_fn

        self._initialized: bool = False
        self._size: Tuple[int, int] = (1, 1)

        self._scene: Optional[Scene] = None
        self._glmesh: Optional[GLMesh] = None

        self.gbuffer: Optional[GBuffer] = None
        self.geometry_pass: Optional[GeometryPass] = None
        self.shadow_pass: Optional[ShadowPass] = None
        self.ssao_pass: Optional[SSAOPass] = None
        self.lighting_pass: Optional[LightingPass] = None
        self.post_pass: Optional[PostProcessingPass] = None


    def initialize(self, size: Tuple[int, int]) -> None:
        self._initialized = True
        self._size = (int(size[0]), int(size[1]))

        if self._scene is not None:
            self._build_for_scene(self._scene, self._size)
        else:
            # minimal resources for pick/render
            w, h = self._size
            self.gbuffer = GBuffer(self.ctx, w, h)
            self.ssao_pass = SSAOPass(self.ctx, self.load_program_fn)
            self.lighting_pass = LightingPass(self.ctx, self.load_program_fn, None, None)
            self.post_pass = PostProcessingPass(self.ctx, self.load_program_fn)

            self.ssao_pass.resize(w, h)
            self.lighting_pass.resize(w, h)
            self.post_pass.resize(w, h)

    def shutdown(self) -> None:
        if self.geometry_pass is not None:
            self.geometry_pass.release()
            self.geometry_pass = None

        if self.shadow_pass is not None:
            self.shadow_pass.release()
            self.shadow_pass = None

        if self.ssao_pass is not None:
            self.ssao_pass.release()
            self.ssao_pass = None

        if self.lighting_pass is not None:
            self.lighting_pass.release()
            self.lighting_pass = None

        if self.post_pass is not None:
            self.post_pass.release()
            self.post_pass = None

        if self.gbuffer is not None:
            self.gbuffer.release()
            self.gbuffer = None

        if self._glmesh is not None:
            try:
                self._glmesh.vbo.release()
            except Exception:
                pass
            try:
                self._glmesh.ibo.release()
            except Exception:
                pass
            self._glmesh = None

        self._scene = None
        self._initialized = False

    def set_scene(self, scene: Scene) -> None:
        self._scene = scene
        if not self._initialized:
            return
        self._build_for_scene(scene, self._size)

    def resize(self, size: Tuple[int, int]) -> None:
        self._size = (int(size[0]), int(size[1]))
        w, h = self._size

        self.ctx.viewport = (0, 0, w, h)

        if self.gbuffer is not None:
            self.gbuffer.resize(w, h)

        if self.ssao_pass is not None:
            self.ssao_pass.resize(w, h)

        if self.lighting_pass is not None:
            self.lighting_pass.resize(w, h)

        if self.post_pass is not None:
            self.post_pass.resize(w, h)


    def render(self, frame: FrameState) -> None:
        if not self._initialized:
            raise RuntimeError("DeferredGLRenderer.render() called before initialize().")
        if self._scene is None:
            raise RuntimeError("DeferredGLRenderer.render() called before set_scene().")
        if self.gbuffer is None or self.geometry_pass is None or self.shadow_pass is None:
            raise RuntimeError("DeferredGLRenderer not fully built. Did initialize/set_scene run?")

        scene = self._scene

        dir_light: Optional[DirectionalLight] = None
        if scene.dir_light is not None:
            dir_light = self.shadow_pass.render(frame.model_matrix, scene.dir_light, frame.light_matrix)

        self.geometry_pass.render(
            self.gbuffer,
            scene.material,
            frame.model_matrix,
            frame.camera.view,
            frame.camera.proj,
            frame.time,
        )

        ssao_tex = None
        
        if frame.use_ssao and self.ssao_pass is not None:
            self.ssao_pass.render(self.gbuffer.position, self.gbuffer.normal, frame.camera.view, frame.camera.proj)
            self.ssao_pass.blur(self.gbuffer.position, self.gbuffer.normal)
            ssao_tex = self.ssao_pass.output_texture
        
        use_ssao = frame.use_ssao and (ssao_tex is not None)

        if self.lighting_pass is None:
            raise RuntimeError("LightingPass not initialized.")
        self.lighting_pass.render(
            self.gbuffer,
            ssao_tex,
            self.shadow_pass.depth_tex,
            scene.point_light,
            dir_light,
            frame.camera.eye,
            frame.camera.inv_view,
            frame.camera.inv_proj,
            frame.env_matrix,
            frame.env_lod_factor,
            use_ssao,
            scene.material.specular_tint,
            frame.time,
            frame.window_size,
        )

        if self.post_pass is None:
            raise RuntimeError("PostProcessingPass not initialized.")
        self.post_pass.render(
            self.lighting_pass.output_texture,
            self.gbuffer.emissive,
            frame.tone_mapping,
            frame.exposure,
            frame.time,
            frame.window_size,
        )


    def pick_world_position(self, x: int, y: int) -> Optional[np.ndarray]:
        if self.gbuffer is None:
            return None
        return self.gbuffer.sample_world_position(float(x), float(y))


    def reload_shaders(self) -> None:
        if self.geometry_pass is not None:
            self.geometry_pass.reload_shaders()
        if self.ssao_pass is not None:
            self.ssao_pass.reload_shaders()
        if self.lighting_pass is not None:
            self.lighting_pass.reload_shaders()
        if self.post_pass is not None:
            self.post_pass.reload_shaders()

    def _build_for_scene(self, scene: Scene, size: Tuple[int, int]) -> None:
        if self.geometry_pass is not None:
            self.geometry_pass.release()
        if self.shadow_pass is not None:
            self.shadow_pass.release()
        if self.lighting_pass is not None:
            self.lighting_pass.release()

        if self._glmesh is not None:
            try:
                self._glmesh.vbo.release()
            except Exception:
                pass
            try:
                self._glmesh.ibo.release()
            except Exception:
                pass
            self._glmesh = None

        self._glmesh = upload_mesh(self.ctx, scene.mesh)

        w, h = int(size[0]), int(size[1])

        # Create/resize gbuffer
        if self.gbuffer is None:
            self.gbuffer = GBuffer(self.ctx, w, h)
        else:
            self.gbuffer.resize(w, h)

        # Passes
        self.geometry_pass = GeometryPass(
            self.ctx,
            self.load_program_fn,
            self._glmesh,
            scene.material,
        )

        if self.ssao_pass is None:
            self.ssao_pass = SSAOPass(self.ctx, self.load_program_fn)
        self.ssao_pass.resize(w, h)

        self.shadow_pass = ShadowPass(self.ctx, self.load_program_fn, self._glmesh)

        self.lighting_pass = LightingPass(
            self.ctx,
            self.load_program_fn,
            scene.envmap,
            scene.sun,
        )
        self.lighting_pass.resize(w, h)

        if self.post_pass is None:
            self.post_pass = PostProcessingPass(self.ctx, self.load_program_fn)
        self.post_pass.resize(w, h)
