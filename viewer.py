import os
import numpy as np
import trimesh
from typing import Optional

import moderngl
from moderngl_window import WindowConfig, run_window_config, find_window_classes
from pyrr import matrix44
from camera import TrackballCamera
from mouse import OSMouse
import time
from scene import Scene, Mesh, Material
from constants import EPSILON, TexUnit
from utils import safe_set_uniform
from ssao import SSAOPass, SSAOConfig
from gbuffer import GBuffer

class Viewer(WindowConfig):
    title = "pbrv"
    window_size = (1280, 720)
    resource_dir = 'resources'
    vsync = True

    scene: Scene = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.set_icon('icons/moderngl.webp')

        if self.wnd.name == 'headless':
            print('ERROR: headless mode not supported. Exiting.')
            exit(1)

        if self.scene is None:
            print('ERROR: No scene found. Exiting.')
            exit(2)

        # Camera / Interaction
        self.os_mouse = OSMouse(self.wnd)
        self.camera = TrackballCamera(aspect=self.wnd.aspect_ratio)

        self.model = matrix44.create_identity()
        self.wnd.backend
        self.zoom_sensitivity = 0.2

        self._drag_mode = None  # 'rotate' or 'pan'
        self._last_click_time = 0.0
        self._double_click_max_delay = 0.3  # seconds
        

        # --- mesh ---
        mesh = self.scene.mesh
        data = np.hstack([mesh.vertices, mesh.normals, mesh.uv, mesh.tangents]).astype("f4")
        self.vbo = self.ctx.buffer(data.tobytes())
        self.ibo = self.ctx.buffer(mesh.faces.astype("i4").tobytes())

        # --- Frame buffers ---
        self.gbuffer = GBuffer(self.ctx, *self.window_size)

        self.ssao_pass = SSAOPass(self.ctx, self.load_program)


        self.geom_prog = None
        self.light_prog = None
        self.reload_shaders()

        # --- material textures (GL) ---
        self._load_material_textures()


    def reload_shaders(self):
        
        if isinstance(self.geom_prog, moderngl.Program):
            self.geom_prog.release()
        if isinstance(self.light_prog, moderngl.Program):
            self.light_prog.release()
        self.geom_prog = self.load_program(
            vertex_shader='shaders/gbuffer.vert',
            fragment_shader='shaders/gbuffer.frag',
        )
        self.light_prog = self.load_program(
            vertex_shader='shaders/deferred_lighting.vert',
            fragment_shader='shaders/deferred_lighting.frag',
        )

        self._setup_material_samplers()
        self.update_vaos()
    
    def update_vaos(self):
        self.geometry_vao = self.ctx.vertex_array(
        self.geom_prog,
            [( self.vbo, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tangent")],
            self.ibo,
        )
        self.screen_vao   = self.ctx.vertex_array(self.light_prog, [])

    # -------------------------------------------------------------------------
    # Material textures
    # -------------------------------------------------------------------------
    def _load_material_textures(self):
        mat = self.scene.material

        self.albedo_tex, self.use_albedo_tex = self._make_texture2d(mat.albedo_map, 3)
        self.normal_tex, self.use_normal_tex = self._make_texture2d(mat.normal_map, 3)
        self.roughness_tex, self.use_roughness_tex = self._make_texture2d(mat.roughness_map, 1)
        self.metalicness_tex, self.use_metalicness_tex = self._make_texture2d(mat.metalicness_map, 1)
        self.ao_tex, self.use_ao_tex = self._make_texture2d(mat.ambient_occlusion_map, 1)

    def update_material_uniforms(self):
        safe_set_uniform(self.geom_prog, 'u_use_albedo_map', self.use_albedo_tex)
        safe_set_uniform(self.geom_prog, 'u_use_normal_map', self.use_normal_tex)
        safe_set_uniform(self.geom_prog, 'u_use_roughness_map', self.use_roughness_tex)
        safe_set_uniform(self.geom_prog, 'u_use_metalicness_map', self.use_metalicness_tex)
        safe_set_uniform(self.geom_prog, 'u_use_ao_map', self.use_ao_tex)
        safe_set_uniform(self.geom_prog, 'u_albedo', self.scene.material.albedo)
        safe_set_uniform(self.geom_prog, 'u_roughness', self.scene.material.roughness)
        safe_set_uniform(self.geom_prog, 'u_metalicness', self.scene.material.metalicness)


    def _make_texture2d(self, img: Optional[np.ndarray], channels:int=3) -> moderngl.Texture:
        to_use = img is not None
        if img is None:
            img = np.ones((1, 1, channels), dtype="f4")
        else:
            img = np.ascontiguousarray(img.astype('f4'))
        h, w = img.shape[:2]

        if img.ndim == 2:
            components = 1
        else:
            components = img.shape[2]

        tex = self.ctx.texture(
            (w, h),
            components=components,
            data=img.tobytes(),
            dtype='f4',
        )
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        return tex, to_use

    def _setup_material_samplers(self):
        safe_set_uniform(self.geom_prog, 'u_albedo_map', TexUnit.ALBEDO_MAP)
        safe_set_uniform(self.geom_prog, 'u_normal_map', TexUnit.NORMAL_MAP)
        safe_set_uniform(self.geom_prog, 'u_roughness_map', TexUnit.ROUGHNESS_MAP)
        safe_set_uniform(self.geom_prog, 'u_metalicness_map', TexUnit.METALICNESS_MAP)
        safe_set_uniform(self.geom_prog, 'u_ao_map', TexUnit.AO_MAP)

    # -------------------------------------------------------------------------
    # Mesh / GBuffer
    # -------------------------------------------------------------------------


    def on_resize(self, width: int, height: int):   
        self.ctx.viewport = (0, 0, width, height)
        self.camera.resize(width, height)
        self.gbuffer.resize(width, height)
        self.ssao_pass.resize(width, height)

    # -------------------------------------------------------------------------
    # Mouse / camera
    # -------------------------------------------------------------------------
    def on_mouse_press_event(self, x, y, button):
        if button == self.wnd.mouse.left:
            now = time.perf_counter()
            dt = now - self._last_click_time

            is_double = dt <= self._double_click_max_delay

            if is_double:
                picked_pos = self.gbuffer.sample_world_position(x, y)
                if picked_pos is not None:
                    self.camera.set_pivot(picked_pos)
                    self.os_mouse.center()
            else:
                self._drag_mode = 'rotate'
                w, h = self.wnd.size
                self.camera.begin_rotate(x, y, w, h)

            self._last_click_time = now

        if button == self.wnd.mouse.right:
            self._drag_mode = 'pan'

    def on_mouse_drag_event(self, x, y, dx, dy):
        w, h = self.wnd.size
        if self._drag_mode == 'rotate':
            self.camera.rotate(x, y, w, h)
        elif self._drag_mode == 'pan':
            self.camera.pan(dx, dy, w, h)

    def on_mouse_release_event(self, x, y, button):
        if button == self.wnd.mouse.left and self._drag_mode == 'rotate':
            self.camera.end_rotate()
        if button == self.wnd.mouse.right and self._drag_mode == 'pan':
            pass
        self._drag_mode = None

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.camera.zoom(y_offset * self.zoom_sensitivity)

    def on_key_event(self, key, action, modifiers):
        if key == self.wnd.keys.F5 and action == self.wnd.keys.ACTION_PRESS:
            self.reload_shaders()
    

    
    # -------------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------------
    def on_render(self, time: float, frame_time: float):
        # ---------- GEOMETRY PASS: fill G-buffer ----------
        self.gbuffer.fbo.use()
        self.ctx.viewport = (0, 0, self.gbuffer.width, self.gbuffer.height)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        view, eye, _ = self.camera.get_view()

        self.geom_prog['u_model'].write(np.array(self.model, dtype='f4').tobytes())
        self.geom_prog['u_view'].write(np.array(view, dtype='f4').tobytes())
        self.geom_prog['u_projection'].write(self.camera.projection.astype('f4').tobytes())
        normal_matrix = np.linalg.inv(self.model).T[:3,  :3]
        self.geom_prog['u_normal_matrix'].write(np.array(normal_matrix, dtype='f4').tobytes())

        # Bind material textures (even if shader doesn't use them, it's fine)
        self.albedo_tex.use(location=TexUnit.ALBEDO_MAP)
        self.normal_tex.use(location=TexUnit.NORMAL_MAP)
        self.roughness_tex.use(location=TexUnit.ROUGHNESS_MAP)
        self.metalicness_tex.use(location=TexUnit.METALICNESS_MAP)
        self.ao_tex.use(location=TexUnit.AO_MAP)

        self.update_material_uniforms()
        safe_set_uniform(self.geom_prog, 'u_time', time)

        # draw mesh into G-buffer
        self.geometry_vao.render()


        # ---------- SSAO (half-res) ----------
        self.ssao_pass.render(self.gbuffer.position, self.gbuffer.normal, view, self.camera.projection)
        self.ssao_pass.blur(self.gbuffer.position, self.gbuffer.normal)

        # ---------- LIGHTING PASS: shade full screen ----------
        self.ctx.screen.use()
        w, h = self.wnd.size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.02, 0.02, 0.02, 1.0)

        # bind G-buffer textures to texture units 0,1,2
        self.gbuffer.position.use(location=TexUnit.GBUFFER_POSITION)
        self.gbuffer.normal.use(location=TexUnit.GBUFFER_NORMAL)
        self.gbuffer.albedo.use(location=TexUnit.GBUFFER_ALBEDO)
        self.gbuffer.rmao.use(location=TexUnit.GBUFFER_RMAO)
        self.ssao_pass.output_texture.use(location=TexUnit.SSAO_BLUR)

        safe_set_uniform(self.light_prog, 'gPosition'  , TexUnit.GBUFFER_POSITION)
        safe_set_uniform(self.light_prog, 'gNormal'    , TexUnit.GBUFFER_NORMAL)
        safe_set_uniform(self.light_prog, 'gAlbedo'    , TexUnit.GBUFFER_ALBEDO)
        safe_set_uniform(self.light_prog, 'gRMAO'      , TexUnit.GBUFFER_RMAO)
        safe_set_uniform(self.light_prog, 'u_ssao'     , TexUnit.SSAO_BLUR)
        safe_set_uniform(self.light_prog, 'u_use_ssao' , not self.use_ao_tex)
        safe_set_uniform(self.light_prog, 'u_viewPos' , tuple(eye))
        safe_set_uniform(self.light_prog, 'u_lightPos', (1.0, 1.0, 1.0))
        safe_set_uniform(self.light_prog, 'u_lightColor', (1.0, 1.0, 1.0))
        safe_set_uniform(self.light_prog, 'u_time', time)

        self.screen_vao.render(mode=moderngl.TRIANGLES, vertices=3)


if __name__ == '__main__':
    mesh = Mesh.from_path('resources/meshes/ship.obj')

    # Example: create a material with an albedo map
    material = Material.from_map_paths(
        albedo_path='resources/textures/ship_a.jpg',
        normal_path='resources/textures/ship_n.jpg',
        roughness_path='resources/textures/ship_r.jpg',
        metalicness_path='resources/textures/ship_m.jpg',
        ambient_occlusion_path='resources/textures/ship_ao.jpg',
    )
    material.roughness = 0.3
    material.metalicness = 0.0

    Viewer.scene = Scene(mesh=mesh, material=material)
    run_window_config(Viewer)
