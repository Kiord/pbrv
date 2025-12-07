from moderngl_window import WindowConfig, run_window_config
from pyrr import matrix44
from camera import TrackballCamera
from mouse import OSMouse
import time
from scene import Scene, Mesh, Material, CubeMap, Panorama
from ssao import SSAOPass
from gbuffer import GBuffer
from geometry_pass import GeometryPass
from lighting_pass import LightingPass
from ibl import EnvironmentMapPrecomputer, PrefilterSettings
from typing import Optional
from moderngl import TextureCube

class Viewer(WindowConfig):
    title = "pbrv"
    window_size = (1280, 720)
    resource_dir = 'resources'
    vsync = True
    use_ssao = False

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
        self.vbo, self.ibo = self.scene.mesh.to_gl(self.ctx)

        # --- Passes ---
        self.gbuffer = GBuffer(self.ctx, *self.window_size)

        self.geometry_pass = GeometryPass(
            self.ctx,
            self.load_program,
            self.scene,
            self.vbo,
            self.ibo,
        )

        self.ssao_pass = SSAOPass(self.ctx, self.load_program)

        self.lighting_pass = LightingPass(self.ctx, self.load_program, self.scene.envmap)
              

    def reload_shaders(self):
        self.geometry_pass.reload_shaders()
        self.ssao_pass.reload_shaders()
        self.lighting_pass.reload_shaders()

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
        view, eye, _ = self.camera.get_view()
        proj = self.camera.projection

        # geometry
        self.geometry_pass.render(self.gbuffer, self.model, view, proj, time)

        # ssao
        if self.use_ssao:
            self.ssao_pass.render(self.gbuffer.position, self.gbuffer.normal, view, proj)
            self.ssao_pass.blur(self.gbuffer.position, self.gbuffer.normal)

        # lighting
        self.lighting_pass.render(
            self.gbuffer,
            self.ssao_pass.output_texture,
            eye,
            view, 
            proj,
            self.use_ssao,
            time,
            self.wnd.size,
        )


if __name__ == '__main__':
    mesh = Mesh.from_path('resources/meshes/lantern.obj')
    material = Material.from_map_paths(
        albedo_path='resources/textures/lantern_a.jpg',
        normal_path='resources/textures/lantern_n.jpg',
        roughness_path='resources/textures/lantern_r.jpg',
        metalness_path='resources/textures/lantern_m.jpg',
        ambient_occlusion_path='resources/textures/lantern_ao.jpg',
    )
    envmap = Panorama.from_path('resources/panoramas/hangar1.jpg')

    Viewer.scene = Scene(mesh=mesh, material=material, envmap=envmap)
    run_window_config(Viewer)
