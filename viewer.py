from moderngl_window import WindowConfig, run_window_config
from camera import TrackballCamera
from scene import Scene, Mesh, Material, Panorama
from ssao import SSAOPass
from gbuffer import GBuffer
from geometry_pass import GeometryPass
from lighting_pass import LightingPass

from input_gestures import CameraInputController

class Viewer(WindowConfig):
    title = "pbrv"
    window_size = (1280, 720)
    resource_dir = 'resources'
    vsync = True
    use_ssao = False

    scene: Scene = None

    tone_mapping = 'aces'
    exposure = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.set_icon('icons/moderngl.webp')

        if self.wnd.name == 'headless':
            print('ERROR: headless mode not supported. Exiting.')
            exit(1)

        if self.scene is None:
            print('ERROR: No scene found. Exiting.')
            exit(2)
        
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

        self.lighting_pass = LightingPass(
            self.ctx, 
            self.load_program, 
            self.scene.envmap,
            self.scene.material.specular_tint,
            self.scene.point_light)
            
        # Camera / Interaction
        
        self.camera = TrackballCamera(aspect=self.wnd.aspect_ratio)
        
        self.input = CameraInputController(self.wnd, self.camera, 
                                           sample_world_position=self.gbuffer.sample_world_position)

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
        self.input.on_press(x, y, button)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.input.on_drag(x, y, dx, dy)

    def on_mouse_release_event(self, x, y, button):
        self.input.on_release(x, y, button)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.input.on_scroll(y_offset)

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
        self.geometry_pass.render(self.gbuffer, self.input.model_matrix, view, proj, time)

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
            self.tone_mapping,
            self.exposure,
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

    point_light = None#PointLight(position=(1.0,1.0,1.0), color=(5.0,5.0,5.0))

    Viewer.scene = Scene(mesh=mesh, material=material, envmap=envmap, point_light=point_light)
    run_window_config(Viewer)
