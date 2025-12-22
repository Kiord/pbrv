from moderngl_window import WindowConfig, run_window_config
from camera import TrackballCamera
from typing import Optional
from scene import Scene, Mesh, Material, Panorama, DirectionalLight, PointLight
from ssao_pass import SSAOPass
from gbuffer import GBuffer
from geometry_pass import GeometryPass
from lighting_pass import LightingPass
from shadow_pass import ShadowPass
from post_pass import PostProcessingPass

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
        vbo, ibo = self.scene.mesh.to_gl(self.ctx)

        # --- Passes ---
        self.gbuffer = GBuffer(self.ctx, *self.window_size)

        self.geometry_pass = GeometryPass(
            self.ctx,
            self.load_program,
            vbo,
            ibo,
            self.scene.material,
        )

        self.ssao_pass = SSAOPass(self.ctx, self.load_program)

        self.shadow_pass = ShadowPass(self.ctx, self.load_program, vbo, ibo)

        self.lighting_pass = LightingPass(
            self.ctx, 
            self.load_program, 
            self.scene.envmap,
            self.scene.sun)
        
        self.post_pass = PostProcessingPass(self.ctx, self.load_program)
            
        # Camera / Interaction
        
        self.camera = TrackballCamera(aspect=self.wnd.aspect_ratio)
        
        self.input = CameraInputController(self.wnd, self.camera, 
                                           sample_world_position=self.gbuffer.sample_world_position)

    def reload_shaders(self):
        self.geometry_pass.reload_shaders()
        self.ssao_pass.reload_shaders() 
        self.lighting_pass.reload_shaders()
        self.post_pass.reload_shaders()

    # -------------------------------------------------------------------------
    # Mesh / GBuffer
    # -------------------------------------------------------------------------


    def on_resize(self, width: int, height: int):   
        self.ctx.viewport = (0, 0, width, height)
        self.camera.resize(width, height)
        self.gbuffer.resize(width, height)
        self.ssao_pass.resize(width, height)
        self.lighting_pass.resize(width, height)
        self.post_pass.resize(width, height)

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
        self.input.on_key_event(key, action, modifiers)
        if key == self.wnd.keys.F5 and action == self.wnd.keys.ACTION_PRESS:
            self.reload_shaders()

    # -------------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------------
    def on_render(self, time: float, frame_time: float):

        dir_light:Optional[DirectionalLight] = None
        if self.scene.dir_light is not None:
            dir_light = self.shadow_pass.render(self.input.model_matrix, 
                                                self.scene.dir_light, 
                                                self.input.env_matrix)

        # geometry
        self.geometry_pass.render(self.gbuffer, self.scene.material, 
                                  self.input.model_matrix, self.camera.view, self.camera.proj, time)

        # ssao
        if self.use_ssao:
            self.ssao_pass.render(self.gbuffer.position, self.gbuffer.normal, self.camera.view, self.camera.proj)
            self.ssao_pass.blur(self.gbuffer.position, self.gbuffer.normal)

        # lighting
        self.lighting_pass.render(
            self.gbuffer,
            self.ssao_pass.output_texture,
            self.shadow_pass.depth_tex,
            self.scene.point_light,
            dir_light,
            self.camera.eye,
            self.camera.inv_view, 
            self.camera.inv_proj,
            self.input.env_matrix,
            self.input.lod_factor,
            self.use_ssao,
            self.scene.material.specular_tint,
            time,
            self.wnd.size,
        )

        self.post_pass.render(
            self.lighting_pass.output_texture,
            self.gbuffer.emissive,
            self.tone_mapping,
            self.exposure,
            time,
            self.wnd.size,
            )

    def on_close(self):
        self.geometry_pass.release()
        self.shadow_pass.release()
        self.ssao_pass.release()
        self.lighting_pass.release()
        self.post_pass.release()
        self.gbuffer.release()

if __name__ == '__main__':
    asset_name = 'drone'
    mesh = Mesh.from_path(f'resources/meshes/{asset_name}.obj')
    material = Material.from_map_paths(
        albedo_path=f'resources/textures/{asset_name}_a.jpg',
        normal_path=f'resources/textures/{asset_name}_n.jpg',
        roughness_path=f'resources/textures/{asset_name}_r.jpg',
        metallic_path=f'resources/textures/{asset_name}_m.jpg',
        emissive_path=f'resources/textures/{asset_name}_e.jpg',
        ambient_occlusion_path=f'resources/textures/{asset_name}_ao.jpg',
    )
    envmap = Panorama.from_path('resources/panoramas/shanghai.exr')

    point_light = None#PointLight(position=(1.0,1.0,1.0), color=(5.0,5.0,5.0))
    dir_light = None#DirectionalLight((1,1,1), (1, -1, 1))
    scene = Scene(mesh=mesh, material=material, envmap=envmap, point_light=point_light,dir_light=dir_light)
    #scene.auto_sun()
    Viewer.scene = scene
    #Viewer.use_ssao = True
    run_window_config(Viewer)
