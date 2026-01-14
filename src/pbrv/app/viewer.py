from typing import Optional, Callable

from moderngl_window import WindowConfig, run_window_config

from pbrv.core.camera import Camera
from pbrv.core.trackball import Trackball
from pbrv.core.input_gestures import CameraInputController

from pbrv.core.scene import Scene

from pbrv.rendering.api import FrameState, Renderer


class Viewer(WindowConfig):
    title = "pbrv"
    window_size = (1280, 720)
    resource_dir = "resources"
    vsync = True

    scene: Scene = None
    use_ssao: bool = False
    use_bloom: bool = False
    tone_mapping: str = "aces"
    exposure: float = 1.0

    renderer_factory: Callable[..., Renderer]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.set_icon("icons/moderngl.webp")

        if self.wnd.name == "headless":
            print("ERROR: headless mode not supported. Exiting.")
            raise SystemExit(1)

        if self.scene is None:
            print("ERROR: No scene found. Exiting.")
            raise SystemExit(2)

        self.orbit = Trackball(ball_size=0.8)
        self.camera = Camera(aspect=self.wnd.aspect_ratio, orientation=self.orbit)

        self.renderer = self.renderer_factory(self.ctx, self.load_program)
        self.renderer.set_scene(self.scene)
        self.renderer.initialize(self.wnd.size)

        self.input = CameraInputController(
            self.wnd,
            self.camera,
            self.orbit,
            pick_world_position=self.renderer.pick_world_position,
        )

    def reload_shaders(self):
        if hasattr(self.renderer, "reload_shaders"):
            self.renderer.reload_shaders()

    def on_resize(self, width: int, height: int):
        self.ctx.viewport = (0, 0, width, height)
        self.camera.resize(width, height)
        self.renderer.resize((width, height))

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

    def on_render(self, time: float, frame_time: float):
        frame = FrameState(
            time=float(time),
            camera=self.camera,
            model_matrix=self.input.model_matrix,
            env_matrix=self.input.env_matrix,
            light_matrix=self.input.light_matrix,
            exposure=float(self.exposure),
            tone_mapping=str(self.tone_mapping),
            env_lod_factor=float(self.input.lod_factor) if self.input.lod_factor is not None else None,
            use_ssao=bool(self.use_ssao),
            use_bloom=bool(self.use_bloom),
            window_size=self.wnd.size,
        )
        self.renderer.render(frame)

    def on_close(self):
        self.renderer.shutdown()


if __name__ == "__main__":
    from pbrv.core.scene import Mesh, Material, Panorama, CubeMap
    from pbrv.core.sun_extraction import SunExtractSettings
    from pbrv.rendering.registry import REGISTRY

    asset_name = "helmet"
    mesh = Mesh.from_path(f"resources/meshes/{asset_name}.obj")
    material = Material.from_map_paths(
        albedo_path=f"resources/textures/{asset_name}_a.jpg",
        normal_path=f"resources/textures/{asset_name}_n.jpg",
        roughness_path=f"resources/textures/{asset_name}_r.jpg",
        metallic_path=f"resources/textures/{asset_name}_m.jpg",
        emissive_path=f"resources/textures/{asset_name}_e.jpg",
        ambient_occlusion_path=f"resources/textures/{asset_name}_ao.jpg",
    )
    #envmap = Panorama.from_path("resources/panoramas/shanghai.exr")
    se_settings = SunExtractSettings(threshold_value=0.9)
    envmap = CubeMap.from_path("resources/cubemaps/learnopengl")

    scene = Scene(mesh=mesh, material=material, envmap=envmap, point_light=None, dir_light=None)
    scene.auto_sun(se_settings)

    Viewer.scene = scene
    Viewer.use_ssao = True

    Viewer.renderer_factory = REGISTRY['deferred_gl']

    run_window_config(Viewer)
