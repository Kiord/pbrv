from viewer import Viewer
from scene import Scene, Material, Mesh
from moderngl_window import run_window_config
import typer

def cli(mesh_path,
        albedo_path,
        normal_path,
        roughness_path,
        metalness_path,
        ambient_occlusion_path):
    mesh = Mesh.from_path(mesh_path)

    material = Material.from_map_paths(
        albedo_path=albedo_path,
        normal_path=normal_path,
        roughness_path=roughness_path,
        metalness_path=metalness_path,
        ambient_occlusion_path=ambient_occlusion_path,
    )
    material.roughness = 0.3
    material.metalness = 0.0

    Viewer.scene = Scene(mesh=mesh, material=material)
    run_window_config(Viewer)


if __name__ == '__main__':
    typer.run(cli)