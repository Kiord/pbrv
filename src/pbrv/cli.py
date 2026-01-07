import sys

import argparse
from pathlib import Path
from typing import Optional, Tuple

from moderngl_window import run_window_config

from pbrv.app.viewer import Viewer
from pbrv.core.scene import Scene, Material, Mesh, Environment, Panorama, CubeMap, Light
from pbrv.core.constants import TONE_MAPPING_IDS

from pbrv.rendering.registry import REGISTRY


def parse_value_or_path(
    value: Optional[str],
    default_value: Optional[Tuple[float, ...]],
    valid_lengths: Tuple[int, ...],
    param_name: str,
) -> tuple[Optional[Tuple[float, ...]], Optional[Path]]:
    
    if value is None:
        return default_value, None

    p = Path(value)
    if p.exists():
        return default_value, p

    parts = value.replace(",", " ").split()
    try:
        floats = tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError(f"{param_name}: '{value}' is not a valid numeric value.")

    if len(floats) not in valid_lengths:
        raise ValueError(
            f"{param_name}: '{value}' has length {len(floats)}, "
            f"expected one of {valid_lengths}."
        )

    return floats, None



def run() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time PBR viewer (mesh + material + moderngl-window options)",
    )

    parser.add_argument(
        "mesh_path",
        type=Path,
        nargs='*',
        help="Mesh file (e.g. .obj, .glb, .ply)",
    )

    parser.add_argument(
        "--albedo",
        "-a",
        dest='albedo',
        metavar="VALUE_OR_PATH",
        help="Albedo map path OR 'r,g,b' (or single scalar)",
    )
    parser.add_argument(
        "--normal",
        "-n",
        type=Path,
        metavar="PATH",
        help="Normal map texture",
    )
    parser.add_argument(
        "--roughness","--rough", "-r",
        dest='roughness',
        metavar="VALUE_OR_PATH",
        help="Roughness scalar (0..1) OR roughness map path",
    )
    parser.add_argument(
        "--metallic","--metal", "-m",
        dest='metallic',
        metavar="VALUE_OR_PATH",
        help="Metallic scalar (0..1) OR metallic map path",
    )
    
    parser.add_argument(
        "--ambient-occlusion",
        "-ao",
        dest="ao",
        type=Path,
        metavar="PATH",
        help="Ambient occlusion map texture",
    )

    parser.add_argument(
        "--emissive",
        "-em",
        dest='emissive',
        metavar="VALUE_OR_PATH",
        help="Albedo map path OR 'r,g,b' (or single scalar)",
    )

    parser.add_argument(
        "--specular","-s",
        dest='specular',
        metavar="VALUE_OR_PATH",
        help="Specular scalar (0..1) OR specular map path. /!\\ This corresponds to the \"specular\" artistic parameter of the Disney's BRDF that scales dielectric F0.",
    )

    parser.add_argument(
        "--specular-tint","-st",
        dest='specular_tint',
        type=float,
        default=0.0,
        help="Specular tint scalar (0..1). /!\\ This corresponds to the \"specular tint\" artistic parameter of the Disney's BRDF that modulates the tint of F0.",
    )

    parser.add_argument(
        "-ssao",
        "--use-ssao",
        dest='use_ssao',
        action='store_true',
        help="Enable SSAO",
    )

    parser.add_argument(
        "--env", "-e",
        dest="env",
        metavar="VALUE_OR_PATH",
        help="Cubemap directory with right/left/top/bottom/front/back images OR panorama image path OR 'r,g,b' OR single scalar (grey)",
    )

    parser.add_argument(
        "-as",
        "--autosun",
        dest='use_autosun',
        action='store_true',
        help="Add to automatically find a main light direction in the provided env map.",
    )

    parser.add_argument(
        "--tone-mapping", '--tonemap', '-t', '-tm',
        dest="tone_mapping",
        type=str,
        choices=TONE_MAPPING_IDS.keys(),
        default='aces',
        help="Tone mapping type.",
    )

    parser.add_argument(
        "--exposure", '-exp',
        dest="exposure",
        type=float,
        default=1.0,
        help="Exposure to apply before tone mapping",
    )

    parser.add_argument(
        "--renderer", '-rend',
        dest="renderer",
        type=str,
        default='deferred_gl',
        help="Renderer to use. For now, only \"deffered_gl\" exists.",
    )

    args, mw_args = parser.parse_known_args()

    if len(args.mesh_path) > 1:
         parser.error(f"Only one mesh path can be processed.")

    if len(args.mesh_path) > 0 and not args.mesh_path[0].exists():
        parser.error(f"Mesh path does not exist: {args.mesh_path[0]}")
    
    if args.use_ssao and args.ao is not None:
        print(f'[Warning] you enabled SSAO explicitely but you provided a ambient occlusion texture path ({args.ao}). SSAO will be used in favor of your texture.')

    try:
        albedo_vals, albedo_map = parse_value_or_path(
            args.albedo,
            default_value=(1.0, 1.0, 1.0),
            valid_lengths=(1, 3),
            param_name="--albedo",
        )
        if len(albedo_vals) == 1:
            albedo_color = (albedo_vals[0],) * 3
        else:
            albedo_color = albedo_vals

        roughness_vals, roughness_map = parse_value_or_path(
            args.roughness,
            default_value=(1.0,),
            valid_lengths=(1,),
            param_name="--roughness",
        )
        roughness_value = roughness_vals[0]

        metallic_vals, metallic_map = parse_value_or_path(
            args.metallic,
            default_value=(0.0,),
            valid_lengths=(1,),
            param_name="--metallic",
        )
        metallic_value = metallic_vals[0]

        emissive_vals, emissive_map = parse_value_or_path(
            args.emissive,
            default_value=(0.0, 0.0, 0.0),
            valid_lengths=(1, 3),
            param_name="--emissive",
        )
        emissive_value = emissive_vals

        specular_vals, specular_map = parse_value_or_path(
            args.specular,
            default_value=(0.3,),
            valid_lengths=(1,),
            param_name="--specular",
        )
        specular_value = specular_vals[0]

        env_color, env_path = parse_value_or_path(
            args.env,
            default_value=None,
            valid_lengths=(1, 3),
            param_name="--env",
        )
        if env_color is not None and len(env_color) == 1:
            env_color = (env_color[0],) * 3
        
        envmap:Optional[Environment] = None
        if env_path is not None:
            if not env_path.exists():
                parser.error(f"{env_path} does not exist")
            is_cubemap = env_path.is_dir()
            is_panorama = env_path.is_file()
            if not(is_cubemap or is_panorama):
                parser.error(f"{env_path} is neither a file nor a directory")
            cls = CubeMap if is_cubemap else Panorama
            envmap = cls.from_path(str(args.env))
        elif env_color is not None:
            envmap = Light(env_color)


    except ValueError as e:
        parser.error(str(e))


   

    if len(args.mesh_path) > 0:
        mesh = Mesh.from_path(str(args.mesh_path[0]))
    else:
        mesh = Mesh.create_sphere()

    material = Material.from_map_paths(
        albedo_path=str(albedo_map) if albedo_map else None,
        normal_path=str(args.normal) if args.normal else None,
        roughness_path=str(roughness_map) if roughness_map else None,
        metallic_path=str(metallic_map) if metallic_map else None,
        emissive_path=str(emissive_map) if emissive_map else None,
        specular_path=str(specular_map) if specular_map else None,
        ambient_occlusion_path=str(args.ao) if args.ao else None,
    )

    material.albedo = albedo_color
    material.roughness = roughness_value
    material.metallic = metallic_value
    material.emissive = emissive_value
    material.specular = specular_value
    material.specular_tint = args.specular_tint

    Viewer.scene = Scene(mesh=mesh, material=material, envmap=envmap)
    if args.use_autosun:
        Viewer.scene.auto_sun()
    Viewer.use_ssao = args.use_ssao
    Viewer.tone_mapping = args.tone_mapping
    Viewer.exposure = args.exposure

    if args.renderer not in REGISTRY:
        raise ValueError(f"Available renders are \"{list(REGISTRY)}\". Found {args.renderer}")
    Viewer.renderer_factory = REGISTRY[args.renderer]

    
    sys.argv = sys.argv[:1] # To trick mgl-window if mw_args is empty
    run_window_config(Viewer, args=mw_args)


if __name__ == "__main__":
    run()
