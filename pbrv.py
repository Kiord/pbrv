import sys

import argparse
from pathlib import Path
from typing import Optional, Tuple

from moderngl_window import run_window_config

from viewer import Viewer
from scene import Scene, Material, Mesh, EnvMap, Panorama, CubeMap


def parse_value_or_path(
    value: Optional[str],
    default_value: Tuple[float, ...],
    valid_lengths: Tuple[int, ...],
    param_name: str,
) -> tuple[Tuple[float, ...], Optional[Path]]:
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



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time PBR viewer (mesh + material + moderngl-window options)",
    )

    parser.add_argument(
        "mesh_path",
        type=Path,
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
        "--metalness","--metal", "-m",
        dest='metalness',
        metavar="VALUE_OR_PATH",
        help="Metalness scalar (0..1) OR metalness map path",
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
        "-ssao",
        "--use-ssao",
        dest='use_ssao',
        action='store_true',
        help="Enable SSAO",
    )
    parser.add_argument(
        "--envmap", '-em',
        dest="envmap_path",
        type=Path,
        metavar="PATH",
        help="Cubemap directory with right/left/top/bottom/front/back images. Or panorama image path",
    )

    args, mw_args = parser.parse_known_args()

    if not args.mesh_path.exists():
        parser.error(f"Mesh path does not exist: {args.mesh_path}")
    
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
            default_value=(0.3,),
            valid_lengths=(1,),
            param_name="--roughness",
        )
        roughness_value = roughness_vals[0]

        metalness_vals, metalness_map = parse_value_or_path(
            args.metalness,
            default_value=(0.0,),
            valid_lengths=(1,),
            param_name="--metalness",
        )
        metalness_value = metalness_vals[0]

    except ValueError as e:
        parser.error(str(e))


    envmap:Optional[EnvMap] = None
    if args.envmap_path is not None:
        if not args.envmap_path.exists():
            parser.error(f"{args.envmap_path} does not exist")
        is_cubemap = args.envmap_path.is_dir()
        is_panorama = args.envmap_path.is_file()
        if not(is_cubemap or is_panorama):
            parser.error(f"{args.envmap_path} is neither a file nor a directory")
        cls = CubeMap if is_cubemap else Panorama
        envmap = cls.from_path(str(args.envmap_path))

    mesh = Mesh.from_path(str(args.mesh_path))

    material = Material.from_map_paths(
        albedo_path=str(albedo_map) if albedo_map else None,
        normal_path=str(args.normal) if args.normal else None,
        roughness_path=str(roughness_map) if roughness_map else None,
        metalness_path=str(metalness_map) if metalness_map else None,
        ambient_occlusion_path=str(args.ao) if args.ao else None,
    )

    material.albedo = albedo_color
    material.roughness = roughness_value
    material.metalness = metalness_value

    Viewer.scene = Scene(mesh=mesh, material=material, envmap=envmap)
    Viewer.use_ssao = args.use_ssao

    
    sys.argv = sys.argv[:1]
    run_window_config(Viewer, args=mw_args)


if __name__ == "__main__":
    main()
