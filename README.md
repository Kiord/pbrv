# pbrv

`pbrv` (PBR Viewer) is a small CLI program to quickly visualize 3D PBR assets.

[<img src="src/pbrv/resources/misc/snapshot.jpg">]()

## Installation

```
python -m pip install -e .
```

## Usage

```
pbrv [--albedo VALUE_OR_PATH] [--normal PATH] [--roughness VALUE_OR_PATH] [--metallic VALUE_OR_PATH] [--ambient-occlusion PATH] [-ssao] [--envmap PATH] mesh_path 
```


The window is a [moderngl window](https://github.com/moderngl/moderngl-window) so you can also use its arguments. For instance, set GLFW backend by adding `--window glfw`.

## Features/Specs
- Interactive window
    - Trackball camera (with path indepedance)
    - Left click to rotate
    - Right click to pan
    - Double click to focus
- Deffered Shading
- Metal/roughness workflow
- Normal mapping (tangent space)
- SSAO (if no AO map)
- Image based lighting
    - Cubemaps
    - Equirectangular panoramas (converted to cubemap)
    - Cubemap prefiltering
        - Irradiance (Cosine)
        - Specular (GGX)

## Dependencies
- moderngl
- moderngl-window
- trimesh
- numpy
- pyrr
- opencv-python
