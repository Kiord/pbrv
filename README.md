# pbrv

`pbrv` (PBR Viewer) is a lightweight CLI program to quickly visualize 3D PBR assets.

[<img src="resources/misc/snapshot.jpg">]()

## Usage

`python pbrv.py [--albedo VALUE_OR_PATH] [--normal PATH] [--roughness VALUE_OR_PATH] [--metalness VALUE_OR_PATH] [--ambient-occlusion PATH] [-ssao] [--envmap PATH] mesh_path `

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
