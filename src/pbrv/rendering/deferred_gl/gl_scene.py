from typing import Tuple, Union
from dataclasses import dataclass
import moderngl
from moderngl import Buffer, Context, TextureCube, Texture
import numpy as np
from pbrv.core.scene import Mesh, CubeMap, Panorama, EnvMap
from pbrv.core.constants import MAX_LUMINANCE

@dataclass
class GLMesh:
    vbo: Buffer
    ibo: Buffer

def upload_mesh(ctx:Context, mesh: Mesh)->GLMesh:
    data = np.hstack([mesh.vertices, mesh.normals, mesh.uv, mesh.tangents]).astype("f4")
    vbo = ctx.buffer(data.tobytes())
    ibo = ctx.buffer(mesh.faces.astype("i4").tobytes())
    return GLMesh(vbo, ibo)


def upload_cubemap(ctx:Context, cubemap:CubeMap)->TextureCube:
    faces = [
        cubemap.right,
        cubemap.left,
        cubemap.top,
        cubemap.bottom,
        cubemap.front,
        cubemap.back,
    ]

    h, w, c = faces[0].shape
    for f in faces[1:]:
        if f.shape != faces[0].shape:
            raise ValueError("All cubemap faces must have the same size")

    # Pack faces in +X, -X, +Y, -Y, +Z, -Z order
    data = np.concatenate(
        [f.reshape(-1, c) for f in faces],
        axis=0,
    )
    data = np.clip(data, 0.0, MAX_LUMINANCE)

    cube_tex = ctx.texture_cube(
        (w, h),
        components=c,
        data=data.astype("f2").tobytes(),
        dtype="f2", 
    )
    cube_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return cube_tex

def upload_panorama(ctx:Context, panorama:Panorama)->Texture:
    h, w = panorama.image.shape[:2]
    data = np.clip(panorama.image, 0.0, MAX_LUMINANCE)
    pano_tex = ctx.texture(
        (w, h),
        components=3,
        data=data.tobytes(),
        dtype="f2",
    )
    pano_tex.build_mipmaps()
    pano_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    pano_tex.repeat_x = True
    pano_tex.repeat_y = True
    
    return pano_tex


def upload_envmap(ctx:Context, envmap:EnvMap)->Union[Texture, TextureCube]:
    if isinstance(envmap, Panorama):
        return upload_panorama(ctx, envmap)
    if isinstance(envmap, CubeMap):
        return upload_cubemap(ctx, envmap)
    assert False, f"Wrong env map type: {type(envmap)}"