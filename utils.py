from typing import Any, Callable, Optional, Sequence, Union
from pathlib import Path
from moderngl import Program, Context
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import trimesh as tm
from trimesh.visual.texture import TextureVisuals
from trimesh.grouping import merge_vertices

def safe_set_uniform(prog:Program, name: str, value: Any):
    if name in prog:
        if isinstance(value, np.ndarray):
            prog[name].write(value.tobytes())
        else:
            prog[name].value = value
        return
    #print(f'WARNING, \"{name}\" not found') 


class Pass:
    def __init__(self, ctx: Context, load_program_fn:Optional[Callable[..., Program]]=None):
        self.ctx = ctx
        if load_program_fn is None:
            load_program_fn = ctx.program
        self.load_program_fn = load_program_fn



def load_rgb_image_auto(
    base_path: Union[str, Path],
    ext_priority: Optional[Sequence[str]] = None,
    out_f:Optional[str]=None
) -> np.ndarray:
    if ext_priority is None:
        ext_priority = [
            ".exr",
            ".hdr",
            ".pfm",
            ".png",
            ".jpg",
            ".jpeg",
            ".tga",
            ".bmp",
            ".tif",
            ".tiff",
        ]

    ext_priority = [e.lower() for e in ext_priority]

    base_path = Path(base_path)
    parent = base_path.parent if base_path.parent != Path("") else Path(".")
    stem = base_path.stem if base_path.suffix else base_path.name

    candidates = []
    for p in parent.iterdir():
        if not p.is_file():
            continue
        if p.stem != stem:
            continue
        suffix = p.suffix.lower()
        if suffix in ext_priority:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No image found for base path '{base_path}' "
            f"with extensions {ext_priority}"
        )

    def ext_rank(p: Path) -> int:
        try:
            return ext_priority.index(p.suffix.lower())
        except ValueError:
            return len(ext_priority)

    candidates.sort(key=ext_rank)
    chosen:Path = candidates[0]
    suffix = chosen.suffix

    img = cv2.imread(str(chosen), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"OpenCV failed to load image: {chosen}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            img = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported channel count: {img.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    if out_f is not None:
        if img.dtype == np.uint8:
            img = img.astype("f4") / 255.0
        img = img.astype(out_f)

    return img, suffix


def uv_sphere(n_lat=32, n_lon=64, dtype=np.float32):
    if n_lon < 3 or n_lat < 3:
        raise ValueError("n_lon and n_lon must be >= 3 for a valid sphere mesh.")
    tm.creation.uv_sphere
    n_lon = int(n_lon)
    n_lat = int(n_lat)

    theta = np.linspace(0.0, 2.0 * np.pi, n_lon + 1, endpoint=True, dtype=dtype)
    u = theta / (2.0 * np.pi)  

    phi = np.linspace(0.0, np.pi, n_lat + 1, endpoint=True, dtype=dtype)
    phi_inner = phi[1:-1] 
    v_inner = 1.0 - (phi_inner / np.pi) 

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    north_v = np.array([0.0, 1.0, 0.0], dtype=dtype)[None, :].repeat(n_lon + 1, axis=0)
    north_uv = np.stack([u, np.ones_like(u)], axis=1).astype(dtype, copy=False)

   
    sin_p = np.sin(phi_inner)[:, None]  # n_lat-1 1
    cos_p = np.cos(phi_inner)[:, None]
    x = sin_p * cos_t[None, :]
    y = cos_p * np.ones_like(cos_t[None, :])
    z = sin_p * sin_t[None, :]
    rings_v = np.stack([x, y, z], axis=2).reshape(-1, 3).astype(dtype, copy=False)

    uu = np.broadcast_to(u[None, :], (phi_inner.size, n_lon + 1))
    vv = np.broadcast_to(v_inner[:, None], (phi_inner.size, n_lon + 1))
    rings_uv = np.stack([uu, vv], axis=2).reshape(-1, 2).astype(dtype, copy=False)

    south_v = np.array([0.0, -1.0, 0.0], dtype=dtype)[None, :].repeat(n_lon + 1, axis=0)
    south_uv = np.stack([u, np.zeros_like(u)], axis=1).astype(dtype, copy=False)

    vertices = np.vstack([north_v, rings_v, south_v])
    uv = np.vstack([north_uv, rings_uv, south_uv])

    row = n_lon + 1
    ring_start = row                      
    south_start = n_lat * row             
    last_ring_start = (n_lat - 1) * row   

    j = np.arange(n_lon, dtype=np.int32)

    top_faces = np.stack([j, ring_start + j + 1, ring_start + j], axis=1).astype(np.int32, copy=False)

    bands = n_lat - 2  
    if bands > 0:
        ii = np.arange(bands, dtype=np.int32)[:, None]  #  bands 1
        jj = j[None, :]                                 #  1 n_lon

        a = ring_start + ii * row + jj
        b = a + 1
        c = a + row
        d = c + 1

        f1 = np.stack([a, c, b], axis=2).reshape(-1, 3)
        f2 = np.stack([b, c, d], axis=2).reshape(-1, 3)
        mid_faces = np.vstack([f1, f2]).astype(np.int32, copy=False)
    else:
        mid_faces = np.empty((0, 3), dtype=np.int32)

    bottom_faces = np.stack([south_start + j, last_ring_start + j, last_ring_start + j + 1],
                            axis=1).astype(np.int32, copy=False)

    faces = np.vstack([top_faces, mid_faces, bottom_faces])
    return vertices, faces, uv