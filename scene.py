from dataclasses import dataclass
from typing import Tuple, Optional
import moderngl
from moderngl import Context
from utils import load_rgb_image_auto
from constants import MAX_LUMINANCE

import numpy as np
import trimesh as tm
import cv2
import os


def _load_map(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None

    print(f'[Material] Loading {path}')
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[Material] Warning: could not load texture '{path}'")
        return None
    
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    return img

@dataclass
class PointLight:
    position: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class Material:
    albedo: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    roughness: float = 1.0
    metalness: float = 0.0
    specular: float = 0.3
    specular_tint: float = 0.0

    albedo_map: Optional[np.ndarray] = None
    normal_map: Optional[np.ndarray] = None
    roughness_map: Optional[np.ndarray] = None
    metalness_map: Optional[np.ndarray] = None
    specular_map: Optional[np.ndarray] = None
    ambient_occlusion_map: Optional[np.ndarray] = None

    @classmethod
    def from_map_paths(
        cls,
        albedo_path: Optional[str] = None,
        normal_path: Optional[str] = None,
        roughness_path: Optional[str] = None,
        metalness_path: Optional[str] = None,
        specular_path: Optional[str] = None,
        ambient_occlusion_path: Optional[str] = None,
    ) -> "Material":
        mat = cls()
        mat.load_maps(
            albedo_path,
            normal_path,
            roughness_path,
            metalness_path,
            specular_path,
            ambient_occlusion_path,
        )
        return mat

    def load_maps(
        self,
        albedo_path: Optional[str] = None,
        normal_path: Optional[str] = None,
        roughness_path: Optional[str] = None,
        metalness_path: Optional[str] = None,
        specular_path: Optional[str] = None,
        ambient_occlusion_path: Optional[str] = None,
    ):
        self.albedo_map = _load_map(albedo_path)
        self.normal_map = _load_map(normal_path)
        self.roughness_map = _load_map(roughness_path)
        self.metalness_map = _load_map(metalness_path)
        self.specular_map = _load_map(specular_path)
        self.ambient_occlusion_map = _load_map(ambient_occlusion_path)


@dataclass
class Mesh:
    vertices:   np.ndarray  # N 3 float32
    normals:    np.ndarray  # N 3 float32
    faces:      np.ndarray  # M 3 int32
    uv:         np.ndarray  # N 2 float32
    tangents:   np.ndarray  # N 3 float32

    @staticmethod
    def _compute_tangents(vertices: np.ndarray,
                          uv: np.ndarray,
                          faces: np.ndarray) -> np.ndarray:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        uv0 = uv[faces[:, 0]]
        uv1 = uv[faces[:, 1]]
        uv2 = uv[faces[:, 2]]

        edge1 = v1 - v0 
        edge2 = v2 - v0 

        duv1 = uv1 - uv0 
        duv2 = uv2 - uv0 

        denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        r = np.zeros_like(denom, dtype=np.float32)
        valid = np.abs(denom) > 1e-8
        r[valid] = 1.0 / denom[valid]

        tan = (duv2[:, 1][:, None] * edge1 - duv1[:, 1][:, None] * edge2) *  r[:, None]  

        t = np.zeros_like(vertices, dtype=np.float32)
        np.add.at(t, faces[:, 0], tan)
        np.add.at(t, faces[:, 1], tan)
        np.add.at(t, faces[:, 2], tan)

        norms = np.linalg.norm(t, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        t /= norms

        return t

    @classmethod
    def create_sphere(cls):
        mesh = tm.creation.uv_sphere(radius=1.0)
        v = mesh.vertices.astype(np.float64)
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        r = np.linalg.norm(v, axis=1)
        r[r == 0.0] = 1.0
        xn, yn, zn = x / r, y / r, z / r
        theta = np.arctan2(zn, xn)       
        phi   = np.arccos(np.clip(yn, -1.0, 1.0))
        u = (theta + np.pi) / (2.0 * np.pi)
        v = phi / np.pi                    
        uv = np.stack([u, v], axis=1)
        mesh.visual.uv = uv
        return Mesh.from_trimesh(cls, mesh)

    def from_trimesh(cls, mesh:tm.Trimesh):
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        scale = 2.0 / np.max(bounds[1] - bounds[0])
        vertices = (mesh.vertices - center) * scale
        normals = mesh.vertex_normals
        faces = mesh.faces

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uv = mesh.visual.uv
            uv[:, 1] = 1 - uv[:, 1]
            tangents = cls._compute_tangents(vertices, uv, faces)
        else:
            uv = np.zeros((len(vertices), 2), dtype=np.float32)
            tangents = np.zeros_like(normals)
        

        return cls(
            vertices=vertices.astype("f4"),
            normals=normals.astype("f4"),
            uv=uv.astype("f4"),
            tangents=tangents.astype("f4"),
            faces=faces.astype("i4"),
        )

    @classmethod
    def from_path(cls, mesh_path: str):
        mesh = tm.load_mesh(mesh_path)
        return Mesh.from_trimesh(cls, mesh)
      

    def to_gl(self, ctx:Context):
        data = np.hstack([self.vertices, self.normals, self.uv, self.tangents]).astype("f4")
        vbo = ctx.buffer(data.tobytes())
        ibo = ctx.buffer(self.faces.astype("i4").tobytes())
        return vbo, ibo

@dataclass
class Panorama:
    image: np.ndarray
    
    @classmethod
    def from_path(cls, image_path: str):
        image, _ = load_rgb_image_auto(image_path, out_f='f2')
        return cls(image=image)
    
    def to_gl(self, ctx:Context):
        h, w = self.image.shape[:2]
        data = np.clip(self.image, 0.0, MAX_LUMINANCE)
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

@dataclass
class CubeMap:
    front:  np.ndarray
    back:   np.ndarray
    left:   np.ndarray
    right:  np.ndarray
    top:    np.ndarray
    bottom: np.ndarray
    
    @classmethod
    def from_path(cls, cubemap_dir: str):
        front, suffix = load_rgb_image_auto(cubemap_dir + os.sep + 'front', out_f='f2') 
        back, _ = load_rgb_image_auto(cubemap_dir + os.sep + 'back', [suffix], out_f='f2') 
        right, _ = load_rgb_image_auto(cubemap_dir + os.sep + 'right', [suffix], out_f='f2') 
        left, _ = load_rgb_image_auto(cubemap_dir + os.sep + 'left', [suffix], out_f='f2') 
        top, _ = load_rgb_image_auto(cubemap_dir + os.sep + 'top', [suffix], out_f='f2') 
        bottom, _ = load_rgb_image_auto(cubemap_dir + os.sep + 'bottom', [suffix], out_f='f2')
        return cls(front=front, back=back, right=right, left=left, top=top, bottom=bottom)
    
    def to_gl(self, ctx:Context):
       
        faces = [
            self.right,
            self.left,
            self.top,
            self.bottom,
            self.front,
            self.back,
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
        #cube_tex.build_mipmaps()
        cube_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # cube_tex.repeat_x = True
        # cube_tex.repeat_y = True
        return cube_tex

EnvMap = Panorama | CubeMap


@dataclass
class Scene:
    mesh: Mesh
    material: Material
    envmap: Optional[EnvMap]=None
    point_light: Optional[PointLight]=None

