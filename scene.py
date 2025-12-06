from dataclasses import dataclass
from typing import Tuple, Optional
import os

import numpy as np
import trimesh as tm
import cv2


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
class Material:
    albedo: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    roughness: float = 0.3
    metalicness: float = 0.1

    albedo_map: Optional[np.ndarray] = None
    normal_map: Optional[np.ndarray] = None
    roughness_map: Optional[np.ndarray] = None
    metalicness_map: Optional[np.ndarray] = None
    ambient_occlusion_map: Optional[np.ndarray] = None

    @classmethod
    def from_map_paths(
        cls,
        albedo_path: Optional[str] = None,
        normal_path: Optional[str] = None,
        roughness_path: Optional[str] = None,
        metalicness_path: Optional[str] = None,
        ambient_occlusion_path: Optional[str] = None,
    ) -> "Material":
        mat = cls()
        mat.load_maps(
            albedo_path,
            normal_path,
            roughness_path,
            metalicness_path,
            ambient_occlusion_path,
        )
        return mat

    def load_maps(
        self,
        albedo_path: Optional[str] = None,
        normal_path: Optional[str] = None,
        roughness_path: Optional[str] = None,
        metalicness_path: Optional[str] = None,
        ambient_occlusion_path: Optional[str] = None,
    ):
        self.albedo_map = _load_map(albedo_path)
        self.normal_map = _load_map(normal_path)
        self.roughness_map = _load_map(roughness_path)
        self.metalicness_map = _load_map(metalicness_path)
        self.ambient_occlusion_map = _load_map(ambient_occlusion_path)


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3), float32
    normals: np.ndarray   # (N, 3), float32
    faces: np.ndarray     # (M, 3), int32
    uv: np.ndarray        # (N, 2), float32
    tangents: np.ndarray  # (N, 3), float32

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

        denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]  # (M,)
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
    def from_path(cls, mesh_path: str):
        mesh = tm.load_mesh(mesh_path)
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


@dataclass
class Scene:
    mesh: Mesh
    material: Material
