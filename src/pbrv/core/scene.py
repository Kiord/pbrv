from dataclasses import dataclass
from typing import Tuple, Optional
from pbrv.core.utils import uv_sphere
from pbrv.core.loading import load_image, load_image_auto, load_mesh
from pbrv.core.constants import EPSILON
from pbrv.core.sun_extraction import SunExtractSettings, SunExtraction, extract_sun_from_cubemap, extract_sun_from_panorama

import numpy as np
import trimesh as tm
import os


@dataclass
class Light:
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class DirectionalLight(Light):
    direction: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    view_proj: Optional[np.ndarray] = None

@dataclass
class PointLight(Light):
    position: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Material:
    albedo: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    roughness: float = 1.0
    metallic: float = 0.0
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    specular: float = 0.3
    specular_tint: float = 0.0

    albedo_map: Optional[np.ndarray] = None
    normal_map: Optional[np.ndarray] = None
    roughness_map: Optional[np.ndarray] = None
    metallic_map: Optional[np.ndarray] = None
    emissive_map: Optional[np.ndarray] = None
    specular_map: Optional[np.ndarray] = None
    ambient_occlusion_map: Optional[np.ndarray] = None

    @classmethod
    def from_map_paths(
        cls,
        albedo_path: Optional[str] = None,
        normal_path: Optional[str] = None,
        roughness_path: Optional[str] = None,
        metallic_path: Optional[str] = None,
        emissive_path: Optional[str] = None,
        specular_path: Optional[str] = None,
        ambient_occlusion_path: Optional[str] = None,
    ) -> "Material":
        mat = cls()
        mat.load_maps(
            albedo_path,
            normal_path,
            roughness_path,
            metallic_path,
            emissive_path,
            specular_path,
            ambient_occlusion_path,
        )
        return mat

    def load_maps(
        self,
        albedo_path: Optional[str] = None,
        normal_path: Optional[str] = None,
        roughness_path: Optional[str] = None,
        metallic_path: Optional[str] = None,
        emissive_path: Optional[str] = None,
        specular_path: Optional[str] = None,
        ambient_occlusion_path: Optional[str] = None,
    ):
        self.albedo_map = load_image(albedo_path, 'f4')
        self.normal_map = load_image(normal_path, 'f4')
        self.roughness_map = load_image(roughness_path, 'f4')
        self.metallic_map = load_image(metallic_path, 'f4')
        self.emissive_map = load_image(emissive_path, 'f4')
        self.specular_map = load_image(specular_path, 'f4')
        self.ambient_occlusion_map = load_image(ambient_occlusion_path, 'f4')


@dataclass
class Mesh:
    vertices:   np.ndarray  # N 3 float32
    normals:    np.ndarray  # N 3 float32
    faces:      np.ndarray  # M 3 int32
    uv:         np.ndarray  # N 2 float32
    tangents:   np.ndarray  # N 3 float32

    @staticmethod
    def _compute_tangents(vertices: np.ndarray,
                          normals: np.ndarray,
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

        t:np.ndarray = np.zeros_like(vertices, dtype=np.float32)
        np.add.at(t, faces[:, 0], tan)
        np.add.at(t, faces[:, 1], tan)
        np.add.at(t, faces[:, 2], tan)

        t = t  / (np.linalg.norm(t, axis=1, keepdims=True) + EPSILON)

        t = t - normals * (t*normals).sum(axis=1, keepdims=True)
        t = t  / (np.linalg.norm(t, axis=1, keepdims=True) + EPSILON)

        return t

    @classmethod
    def create_sphere(cls):
        v, f, uv = uv_sphere()
        t = cls._compute_tangents(v, v, uv, f)
        return cls(v, v, f, uv, t)

    @classmethod
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
            tangents = cls._compute_tangents(vertices, normals, uv, faces)
        else:
            uv = np.zeros((len(vertices), 2), dtype=np.float32)
            tangents = np.zeros_like(normals)
            tangents[:, 0] = 1.0
            tangents = np.cross(tangents, normals, axis=-1)
            tangents = tangents  / (np.linalg.norm(tangents, axis=1, keepdims=True) + EPSILON)

        
        return cls(
            vertices=vertices.astype("f4"),
            normals=normals.astype("f4"),
            uv=uv.astype("f4"),
            tangents=tangents.astype("f4"),
            faces=faces.astype("i4"),
        )

    @classmethod
    def from_path(cls, mesh_path: str):
        mesh = load_mesh(mesh_path)
        return cls.from_trimesh(mesh)
    
@dataclass
class Panorama:
    image: np.ndarray
    
    @classmethod
    def from_path(cls, image_path: str):
        image, _ = load_image_auto(image_path, out_f='f2')
        return cls(image=image)
    

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
        front, suffix = load_image_auto(cubemap_dir + os.sep + 'front', out_f='f2') 
        back, _ = load_image_auto(cubemap_dir + os.sep + 'back', [suffix], out_f='f2') 
        right, _ = load_image_auto(cubemap_dir + os.sep + 'right', [suffix], out_f='f2') 
        left, _ = load_image_auto(cubemap_dir + os.sep + 'left', [suffix], out_f='f2') 
        top, _ = load_image_auto(cubemap_dir + os.sep + 'top', [suffix], out_f='f2') 
        bottom, _ = load_image_auto(cubemap_dir + os.sep + 'bottom', [suffix], out_f='f2')
        return cls(front=front, back=back, right=right, left=left, top=top, bottom=bottom)
    

EnvMap = Panorama | CubeMap | Light


@dataclass
class Scene:
    mesh: Mesh
    material: Material
    envmap: Optional[EnvMap]=None
    point_light: Optional[PointLight]=None
    dir_light: Optional[DirectionalLight]=None
    sun: Optional[SunExtraction]=None

    def auto_sun(self, settings: SunExtractSettings = SunExtractSettings())->None:
        if self.dir_light is not None:
            print("[Warning] Calling auto sun will overwrite the existing directional light.")

        if self.envmap is None:
            self.sun = None
            print("[Warning] Cancelling auto sun because no environment map is set.")
            return
        
        if isinstance(self.envmap, Panorama):
            sun = extract_sun_from_panorama(self.envmap.image, settings)
        elif isinstance(self.envmap, CubeMap):
            faces = [
                self.envmap.right, self.envmap.left,
                self.envmap.top, self.envmap.bottom,
                self.envmap.front, self.envmap.back,
            ]  #  +X,-X,+Y,-Y,+Z,-Z
            sun = extract_sun_from_cubemap(faces, settings)

        self.sun = sun
        if sun is None:
            return

        color = 10 * sun.color_integral # more realistic when boosted...
        self.dir_light = DirectionalLight(
            direction=-sun.direction.astype(np.float32),
            color=color.astype(np.float32),
        )
