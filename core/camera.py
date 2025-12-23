import math
import numpy as np
from pyrr import matrix44, quaternion

from core.constants import UP, FRONT, EPSILON
from core.orientation import OrientationProvider, IdentityOrientation


class Camera:
    def __init__(
        self,
        *,
        aspect: float,
        orientation: OrientationProvider | None = None,
        fov_deg: float = 60.0,
        near: float = 0.05,
        far: float = 1000.0,
        distance: float = 2.0,
        pivot: np.ndarray | None = None,
    ):
        self.aspect = float(aspect)
        self.fov_deg = float(fov_deg)
        self.near = float(near)
        self.far = float(far)

        self.distance = float(distance)
        self.pivot = np.array([0.0, 0.0, 0.0], dtype=np.float32) if pivot is None else pivot.astype(np.float32)

        self.orientation: OrientationProvider = orientation if orientation is not None else IdentityOrientation()

        self._proj_dirty = True
        self._view_dirty = True

        self._proj = matrix44.create_identity(dtype=np.float32)
        self._inv_proj = matrix44.create_identity(dtype=np.float32)

        self._view = matrix44.create_identity(dtype=np.float32)
        self._inv_view = matrix44.create_identity(dtype=np.float32)

        self._eye = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._up = UP.copy()

    def mark_view_dirty(self) -> None:
        self._view_dirty = True

    def resize(self, width: int, height: int):
        self.aspect = width / float(max(1, height))
        self._proj_dirty = True

    def set_pivot(self, new_pivot: np.ndarray):
        new_pivot = np.asarray(new_pivot, dtype=np.float32)
        # Use property to ensure eye is valid
        self.distance = float(np.linalg.norm(self.eye - new_pivot))
        self.pivot = new_pivot
        self._view_dirty = True

    def zoom(self, delta: float):
        self.distance = max(0.01, self.distance + float(delta))
        self._view_dirty = True

    def pan(self, dx: float, dy: float, width: int, height: int):
        # Use properties to ensure view state is built
        eye = self.eye
        upv = self.up

        fwd = self.pivot - eye
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < EPSILON:
            return
        fwd /= fwd_len

        right = np.cross(fwd, upv)
        r_len = np.linalg.norm(right)
        if r_len < EPSILON:
            return
        right /= r_len

        up = np.cross(right, fwd)
        up_len = np.linalg.norm(up)
        if up_len < EPSILON:
            return
        up /= up_len

        fov_rad = math.radians(self.fov_deg)
        aspect = width / float(max(1, height))

        world_per_pixel_y = 0.5 * 2.0 * self.distance * math.tan(fov_rad / 2.0) / float(max(1, height))
        world_per_pixel_x = 0.5 * world_per_pixel_y * aspect

        pan_world = (-right * dx * world_per_pixel_x) + (up * dy * world_per_pixel_y)
        self.pivot += pan_world
        self._view_dirty = True

    def _rebuild_proj(self):
        self._proj = matrix44.create_perspective_projection(self.fov_deg, self.aspect, self.near, self.far)
        self._inv_proj = np.linalg.inv(self._proj)
        self._proj_dirty = False

    def _rebuild_view_eye_up(self):
        quat = self.orientation.quat()

        offset = -FRONT * self.distance
        self._eye = self.pivot + quaternion.apply_to_vector(quat, offset)
        self._up = quaternion.apply_to_vector(quat, UP)

        self._view = matrix44.create_look_at(self._eye, self.pivot, self._up)
        self._inv_view = np.linalg.inv(self._view)
        self._view_dirty = False

    @property
    def proj(self) -> np.ndarray:
        if self._proj_dirty:
            self._rebuild_proj()
        return self._proj

    @property
    def inv_proj(self) -> np.ndarray:
        if self._proj_dirty:
            self._rebuild_proj()
        return self._inv_proj

    @property
    def view(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view_eye_up()
        return self._view

    @property
    def inv_view(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view_eye_up()
        return self._inv_view

    @property
    def eye(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view_eye_up()
        return self._eye

    @property
    def up(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view_eye_up()
        return self._up
