import numpy as np
from pyrr import matrix44, quaternion
from core.trackball import Trackball
from core.constants import UP, FRONT, EPSILON
from typing import Tuple
import math
from functools import cached_property

class Camera:
    """Pure camera model: pivot + distance + projection + view/pan/zoom.
    Requires an orientation provider implementing get_quat().
    """

    def __init__(
        self,
        *,
        pivot: Tuple[float, float, float] = (0, 0, 0),
        distance: float = 3.0,
        fov_deg: float = 60.0,
        aspect: float = 16.0 / 9.0,
        near: float = 0.001,
        far: float = 22.0,
        min_distance: float = 0.005,
        max_distance: float = 20.0,
        zoom_speed: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pivot = np.asanyarray(pivot, dtype=np.float32)

        self.distance = float(distance)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.zoom_speed = float(zoom_speed)

        self.fov_deg = float(fov_deg)
        self.near = float(near)
        self.far = float(far)
        self.aspect = float(aspect)

        # self.projection = matrix44.create_perspective_projection(
        #     self.fov_deg, float(aspect), self.near, self.far
        # )

        # self.view = 

        self._proj = None
        self._inv_proj = None
        self._view = None
        self._inv_view = None
        self._up = None
        self._eye = None
        self._view_dirty = True
        self._proj_dirty = True

  

    # --- projection ---
    def resize(self, width: int, height: int):
        if height <= 0:
            height = 1
        self.aspect = width / float(height)
        self._proj_dirty = True

    # --- navigation ---
    def set_pivot(self, new_pivot: Tuple[float, float, float]):
        new_pivot = np.asanyarray(new_pivot, dtype=np.float32)
        self.distance = float(np.linalg.norm(self._eye - new_pivot))
        self.pivot = new_pivot
        self._view_dirty = True

    def zoom(self, delta: float):
        factor = 1.0 - self.zoom_speed * float(delta)
        self.distance *= factor
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
        self._view_dirty = True

    def pan(self, dx: float, dy: float, width: int, height: int):
        if width <= 0 or height <= 0:
            return


        fwd = self.pivot - self._eye
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm < EPSILON:
            return
        fwd /= fwd_norm

        right = np.cross(fwd, self._up)
        r_len = float(np.linalg.norm(right))
        if r_len < EPSILON:
            return
        right /= r_len

        up = np.cross(right, fwd)
        up_len = float(np.linalg.norm(up))
        if up_len < EPSILON:
            return
        up /= up_len

        fov_rad = math.radians(self.fov_deg)
        aspect = width / float(height)

        world_per_pixel_y = 0.5 * 2.0 * self.distance * math.tan(fov_rad / 2.0) / float(height)
        world_per_pixel_x = 0.5 * world_per_pixel_y * aspect

        pan_world = (-right * dx * world_per_pixel_x) + (up * dy * world_per_pixel_y)
        self.pivot += pan_world

        self._view_dirty = True

    
    def _rebuild_proj(self):
        self._proj = matrix44.create_perspective_projection(
            self.fov_deg, self.aspect, self.near, self.far
        )
        self._inv_proj = np.linalg.inv(self._proj)
        self._proj_dirty = False

    def _rebuild_view_eye_up(self):
        if hasattr(self, 'get_quat'):
            quat = self.get_quat()
        else:
            quat = quaternion.create(dtype=np.float32)

        offset = -FRONT * self.distance
        
        self._eye = self.pivot + quaternion.apply_to_vector(quat, offset)
        self._up = quaternion.apply_to_vector(quat, UP)

        self._view = matrix44.create_look_at(self._eye, self.pivot, self._up)
        self._inv_view = np.linalg.inv(self._view)

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
    def eye(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view_eye_up()
        return self._eye
    
    @property
    def up(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view_eye_up()
        return self._up

    @property
    def inv_view(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view_eye_up()
        return self._inv_view


class TrackballCamera(Trackball, Camera):
    """Camera = Trackball orientation + Camera model"""

    def __init__(self, *, ball_size: float = 0.8, **kwargs):
        super().__init__(ball_size=ball_size, **kwargs)
