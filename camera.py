import numpy as np
from pyrr import matrix44, quaternion
from trackball import Trackball
from constants import UP, FRONT, EPSILON
from typing import Tuple
import math

# New

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
        far: float = 10.0,
        min_distance: float = 0.005,
        max_distance: float = 5.0,
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

        self.projection = matrix44.create_perspective_projection(
            self.fov_deg, float(aspect), self.near, self.far
        )

    # --- projection ---
    def resize(self, width: int, height: int):
        if height <= 0:
            height = 1
        aspect = width / float(height)
        self.projection = matrix44.create_perspective_projection(
            self.fov_deg, aspect, self.near, self.far
        )

    # --- navigation ---
    def set_pivot(self, new_pivot: Tuple[float, float, float]):
        _, eye, _ = self.get_view()
        new_pivot = np.asanyarray(new_pivot, dtype=np.float32)
        self.distance = float(np.linalg.norm(eye - new_pivot))
        self.pivot = new_pivot

    def zoom(self, delta: float):
        factor = 1.0 - self.zoom_speed * float(delta)
        self.distance *= factor
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))

    def pan(self, dx: float, dy: float, width: int, height: int):
        if width <= 0 or height <= 0:
            return

        _, eye, up = self.get_view()

        fwd = self.pivot - eye
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm < EPSILON:
            return
        fwd /= fwd_norm

        right = np.cross(fwd, up)
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

    def get_view(self):
        if hasattr(self, 'get_quat'):
            quat = self.get_quat()
        else:
            quat = quaternion.create(dtype=np.float32)

        offset = -FRONT * self.distance
        
        eye = self.pivot + quaternion.apply_to_vector(quat, offset)
        up = quaternion.apply_to_vector(quat, UP)

        view = matrix44.create_look_at(eye, self.pivot, up)
        return view, eye, up


class TrackballCamera(Trackball, Camera):
    """Camera = Trackball orientation + Camera model"""

    def __init__(self, *, ball_size: float = 0.8, **kwargs):
        super().__init__(ball_size=ball_size, **kwargs)
