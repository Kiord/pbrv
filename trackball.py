import math
import numpy as np
from pyrr import matrix44, quaternion

SQRT_TWO = math.sqrt(2.0)
EPSILON = 1e-8

class Trackball:
    """From https://basilisk.fr/src/gl/trackball.c
    API:
        begin(x, y, width, height)
        drag(x, y, width, height)
        end()
        get_matrix() -> Matrix44
    """

    def __init__(self, ball_size: float = 0.8):
        self._quat = quaternion.create(dtype=np.float32)
        self._start_quat = quaternion.create(dtype=np.float32)

        self._p1 = np.zeros(2, dtype=np.float32)

        self._dragging = False
        self.ball_size = float(ball_size)

    @staticmethod
    def _map_to_ndc(x: float, y: float, width: int, height: int):
        if width <= 0 or height <= 0:
            return 0.0, 0.0
        nx = (2.0 * x - width) / float(width)
        ny = (height - 2.0 * y) / float(height)
        return float(nx), float(ny)


    @staticmethod
    def _project_to_sphere(r: float, x: float, y: float) -> float:
        d = float(math.hypot(x, y))

        inside_sphere_threshold = r * (SQRT_TWO / 2.0)

        if d < inside_sphere_threshold:  # inside sphere
            return float(math.sqrt(r * r - d * d))

        # on hyperbola
        t = r / SQRT_TWO
        return float((t * t) / d)


    def begin(self, x: float, y: float, width: int, height: int):
        self._start_quat = self._quat.copy()
        self._p1[:] = self._map_to_ndc(x, y, width, height)
        self._dragging = True

    def drag(self, x: float, y: float, width: int, height: int):
        if not self._dragging:
            return

        p1x, p1y = float(self._p1[0]), float(self._p1[1])
        p2x, p2y = self._map_to_ndc(x, y, width, height)

        if p1x == p2x and p1y == p2y:
            self._quat = self._start_quat.copy()
            return

        p1 = np.array(
            [p1x, p1y, self._project_to_sphere(self.ball_size, p1x, p1y)],
            dtype=np.float32,
        )
        p2 = np.array(
            [p2x, p2y, self._project_to_sphere(self.ball_size, p2x, p2y)],
            dtype=np.float32,
        )

        axis = np.cross(p2, p1)
        axis_len = float(np.linalg.norm(axis))
        if axis_len < EPSILON:
            self._quat = self._start_quat.copy()
            return
        axis /= axis_len

        d_vec = p1 - p2
        d = float(np.linalg.norm(d_vec))
        denom = 2.0 * self.ball_size
        t = d / denom
        t = max(-1.0, min(1.0, t))
        angle = 2.0 * math.asin(t)

        q_drag = quaternion.create_from_axis_rotation(axis, angle)

        self._quat = quaternion.cross(self._start_quat, q_drag)
        

    def end(self):
        self._dragging = False

    def get_matrix(self) -> np.ndarray:
        return matrix44.create_from_quaternion(self._quat)

    def get_quat(self) -> np.ndarray:
        return self._quat
