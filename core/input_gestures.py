# core/input_gestures.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Protocol

import numpy as np
from pyrr import matrix44, matrix33, quaternion as Q
from enum import Enum, auto

from core.trackball import Trackball
from core.camera import Camera
from core.mouse import OSMouse
from moderngl_window.context.base import BaseWindow


@dataclass
class DoubleClickDetector:
    max_delay: float = 0.30
    _armed: bool = False
    _last_time: float = 0.0
    _last_pos: Tuple[int, int] = (-1, -1)

    def feed(self, x: int, y: int) -> bool:
        now = time.perf_counter()
        if not self._armed:
            self._arm(now, x, y)
            return False

        dt = now - self._last_time
        if dt <= self.max_delay and self._pixel_distance(x, y) <= 2:
            self._reset()
            return True

        self._arm(now, x, y)
        return False

    def _pixel_distance(self, x: int, y: int) -> int:
        return abs(x - self._last_pos[0]) + abs(y - self._last_pos[1])

    def _arm(self, t: float, x: int, y: int) -> None:
        self._armed = True
        self._last_time = t
        self._last_pos = (x, y)

    def _reset(self) -> None:
        self._armed = False
        self._last_time = 0.0
        self._last_pos = (-1, -1)


class DragRotateGesture:
    def __init__(self):
        self._pending_rotate = False
        self._rotating = False
        self._press_xy = (0, 0)
        self._press_wh = (1, 1)

    def on_press(self, x: int, y: int, w: int, h: int) -> None:
        self._pending_rotate = True
        self._rotating = False
        self._press_xy = (x, y)
        self._press_wh = (w, h)

    def on_drag(self, rotator, x: int, y: int, w: int, h: int) -> None:
        if self._pending_rotate and not self._rotating:
            px, py = self._press_xy
            pw, ph = self._press_wh
            rotator.begin_rotate(px, py, pw, ph)
            self._rotating = True

        if self._rotating:
            rotator.rotate(x, y, w, h)

    def on_release(self, rotator) -> None:
        if self._rotating:
            rotator.end_rotate()
        self._pending_rotate = False
        self._rotating = False

    def cancel(self) -> None:
        self._pending_rotate = False
        self._rotating = False


class Modifiers:
    __slots__ = ("shift", "ctrl", "alt")

    def __init__(self):
        self.shift = False
        self.ctrl = False
        self.alt = False

    def set_from(self, mods) -> None:
        self.shift = bool(getattr(mods, "shift", self.shift))
        self.ctrl = bool(getattr(mods, "ctrl", self.ctrl))
        self.alt = bool(getattr(mods, "alt", self.alt))


class Manipulator(Protocol):
    def cancel(self) -> None: ...
    def on_press(self, x: int, y: int, w: int, h: int) -> None: ...
    def on_drag(self, x: int, y: int, w: int, h: int) -> None: ...
    def on_release(self) -> None: ...


class OrbitManipulator:
    def __init__(self, camera: Camera, orbit: Trackball):
        self.camera = camera
        self.orbit = orbit
        self.gesture = DragRotateGesture()

    def cancel(self) -> None:
        self.gesture.cancel()

    def on_press(self, x: int, y: int, w: int, h: int) -> None:
        self.gesture.on_press(x, y, w, h)

    def on_drag(self, x: int, y: int, w: int, h: int) -> None:
        self.gesture.on_drag(self.orbit, x, y, w, h)
        self.camera.mark_view_dirty()

    def on_release(self) -> None:
        self.gesture.on_release(self.orbit)
        self.camera.mark_view_dirty()


class ArcballWorldManipulator:
    def __init__(self, camera: Camera, *, ball_size: float, env: bool):
        self.camera = camera
        self.env = env
        self.tb = Trackball(ball_size=ball_size)
        self.gesture = DragRotateGesture()

        self._base_quat = np.array([0, 0, 0, 1], dtype=np.float32)
        self.quat = np.array([0, 0, 0, 1], dtype=np.float32)

        self.matrix = (
            matrix33.create_identity(dtype=np.float32)
            if self.env
            else matrix44.create_identity(dtype=np.float32)
        )

    def cancel(self) -> None:
        self.gesture.cancel()

    def on_press(self, x: int, y: int, w: int, h: int) -> None:
        self._base_quat = self.quat.copy()
        self.tb.reset_rotation()
        self.gesture.on_press(x, y, w, h)

    def on_drag(self, x: int, y: int, w: int, h: int) -> None:
        self.gesture.on_drag(self.tb, x, y, w, h)

        q_cam = self.camera.orientation.quat()
        q_cam_conj = Q.conjugate(q_cam)
        q_world_delta = Q.cross(q_cam, Q.cross(self.tb.quat(), q_cam_conj))

        if not self.env:
            self.quat = Q.normalize(Q.cross(self._base_quat, q_world_delta))
            self.matrix = matrix44.create_from_quaternion(self.quat)
        else:
            self.quat = Q.normalize(Q.cross(Q.conjugate(q_world_delta), self._base_quat))
            self.matrix = matrix33.create_from_quaternion(self.quat)

    def on_release(self) -> None:
        self.gesture.on_release(self.tb)


class Mode(Enum):
    CAMERA = auto()
    MODEL = auto()
    ENV = auto()


class CameraInputController:
    def __init__(
        self,
        wnd: BaseWindow,
        camera: Camera,
        orbit: Trackball,
        zoom_sensitivity: float = 0.1,
        double_click_delay: float = 0.30,
        ball_size: float = 0.8,
        pick_world_position: Optional[Callable[[int, int], Optional[np.ndarray]]] = None,
    ):
        self.wnd = wnd
        self.camera = camera
        self.zoom_sensitivity = float(zoom_sensitivity)

        self.os_mouse = OSMouse(self.wnd)
        self.double = DoubleClickDetector(max_delay=float(double_click_delay))

        self.modifiers = Modifiers()
        self._pick_world_position = pick_world_position

        self._panning = False
        self._active_mode = Mode.CAMERA

        self._orbit = OrbitManipulator(camera, orbit)
        self._model = ArcballWorldManipulator(camera, ball_size=ball_size, env=False)
        self._env = ArcballWorldManipulator(camera, ball_size=ball_size, env=True)

        self.lod_factor = 0.0
        self._lod_factor_speed = 0.01

    @property
    def model_matrix(self) -> np.ndarray:
        return self._model.matrix

    @property
    def env_matrix(self) -> np.ndarray:
        return self._env.matrix

    def _choose_mode(self) -> Mode:
        if self.modifiers.ctrl:
            return Mode.MODEL
        if self.modifiers.shift:
            return Mode.ENV
        return Mode.CAMERA

    def _active_manipulator(self) -> Manipulator:
        if self._active_mode == Mode.MODEL:
            return self._model
        if self._active_mode == Mode.ENV:
            return self._env
        return self._orbit

    def _cancel_all_rotations(self) -> None:
        self._orbit.cancel()
        self._model.cancel()
        self._env.cancel()

    def _is_object(self, x: int, y: int) -> bool:
        if self._pick_world_position is None:
            return False
        p = self._pick_world_position(x, y)
        return p is not None

    def _on_double_click(self, x: int, y: int) -> None:
        if self._pick_world_position is None:
            return
        p = self._pick_world_position(x, y)
        if p is not None:
            self.camera.set_pivot(p)

    def on_press(self, x: int, y: int, button) -> None:
        self.modifiers.set_from(self.wnd.modifiers)
        self._active_mode = self._choose_mode()

        if self.double.feed(x, y):
            self._on_double_click(x, y)
            self._cancel_all_rotations()
            return

        if button == self.wnd.mouse.right:
            self._panning = True
            return

        w, h = self.wnd.size
        self._active_manipulator().on_press(x, y, w, h)

    def on_drag(self, x: int, y: int, dx: int, dy: int) -> None:
        w, h = self.wnd.size
        if self._panning:
            self.camera.pan(dx, dy, w, h)
            return
        self._active_manipulator().on_drag(x, y, w, h)

    def on_release(self, x: int, y: int, button) -> None:
        if button == self.wnd.mouse.right:
            self._panning = False
            return
        if button == self.wnd.mouse.left:
            self._active_manipulator().on_release()

    def on_scroll(self, y_offset: float) -> None:
        self.camera.zoom(-float(y_offset) * self.zoom_sensitivity)

    def on_key_event(self, key, action, modifiers) -> None:
        self.modifiers.set_from(modifiers)
