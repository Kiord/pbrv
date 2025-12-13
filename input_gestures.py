import time
from dataclasses import dataclass

import numpy as np
from pyrr import matrix44 as M, quaternion as Q

from trackball import Trackball
from camera import TrackballCamera
from mouse import OSMouse
from moderngl_window.context.base import BaseWindow

from typing import Callable
from enum import Enum, auto


@dataclass
class DoubleClickDetector:
    max_delay: float = 0.30
    _armed: bool = False
    _last_time: float = 0.0

    def feed(self) -> bool:
        now = time.perf_counter()

        if not self._armed:
            self._armed = True
            self._last_time = now
            return False

        dt = now - self._last_time
        if dt <= self.max_delay:
            self.reset()
            return True

        self._armed = True
        self._last_time = now
        return False

    def reset(self) -> None:
        self._armed = False
        self._last_time = 0.0


class LeftClickGesture:

    def __init__(self, double_delay: float = 0.30):
        self.double = DoubleClickDetector(double_delay)
        self._pending_rotate = False
        self._rotating = False
        self._press_xy = (0, 0)
        self._press_wh = (1, 1)

    def on_press(self, x: int, y: int, w: int, h: int, can_arm_double:bool|None=None) -> bool:
        if can_arm_double is not None and can_arm_double:
            if self.double.feed():
                self.cancel_rotate()
                return True

        self._pending_rotate = True
        self._press_xy = (x, y)
        self._press_wh = (w, h)
        return False

    def on_drag(self, rotator:Trackball, x: int, y: int, w: int, h: int) -> None:
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

    def cancel_rotate(self) -> None:
        self._pending_rotate = False
        self._rotating = False



class Mode(Enum):
    CAMERA = auto()
    MODEL = auto()
    ENV = auto()
    NONE = auto()

class Modifiers:
    __slots__ = ("shift", "ctrl", "alt")

    def __init__(self):
        self.shift = False
        self.ctrl = False
        self.alt = False

    def set_from(self, mods) -> None:
        self.shift = bool(getattr(mods, "shift", self.shift))
        self.ctrl  = bool(getattr(mods, "ctrl",  self.ctrl))
        self.alt   = bool(getattr(mods, "alt",   self.alt))

class CameraInputController:

    
    def __init__(
        self,
        wnd: BaseWindow,
        camera: TrackballCamera,
        zoom_sensitivity: float = 0.2,
        double_click_delay: float = 0.30,
        ball_size: float = 0.8,
        sample_world_position: Callable[[int,int], bool] | None = None
    ):
        self.wnd = wnd
        self.camera = camera
        self.zoom_sensitivity = zoom_sensitivity
        self.os_mouse = OSMouse(self.wnd)
        self.left = LeftClickGesture(double_delay=double_click_delay)
        self._panning = False

        self._model_tb = Trackball(ball_size=ball_size)
        self._env_tb = Trackball(ball_size=ball_size)

        self.model_quat = np.array([0, 0, 0, 1], dtype=np.float32)
        self.env_quat = np.array([0, 0, 0, 1], dtype=np.float32)

        self.model_matrix = M.create_identity(dtype=np.float32)
        self.env_matrix = M.create_identity(dtype=np.float32)

        self._active = Mode.CAMERA 
        self._base_quat = np.array([0, 0, 0, 1], dtype=np.float32)

        self.lod_factor = 0
        self._lod_factor_speed = 0.01

        self._sample_world_position = sample_world_position

        self.left = LeftClickGesture(double_delay=double_click_delay)

        self.modifiers = Modifiers()

    def _choose_active(self) -> str:
        if self.modifiers.ctrl:
            return  Mode.MODEL
        if self.modifiers.shift:
            return Mode.ENV
        return Mode.CAMERA

    def on_press(self, x: int, y: int, button) -> None:
        w, h = self.wnd.size
        if button == self.wnd.mouse.left:
            self._active = self._choose_active()

            if self._active == Mode.MODEL:
                self._base_quat = self.model_quat.copy()
                self._model_tb.reset_rotation()
            elif self._active == Mode.ENV:
                self._base_quat = self.env_quat.copy()
                self._env_tb.reset_rotation()

            can_arm_double = self._is_object(x, y)
            if self.left.on_press(x, y, w, h, can_arm_double):
                self.left.cancel_rotate()
                self._active = Mode.CAMERA
                self._on_double_click(x, y)
            return

        if button == self.wnd.mouse.right:
            self._panning = True
            self.left.cancel_rotate()
            return

    def on_drag(self, x: int, y: int, dx: int, dy: int) -> None:
        w, h = self.wnd.size

        if self._panning:
            self.camera.pan(dx, dy, w, h)
            return

        if self._active == Mode.CAMERA:
            self.left.on_drag(self.camera, x, y, w, h)
            return
        
        tb = self._model_tb if self._active == Mode.MODEL else self._env_tb
        self.left.on_drag(tb, x, y, w, h)

        q_cam = self.camera.get_quat()
        q_cam_conj = Q.conjugate(q_cam)
        q_world_delta = Q.cross(q_cam, Q.cross(tb.get_quat(), q_cam_conj))

        if self._active == Mode.MODEL:
            # local/object feel
            self.model_quat = Q.normalize(Q.cross(self._base_quat, q_world_delta))
            self.model_matrix = M.create_from_quaternion(self.model_quat)
        else:
            # env feel: inverse delta, applied in world space
            self.env_quat = Q.normalize(Q.cross(Q.conjugate(q_world_delta), self._base_quat))
            self.env_matrix = M.create_from_quaternion(self.env_quat)

    def on_release(self, x: int, y: int, button) -> None:
        if button == self.wnd.mouse.left:
            if self._active == Mode.CAMERA:
                self.left.on_release(self.camera)
            elif self._active == Mode.MODEL:
                self.left.on_release(self._model_tb)
            else:
                self.left.on_release(self._env_tb)
            self._active = Mode.CAMERA
            return

        if button == self.wnd.mouse.right:
            self._panning = False
            return

    def on_scroll(self, y_offset: float) -> None:
        if self.modifiers.shift:
            self.lod_factor += y_offset * self._lod_factor_speed
            self.lod_factor = min(1, max(0, self.lod_factor))
        else:
            self.camera.zoom(y_offset * self.zoom_sensitivity)

    
    def _on_double_click(self, x: int, y: int):
        if self._sample_world_position is None:
            return None
        picked_pos = self._sample_world_position(x, y)
        if picked_pos is not None:
            self.camera.set_pivot(picked_pos)
            self.os_mouse.center()

    def _is_object(self, x, y):
        if self._sample_world_position is None:
            return None
        return self._sample_world_position(x, y) is not None
    
    def on_key_event(self, key, action, modifiers):
        self.modifiers.set_from(modifiers)