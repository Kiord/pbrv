import time
from dataclasses import dataclass
from typing import Callable, Optional
from gbuffer import GBuffer
from mouse import OSMouse

@dataclass
class DoubleClickDetector:
    max_delay: float = 0.30
    _armed: bool = False
    _last_time: float = 0.0

    def feed(self, now: float | None = None) -> bool:
        if now is None:
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

    def on_press(self, x: int, y: int, w: int, h: int) -> bool:
        if self.double.feed():
            self.cancel_rotate()
            return True

        self._pending_rotate = True
        self._press_xy = (x, y)
        self._press_wh = (w, h)
        return False

    def on_drag(self, camera, x: int, y: int, w: int, h: int) -> None:
        if self._pending_rotate and not self._rotating:
            px, py = self._press_xy
            pw, ph = self._press_wh
            camera.begin_rotate(px, py, pw, ph)
            self._rotating = True

        if self._rotating:
            camera.rotate(x, y, w, h)

    def on_release(self, camera) -> None:
        if self._rotating:
            camera.end_rotate()
        self._pending_rotate = False
        self._rotating = False

    def cancel_rotate(self) -> None:
        self._pending_rotate = False
        self._rotating = False


class CameraInputController:

    def __init__(
        self,
        wnd,
        camera,
        zoom_sensitivity: float = 0.2,
        double_click_delay: float = 0.30,
    ):
        self.wnd = wnd
        self.camera = camera
        self.zoom_sensitivity = zoom_sensitivity
        self.os_mouse = OSMouse(self.wnd)
        self.left = LeftClickGesture(double_delay=double_click_delay)
        self._panning = False

    def on_press(self, x: int, y: int, gbuffer:GBuffer, button) -> None:
        w, h = self.wnd.size

        if button == self.wnd.mouse.left:
            if self.left.on_press(x, y, w, h):
                self._on_double_click(x, y, gbuffer)
            return

        if button == self.wnd.mouse.right:
            self._panning = True
            self.left.cancel_rotate()
            return

    def on_drag(self, x: int, y: int, dx: int, dy: int) -> None:
        w, h = self.wnd.size

        if self._panning:
            self.camera.pan(dx, dy, w, h)
        else:
            self.left.on_drag(self.camera, x, y, w, h)

    def on_release(self, x: int, y: int, button) -> None:
        if button == self.wnd.mouse.left:
            self.left.on_release(self.camera)
            return

        if button == self.wnd.mouse.right:
            self._panning = False
            return

    def on_scroll(self, y_offset: float) -> None:
        self.camera.zoom(y_offset * self.zoom_sensitivity)

    def _on_double_click(self, x:int, y:int, gbuffer:GBuffer):
        picked_pos = gbuffer.sample_world_position(x, y)
        if picked_pos is not None:
            self.camera.set_pivot(picked_pos)
            self.os_mouse.center()