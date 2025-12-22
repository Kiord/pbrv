from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class OSMouse:
    """Thin wrapper around moderngl-window BaseWindow to set OS cursor position.
    Supports (best-effort):
      - glfw
      - pyglet
      - pygame2
      - sdl2
      - pyqt5
      - pyside2
      - tk

    Falls back to a no-op if the backend or underlying API is missing.
    """

    wnd: Any

    def __post_init__(self) -> None:
        self.backend: str = getattr(self.wnd, "name", "").lower()

    # ---------------- public API ----------------

    def set_position(self, x: float, y: float) -> None:
        """Set the OS cursor position in window coordinates."""

        if self.backend == "glfw":
            self._set_pos_glfw(x, y)
        elif self.backend == "pyglet":
            self._set_pos_pyglet(x, y)
        elif self.backend == "pygame2":
            self._set_pos_pygame2(x, y)
        elif self.backend == "sdl2":
            self._set_pos_sdl2(x, y)
        elif self.backend == "pyqt5":
            self._set_pos_pyqt5(x, y)
        elif self.backend == "pyside2":
            self._set_pos_pyside2(x, y)
        elif self.backend == "tk":
            self._set_pos_tk(x, y)
        else:
            # do nothing
            return

    def center(self) -> None:
        try:
            width, height = self.wnd.size
        except Exception:
            return
        self.set_position(width / 2.0, height / 2.0)


    def _set_pos_glfw(self, x: float, y: float) -> None:
        try:
            import glfw  # type: ignore
        except Exception:
            return

        window_handle: Optional[Any] = getattr(self.wnd, "_window", None)
        if window_handle is None:
            return

        try:
            glfw.set_cursor_pos(window_handle, float(x), float(y))
        except Exception:
            pass

    def _set_pos_pyglet(self, x: float, y: float) -> None:
        window_obj: Optional[Any] = getattr(self.wnd, "_window", None)
        if window_obj is None:
            return

        try:
            window_obj.set_mouse_position(int(x), int(y))
        except Exception:
            pass

    def _set_pos_pygame2(self, x: float, y: float) -> None:
        try:
            import pygame  # type: ignore
        except Exception:
            return

        try:
            pygame.mouse.set_pos((int(x), int(y)))
        except Exception:
            pass

    def _set_pos_sdl2(self, x: float, y: float) -> None:
        try:
            import sdl2  # type: ignore
        except Exception:
            return

        window_handle: Optional[Any] = getattr(self.wnd, "_window", None)
        if window_handle is None:
            return

        try:
            sdl2.SDL_WarpMouseInWindow(window_handle, int(x), int(y))
        except Exception:
            pass

    def _set_pos_pyqt5(self, x: float, y: float) -> None:
        window_obj: Optional[Any] = getattr(self.wnd, "_window", None)
        if window_obj is None:
            return

        try:
            from PyQt5.QtGui import QCursor  # type: ignore
            from PyQt5.QtCore import QPoint  # type: ignore
        except Exception:
            return

        try:
            global_pos = window_obj.mapToGlobal(QPoint(int(x), int(y)))
            QCursor.setPos(global_pos)
        except Exception:
            pass

    def _set_pos_pyside2(self, x: float, y: float) -> None:
        window_obj: Optional[Any] = getattr(self.wnd, "_window", None)
        if window_obj is None:
            return

        try:
            from PySide2.QtGui import QCursor  # type: ignore
            from PySide2.QtCore import QPoint  # type: ignore
        except Exception:
            return

        try:
            global_pos = window_obj.mapToGlobal(QPoint(int(x), int(y)))
            QCursor.setPos(global_pos)
        except Exception:
            pass

    def _set_pos_tk(self, x: float, y: float) -> None:
        window_obj: Optional[Any] = getattr(self.wnd, "_window", None)
        if window_obj is None:
            return

        try:
            window_obj.event_generate(
                "<Motion>", warp=True, x=int(x), y=int(y)
            )
        except Exception:
            pass
