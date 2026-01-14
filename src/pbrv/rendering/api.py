from typing import Protocol, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from pbrv.core.camera import Camera

class Renderer(Protocol):
    name: str

    def initialize(self, size: Tuple[int, int]) -> None: ...
    def resize(self, size: Tuple[int, int]) -> None: ...
    def set_scene(self, scene) -> None: ...
    def render(self, frame) -> None: ...
    def shutdown(self) -> None: ...

    def pick_world_position(self, x: int, y: int) -> Optional[np.ndarray]: ...


@dataclass
class FrameState:
    time:float
    camera:Camera
    model_matrix: np.ndarray
    env_matrix: np.ndarray
    light_matrix: np.ndarray
    exposure: float
    tone_mapping: str
    env_lod_factor: Optional[float]
    use_ssao:Optional[bool]
    use_bloom:Optional[bool]
    window_size:Tuple[int, int]

