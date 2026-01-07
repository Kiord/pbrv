from typing import Protocol
import numpy as np
from pyrr import quaternion


class OrientationProvider(Protocol):
    def quat(self) -> np.ndarray: ...


class IdentityOrientation(OrientationProvider):
    def quat(self) -> np.ndarray:
        return quaternion.create(dtype=np.float32)
