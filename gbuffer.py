from dataclasses import dataclass, field
from typing import Tuple, Optional
import moderngl
import numpy as np

@dataclass
class GBuffer:
    ctx: moderngl.Context
    width: int
    height: int

    position: moderngl.Texture = field(init=False)
    normal:   moderngl.Texture = field(init=False)
    albedo:   moderngl.Texture = field(init=False)
    rmaos:    moderngl.Texture = field(init=False)
    emissive: moderngl.Texture = field(init=False)
    depth:    moderngl.Texture = field(init=False)
    fbo:      moderngl.Framebuffer = field(init=False)

    def __post_init__(self) -> None:
        self._create_resources()

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def _create_resources(self) -> None:
        self.position = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.normal   = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.albedo   = self.ctx.texture((self.width, self.height), 4, dtype="f1")
        self.rmaos    = self.ctx.texture((self.width, self.height), 4, dtype="f1")
        self.emissive = self.ctx.texture((self.width, self.height), 4, dtype="f1")

        self.depth    = self.ctx.depth_texture((self.width, self.height))

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.position, self.normal, self.albedo, self.rmaos, self.emissive],
            depth_attachment=self.depth,
        )

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return

        for tex in (
            getattr(self, "position", None),
            getattr(self, "normal", None),
            getattr(self, "albedo", None),
            getattr(self, "rmaos", None),
            getattr(self, "emissive", None),
            getattr(self, "depth", None),
        ):
            if tex is not None:
                tex.release()

        if getattr(self, "fbo", None) is not None:
            self.fbo.release()

        self.width, self.height = width, height
        self._create_resources()


    def sample_world_position(
        self,
        x: float,
        y: float,
    ) -> Optional[np.ndarray]:
 
        w, h = self.width, self.height

        ix = int(x)
        iy = int(h - 1 - int(y))

        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))

        raw = self.fbo.read(
            viewport=(ix, iy, 1, 1),
            components=4,
            attachment=0,  # position
            dtype="f2",    
            alignment=1,
        )

        pos = np.frombuffer(raw, dtype=np.float16)

        if pos.size < 3:
            return None

        world_pos = pos[:3].astype(np.float32)

        if np.any(world_pos>1.1):
            return None

        return world_pos