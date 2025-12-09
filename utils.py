from typing import Any, Callable, Optional, Sequence, Union
from pathlib import Path
from moderngl import Program, Context
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def safe_set_uniform(prog:Program, name: str, value: Any):
    if name in prog:
        if isinstance(value, np.ndarray):
            prog[name].write(value.tobytes())
        else:
            prog[name].value = value


class Pass:
    def __init__(self, ctx: Context, load_program_fn:Optional[Callable[..., Program]]=None):
        self.ctx = ctx
        if load_program_fn is None:
            load_program_fn = ctx.program
        self.load_program_fn = load_program_fn



def load_rgb_image_auto(
    base_path: Union[str, Path],
    ext_priority: Optional[Sequence[str]] = None,
    out_f:Optional[str]=None
) -> np.ndarray:
    if ext_priority is None:
        ext_priority = [
            ".exr",
            ".hdr",
            ".pfm",
            ".png",
            ".jpg",
            ".jpeg",
            ".tga",
            ".bmp",
            ".tif",
            ".tiff",
        ]

    ext_priority = [e.lower() for e in ext_priority]

    base_path = Path(base_path)
    parent = base_path.parent if base_path.parent != Path("") else Path(".")
    stem = base_path.stem if base_path.suffix else base_path.name

    candidates = []
    for p in parent.iterdir():
        if not p.is_file():
            continue
        if p.stem != stem:
            continue
        suffix = p.suffix.lower()
        if suffix in ext_priority:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No image found for base path '{base_path}' "
            f"with extensions {ext_priority}"
        )

    def ext_rank(p: Path) -> int:
        try:
            return ext_priority.index(p.suffix.lower())
        except ValueError:
            return len(ext_priority)

    candidates.sort(key=ext_rank)
    chosen:Path = candidates[0]
    suffix = chosen.suffix

    img = cv2.imread(str(chosen), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"OpenCV failed to load image: {chosen}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            img = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported channel count: {img.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    if out_f is not None:
        if img.dtype == np.uint8:
            img = img.astype("f4") / 255.0
        img = img.astype(out_f)

    return img, suffix