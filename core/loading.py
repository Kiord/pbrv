import cv2
import numpy as np
from typing import Optional, Union, Sequence
from pathlib import Path
import trimesh as tm

def load_image(path: Optional[str],
               out_f:Optional[str]=None) -> Optional[np.ndarray]:
    if path is None:
        return None

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"OpenCV failed to load image: {path}")

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

    return img

def load_image_auto(
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

    img  = load_image(chosen, out_f)
    
    return img, suffix


def load_mesh(path: Union[str, Path])->tm.Trimesh:
    return tm.load_mesh(path, process=False)
    