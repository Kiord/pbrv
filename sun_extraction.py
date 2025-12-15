
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple
import numpy as np

import cv2 
from constants import REL_LUMINANCE, MAX_LUMINANCE, EPSILON




@dataclass
class SunExtractSettings:
    pano_size: Tuple[int, int] = (256, 128)  # (W, H)
    cube_face_size: int = 128
    threshold_mode: Literal["quantile", "relmax"] = "quantile"
    threshold_value: float = 0.998  # quantile if "quantile", or factor if "relmax" (e.g. 0.9)
    power: float = 6.0
    angle_percentile: float = 95.0 
    min_selected: int = 16


@dataclass
class SunExtraction:
    direction: np.ndarray     
    color_integral: np.ndarray
    omega: float              
    exclude_cos: float
    peak_luminance: float    
    feather: float = 0.02


def extract_sun_from_panorama(
    pano_rgb: np.ndarray,
    settings: SunExtractSettings = SunExtractSettings(),
) -> Optional[SunExtraction]:

    img = _prep_rgb(pano_rgb, MAX_LUMINANCE)
    img = _resize_rgb(img, settings.pano_size)

    H, W, _ = img.shape
    lum = _luminance(img)
    peak = float(np.max(lum))
    if peak <= EPSILON:
        return None

    thr = _threshold(lum, settings.threshold_mode, settings.threshold_value)
    mask = lum >= thr
    if int(mask.sum()) < settings.min_selected:
        return None

    # Directions + solid angle per pixel for lat-long
    uu = (np.arange(W, dtype=np.float32) + 0.5) / float(W)
    vv = (np.arange(H, dtype=np.float32) + 0.5) / float(H)
    U, V = np.meshgrid(uu, vv)  # H W

    dirs = _dir_from_latlong(U, V)  # H W 3
    domega = _domega_latlong(U, V, W, H)  # H W

    return _reduce_sun(mask, dirs, img, lum, domega, peak, settings)


def extract_sun_from_cubemap(
    cube_faces: Sequence[np.ndarray],
    settings: SunExtractSettings = SunExtractSettings(),
) -> Optional[SunExtraction]:
    if len(cube_faces) != 6:
        raise ValueError("Need 6 faces in (+X,-X,+Y,-Y,+Z,-Z) order")

    faces = [_prep_rgb(f, MAX_LUMINANCE) for f in cube_faces]
    faces = [_resize_rgb(f, (settings.cube_face_size, settings.cube_face_size)) for f in faces]

    N = settings.cube_face_size
    uu = (np.arange(N, dtype=np.float32) + 0.5) / float(N)
    vv = (np.arange(N, dtype=np.float32) + 0.5) / float(N)
    U, V = np.meshgrid(uu, vv)  # NxN

    # Precompute cubemap solid angle per texel
    domega = _domega_cubemap(U, V, N)  # NxN

    # Find global peak luminance + threshold
    lum_faces = []
    peak = 0.0
    for f in faces:
        lum = _luminance(f)
        lum_faces.append(lum)
        peak = max(peak, float(np.max(lum)))
    if peak <= EPSILON:
        return None

    # threshold should be global, so we flatten all lums
    all_lum = np.concatenate([l.reshape(-1) for l in lum_faces], axis=0)
    thr = _threshold(all_lum, settings.threshold_mode, settings.threshold_value)

    # Accumulate across faces
    wsum = 0.0
    dir_sum = np.zeros(3, dtype=np.float64)
    color_integral = np.zeros(3, dtype=np.float64)
    omega = 0.0
    picked_dirs = []

    for face_idx, (rgb, lum) in enumerate(zip(faces, lum_faces)):
        mask = lum >= thr
        if int(mask.sum()) < 1:
            continue

        dirs = _face_uv_to_dir(face_idx, U, V)  # NxNx3
        picked_dirs.append(dirs[mask].reshape(-1, 3))

        w = _weights(lum, thr, domega, settings.power)
        wmask = w[mask]
        wsum += float(wmask.sum())

        dir_sum += (dirs[mask] * wmask[:, None]).sum(axis=0)
        color_integral += (rgb[mask] * domega[mask, None]).sum(axis=0)
        omega += float(domega[mask].sum())

    if wsum <= EPSILON:
        return None

    sun_dir = (dir_sum / wsum).astype(np.float32)
    sun_dir /= (np.linalg.norm(sun_dir) + EPSILON)

    exclude_cos = _estimate_exclude_cos(sun_dir, np.concatenate(picked_dirs, axis=0), settings.angle_percentile)

    return SunExtraction(
        direction=sun_dir,
        color_integral=color_integral.astype(np.float32),
        omega=float(omega),
        exclude_cos=float(exclude_cos),
        peak_luminance=float(peak),
    )


# Shared helpers

def _prep_rgb(rgb: np.ndarray, max_luminance: Optional[float]) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if max_luminance is not None:
        x = np.clip(x, 0.0, float(max_luminance))
    return x


def _resize_rgb(rgb: np.ndarray, out_size_wh: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = int(out_size_wh[0]), int(out_size_wh[1])
    h, w = rgb.shape[:2]
    if (w, h) == (out_w, out_h):
        return rgb
    return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ REL_LUMINANCE


def _threshold(lum: np.ndarray, mode: str, value: float) -> float:
    if mode == "quantile":
        q = float(np.clip(value, 0.0, 1.0))
        return float(np.quantile(lum, q))
    elif mode == "relmax":
        f = float(np.clip(value, 0.0, 1.0))
        return float(np.max(lum)) * f
    else:
        raise ValueError(f"Unknown threshold_mode: {mode}")


def _weights(lum: np.ndarray, thr: float, domega: np.ndarray, power: float) -> np.ndarray:
    return np.power(np.clip(lum - thr, 0.0, None), float(power)) * domega


def _reduce_sun(
    mask: np.ndarray,
    dirs: np.ndarray,
    rgb: np.ndarray,
    lum: np.ndarray,
    domega: np.ndarray,
    peak: float,
    settings: SunExtractSettings,
) -> SunExtraction:
    thr = float(np.min(lum[mask]))  # same as used to build mask originally (close enough)
    w = _weights(lum, thr, domega, settings.power)
    wmask = w[mask]
    wsum = float(wmask.sum())
    if wsum <= EPSILON:
        raise RuntimeError("Sun reduction failed: wsum <= 0")

    sun_dir = (dirs[mask] * wmask[:, None]).sum(axis=0) / wsum
    sun_dir = sun_dir.astype(np.float32)
    sun_dir /= (np.linalg.norm(sun_dir) + EPSILON)

    omega = float(domega[mask].sum())
    color_integral = (rgb[mask] * domega[mask, None]).sum(axis=0).astype(np.float32)  # ∫ L dω

    picked_dirs = dirs[mask].reshape(-1, 3)
    exclude_cos = _estimate_exclude_cos(sun_dir, picked_dirs, settings.angle_percentile)

    return SunExtraction(
        direction=sun_dir,
        color_integral=color_integral,
        omega=omega,
        exclude_cos=float(exclude_cos),
        peak_luminance=float(peak),
    )

def _dir_from_latlong(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    lon = (U - 0.5) * (2.0 * np.pi)
    lat = (0.5 - V) * np.pi
    coslat = np.cos(lat)
    return np.stack([coslat * np.cos(lon), np.sin(lat), coslat * np.sin(lon)], axis=-1).astype(np.float32)


def _domega_latlong(U: np.ndarray, V: np.ndarray, W: int, H: int) -> np.ndarray:
    dlon = (2.0 * np.pi) / float(W)
    dlat = np.pi / float(H)
    lat = (0.5 - V) * np.pi
    domega = dlon * dlat * np.cos(lat)
    return np.clip(domega, 0.0, None).astype(np.float32)


def _face_uv_to_dir(face: int, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    s = U * 2.0 - 1.0
    t = V * 2.0 - 1.0

    if face == 0:   # +X (right)
        d = np.stack([np.ones_like(s), -t, -s], axis=-1)
    elif face == 1: # -X (left)
        d = np.stack([-np.ones_like(s), -t,  s], axis=-1)
    elif face == 2: # +Y (top)
        d = np.stack([ s,  np.ones_like(s),  t], axis=-1)
    elif face == 3: # -Y (bottom)
        d = np.stack([ s, -np.ones_like(s), -t], axis=-1)
    elif face == 4: # +Z (front)
        d = np.stack([ s, -t,  np.ones_like(s)], axis=-1)
    elif face == 5: # -Z (back)
        d = np.stack([-s, -t, -np.ones_like(s)], axis=-1)
    else:
        raise ValueError("face must be 0..5")

    n = np.linalg.norm(d, axis=-1, keepdims=True) + EPSILON
    return (d / n).astype(np.float32)


def _domega_cubemap(U: np.ndarray, V: np.ndarray, N: int) -> np.ndarray:
    s = U * 2.0 - 1.0
    t = V * 2.0 - 1.0
    delta = 2.0 / float(N)
    denom = np.power(1.0 + s * s + t * t, 1.5)
    return (delta * delta / denom).astype(np.float32)


def _estimate_exclude_cos(sun_dir: np.ndarray, picked_dirs: np.ndarray, angle_percentile: float) -> float:
    cosang = np.clip(picked_dirs @ sun_dir, -1.0, 1.0)
    ang = np.arccos(cosang)
    theta = float(np.percentile(ang, float(angle_percentile)))
    return float(np.cos(theta))
