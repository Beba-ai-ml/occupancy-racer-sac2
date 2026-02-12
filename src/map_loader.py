from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

from .config import load_yaml, resolve_from


@dataclass(frozen=True)
class MapData:
    image: np.ndarray
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float
    free_mask: np.ndarray
    occupied_mask: np.ndarray
    spawn_mask: np.ndarray | None = None
    kill_mask: np.ndarray | None = None
    lookat_mask: np.ndarray | None = None


DEFAULT_MAP_META = {
    "resolution": 0.05,
    "origin": (-16.4, -13.6, 0.0),
    "negate": 0,
    "occupied_thresh": 0.65,
    "free_thresh": 0.25,
}


def read_pgm(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        magic = handle.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise ValueError(f"Unsupported PGM format: {magic}")

        def token_reader():
            while True:
                line = handle.readline()
                if not line:
                    return
                line = line.split(b"#", 1)[0]
                for token in line.split():
                    yield token

        tokens = token_reader()
        try:
            width = int(next(tokens))
            height = int(next(tokens))
            maxval = int(next(tokens))
        except StopIteration as exc:
            raise ValueError("Invalid PGM header") from exc

        if magic == b"P5":
            data = handle.read(width * height)
            if len(data) != width * height:
                raise ValueError("PGM data is shorter than expected")
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        else:
            values = [int(token) for token in tokens]
            if len(values) < width * height:
                raise ValueError("PGM ASCII data is shorter than expected")
            image = np.array(values[: width * height], dtype=np.uint16).reshape((height, width))

        if maxval != 255:
            image = (image.astype(np.float32) * (255.0 / maxval)).round().astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        return image


def _map_data_from_meta(image: np.ndarray, meta: dict) -> MapData:
    resolution = float(meta.get("resolution", DEFAULT_MAP_META["resolution"]))
    origin = tuple(meta.get("origin", DEFAULT_MAP_META["origin"]))
    negate = int(meta.get("negate", DEFAULT_MAP_META["negate"]))
    occupied_thresh = float(meta.get("occupied_thresh", DEFAULT_MAP_META["occupied_thresh"]))
    free_thresh = float(meta.get("free_thresh", DEFAULT_MAP_META["free_thresh"]))

    occ = image.astype(np.float32) / 255.0
    if negate == 0:
        occ = 1.0 - occ
        white_mask = image >= 250
    else:
        white_mask = image <= 5
    free_mask = (occ < free_thresh) & white_mask
    occupied_mask = occ >= occupied_thresh

    return MapData(
        image=image,
        resolution=resolution,
        origin=origin,
        negate=negate,
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh,
        free_mask=free_mask,
        occupied_mask=occupied_mask,
    )


def _load_zones_png(
    map_pgm_path: Path, expected_shape: Tuple[int, int]
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    zones_path = map_pgm_path.with_name(map_pgm_path.stem + "_zones.png")
    if not zones_path.exists():
        return None, None, None
    if _PILImage is None:
        return None, None, None

    img = _PILImage.open(zones_path).convert("RGB")
    arr = np.asarray(img)
    if arr.shape[:2] != expected_shape:
        raise ValueError(
            f"Zone file {zones_path.name} shape {arr.shape[:2]} "
            f"doesn't match map shape {expected_shape}"
        )
    kill_mask = arr[:, :, 0] >= 128
    spawn_mask = arr[:, :, 1] >= 128
    lookat_mask = arr[:, :, 2] >= 128
    return (
        spawn_mask if np.any(spawn_mask) else None,
        kill_mask if np.any(kill_mask) else None,
        lookat_mask if np.any(lookat_mask) else None,
    )


def load_map(map_path: Path) -> MapData:
    if map_path.suffix.lower() in {".yaml", ".yml"}:
        meta = load_yaml(map_path)

        image_rel = meta.get("image")
        if not image_rel:
            raise ValueError("Map YAML missing 'image' field")

        image_path = resolve_from(map_path.parent, image_rel)
        image = read_pgm(image_path)
        map_data = _map_data_from_meta(image, meta)
        spawn_mask, kill_mask, lookat_mask = _load_zones_png(image_path, image.shape)
        if spawn_mask is not None or kill_mask is not None or lookat_mask is not None:
            map_data = replace(map_data, spawn_mask=spawn_mask, kill_mask=kill_mask, lookat_mask=lookat_mask)
        return map_data

    image = read_pgm(map_path)
    map_data = _map_data_from_meta(image, {})
    spawn_mask, kill_mask, lookat_mask = _load_zones_png(map_path, image.shape)
    if spawn_mask is not None or kill_mask is not None or lookat_mask is not None:
        map_data = replace(map_data, spawn_mask=spawn_mask, kill_mask=kill_mask, lookat_mask=lookat_mask)
    return map_data
