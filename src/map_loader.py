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
    raceline_mask: np.ndarray | None = None
    spawn_zones: np.ndarray | None = None   # uint8: 0=none, 1-3=zone id
    lookat_zones: np.ndarray | None = None  # uint8: 0=none, 1-3=zone id


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


def decode_zone_channel(channel: np.ndarray) -> np.ndarray:
    """Decode a PNG channel into zone IDs (uint8: 0=none, 1-3).

    Thresholds are midpoints between encoded values (255, 170, 85, 0):
      >=213 (mid 255/170) → zone 1
      >=128 (mid 170/85)  → zone 2
      >=43  (mid 85/0)    → zone 3
    Backward compatible: old binary 0/255 files decode as zone 0/1.
    """
    zones = np.zeros(channel.shape, dtype=np.uint8)
    zones[channel >= 213] = 1
    zones[(channel >= 128) & (channel < 213)] = 2
    zones[(channel >= 43) & (channel < 128)] = 3
    return zones


def encode_zone_channel(zones: np.ndarray) -> np.ndarray:
    """Encode zone IDs (uint8: 0-3) into a PNG channel value.

    Zone 1→255, zone 2→170, zone 3→85, none→0.
    """
    channel = np.zeros(zones.shape, dtype=np.uint8)
    channel[zones == 1] = 255
    channel[zones == 2] = 170
    channel[zones == 3] = 85
    return channel


def _load_zones_png(
    map_pgm_path: Path, expected_shape: Tuple[int, int]
) -> Tuple[
    np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None,
    np.ndarray | None, np.ndarray | None,
]:
    """Returns (spawn_mask, kill_mask, lookat_mask, raceline_mask, spawn_zones, lookat_zones)."""
    zones_path = map_pgm_path.with_name(map_pgm_path.stem + "_zones.png")
    if not zones_path.exists():
        return None, None, None, None, None, None
    if _PILImage is None:
        return None, None, None, None, None, None

    img = _PILImage.open(zones_path)
    has_alpha = img.mode == "RGBA"
    arr = np.asarray(img.convert("RGBA") if has_alpha else img.convert("RGB"))
    if arr.shape[:2] != expected_shape:
        raise ValueError(
            f"Zone file {zones_path.name} shape {arr.shape[:2]} "
            f"doesn't match map shape {expected_shape}"
        )
    kill_mask = arr[:, :, 0] >= 128
    spawn_zones = decode_zone_channel(arr[:, :, 1])
    lookat_zones = decode_zone_channel(arr[:, :, 2])
    spawn_mask = spawn_zones > 0
    lookat_mask = lookat_zones > 0
    raceline_mask = arr[:, :, 3] >= 128 if has_alpha else None
    return (
        spawn_mask if np.any(spawn_mask) else None,
        kill_mask if np.any(kill_mask) else None,
        lookat_mask if np.any(lookat_mask) else None,
        raceline_mask if raceline_mask is not None and np.any(raceline_mask) else None,
        spawn_zones if np.any(spawn_zones) else None,
        lookat_zones if np.any(lookat_zones) else None,
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
        spawn_mask, kill_mask, lookat_mask, raceline_mask, spawn_zones, lookat_zones = _load_zones_png(image_path, image.shape)
        if any(v is not None for v in (spawn_mask, kill_mask, lookat_mask, raceline_mask, spawn_zones, lookat_zones)):
            map_data = replace(map_data, spawn_mask=spawn_mask, kill_mask=kill_mask, lookat_mask=lookat_mask,
                               raceline_mask=raceline_mask, spawn_zones=spawn_zones, lookat_zones=lookat_zones)
        return map_data

    image = read_pgm(map_path)
    map_data = _map_data_from_meta(image, {})
    spawn_mask, kill_mask, lookat_mask, raceline_mask, spawn_zones, lookat_zones = _load_zones_png(map_path, image.shape)
    if any(v is not None for v in (spawn_mask, kill_mask, lookat_mask, raceline_mask, spawn_zones, lookat_zones)):
        map_data = replace(map_data, spawn_mask=spawn_mask, kill_mask=kill_mask, lookat_mask=lookat_mask,
                           raceline_mask=raceline_mask, spawn_zones=spawn_zones, lookat_zones=lookat_zones)
    return map_data
