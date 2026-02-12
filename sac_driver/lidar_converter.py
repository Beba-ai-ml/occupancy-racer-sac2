"""Convert ROS2 LaserScan into training-aligned lidar rays."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)


def _get_attr(obj: object, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise AttributeError(f"scan_msg missing required field: {name}")


@dataclass
class LidarConverter:
    target_angles_deg: Sequence[float]
    max_range_m: float
    angle_offset_deg: float = 0.0
    angle_direction: float = 1.0
    use_interpolation: bool = True

    def _to_scan_angle_rad(self, target_angle_deg: float) -> float:
        # Convert target angle into scan frame with optional offset and direction.
        angle_deg = self.angle_direction * (float(target_angle_deg) + float(self.angle_offset_deg))
        return math.radians(angle_deg)

    def convert(self, scan_msg: object) -> np.ndarray:
        angle_min = float(_get_attr(scan_msg, "angle_min"))
        angle_increment = float(_get_attr(scan_msg, "angle_increment"))
        ranges = _get_attr(scan_msg, "ranges")
        ranges_np = np.asarray(ranges, dtype=np.float32)

        if ranges_np.size == 0:
            _LOGGER.warning("LaserScan ranges is empty; returning max-range vector.")
            return np.ones(len(self.target_angles_deg), dtype=np.float32)
        if angle_increment <= 0:
            raise ValueError("LaserScan angle_increment must be > 0.")

        max_range = float(self.max_range_m)
        out = np.empty(len(self.target_angles_deg), dtype=np.float32)

        for i, target_angle_deg in enumerate(self.target_angles_deg):
            target_rad = self._to_scan_angle_rad(target_angle_deg)
            idx_float = (target_rad - angle_min) / angle_increment

            if idx_float < 0 or idx_float > (ranges_np.size - 1):
                _LOGGER.warning(
                    "Target angle %.2f deg outside scan range; using max range.",
                    target_angle_deg,
                )
                dist = max_range
            else:
                if self.use_interpolation:
                    idx0 = int(math.floor(idx_float))
                    idx1 = min(idx0 + 1, ranges_np.size - 1)
                    w = idx_float - idx0
                    v0 = float(ranges_np[idx0])
                    v1 = float(ranges_np[idx1])
                    dist = (1.0 - w) * v0 + w * v1
                else:
                    idx = int(round(idx_float))
                    dist = float(ranges_np[idx])

                if not math.isfinite(dist) or dist <= 0.0:
                    dist = max_range
                elif dist > max_range:
                    dist = max_range

            out[i] = dist / max_range

        return np.clip(out, 0.0, 1.0)


__all__ = ["LidarConverter"]
