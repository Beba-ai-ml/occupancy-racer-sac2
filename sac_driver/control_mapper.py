"""Map policy actions to vehicle command structures (offline, ROS-agnostic)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class ControlMapper:
    max_steering_angle_deg: float
    max_speed_mps: float
    max_accel_mps2: float
    speed_limit_mps: Optional[float] = None
    steer_rate_limit_deg_s: Optional[float] = None
    accel_rate_limit_mps2: Optional[float] = None
    safe_mode: bool = False
    safe_speed_limit_mps: Optional[float] = None
    safe_steer_scale: float = 0.5
    safe_accel_scale: float = 0.5
    wheelbase_m: Optional[float] = None
    max_yaw_rate_rad_s: Optional[float] = None
    default_dt: float = 0.1

    def __post_init__(self) -> None:
        self._max_steer_rad = math.radians(float(self.max_steering_angle_deg))
        self._last_steer_rad = 0.0
        self._last_accel = 0.0

    def _effective_speed_limit(self) -> float:
        base = float(self.speed_limit_mps) if self.speed_limit_mps is not None else float(self.max_speed_mps)
        if self.safe_mode and self.safe_speed_limit_mps is not None:
            return min(base, float(self.safe_speed_limit_mps))
        return base

    def _apply_rate_limit(self, value: float, last: float, rate_limit: Optional[float], dt: float) -> float:
        if rate_limit is None or rate_limit <= 0.0:
            return value
        delta = rate_limit * dt
        return _clamp(value, last - delta, last + delta)

    def map_to_ackermann(
        self,
        steering_raw: float,
        accel_raw: float,
        current_speed: float,
        dt: Optional[float] = None,
    ) -> Dict[str, float]:
        if dt is None:
            dt = self.default_dt
        dt = max(float(dt), 1e-4)

        steer_norm = _clamp(float(steering_raw), -1.0, 1.0)
        steer_rad = steer_norm * self._max_steer_rad
        accel = _clamp(float(accel_raw), -self.max_accel_mps2, self.max_accel_mps2)

        if self.safe_mode:
            steer_rad *= float(self.safe_steer_scale)
            accel *= float(self.safe_accel_scale)

        steer_rate = None
        if self.steer_rate_limit_deg_s is not None:
            steer_rate = math.radians(float(self.steer_rate_limit_deg_s))
        steer_rad = self._apply_rate_limit(steer_rad, self._last_steer_rad, steer_rate, dt)
        accel = self._apply_rate_limit(accel, self._last_accel, self.accel_rate_limit_mps2, dt)

        speed_limit = self._effective_speed_limit()
        speed_cmd = float(current_speed) + accel * dt
        speed_cmd = _clamp(speed_cmd, -speed_limit, speed_limit)

        self._last_steer_rad = steer_rad
        self._last_accel = accel

        return {
            "steering_angle": steer_rad,
            "speed": speed_cmd,
            "acceleration": accel,
        }

    def map_to_twist(
        self,
        steering_raw: float,
        accel_raw: float,
        current_speed: float,
        dt: Optional[float] = None,
    ) -> Dict[str, float]:
        ack = self.map_to_ackermann(steering_raw, accel_raw, current_speed, dt=dt)
        speed_cmd = ack["speed"]
        steer_rad = ack["steering_angle"]

        if self.wheelbase_m and abs(steer_rad) > 1e-6:
            yaw_rate = (speed_cmd / float(self.wheelbase_m)) * math.tan(steer_rad)
        elif self.max_yaw_rate_rad_s is not None:
            steer_norm = _clamp(float(steering_raw), -1.0, 1.0)
            yaw_rate = steer_norm * float(self.max_yaw_rate_rad_s)
        else:
            yaw_rate = 0.0

        return {
            "linear_x": speed_cmd,
            "angular_z": yaw_rate,
        }


__all__ = ["ControlMapper"]
