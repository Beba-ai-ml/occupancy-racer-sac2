"""Build stacked state vectors matching training layout."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional, Sequence

import numpy as np


class StateBuilder:
    def __init__(
        self,
        stack_frames: int,
        lidar_dim: int,
        max_speed_mps: float,
        use_imu: bool = False,
    ) -> None:
        self.stack_frames = max(1, int(stack_frames))
        self.lidar_dim = int(lidar_dim)
        self.max_speed_mps = float(max_speed_mps) if max_speed_mps > 0 else 1.0
        self.use_imu = bool(use_imu)
        # IMU channels are always present in training obs (+5: collision, speed, servo, linear_accel, angular_vel)
        self._extra_dims = 5
        self._stack: Deque[np.ndarray] = deque(maxlen=self.stack_frames)

    @property
    def single_obs_dim(self) -> int:
        return self.lidar_dim + self._extra_dims

    @property
    def state_dim(self) -> int:
        return self.single_obs_dim * self.stack_frames

    def reset(self, first_obs: Optional[np.ndarray] = None) -> np.ndarray:
        if first_obs is None:
            first_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        first_obs = np.asarray(first_obs, dtype=np.float32).reshape(-1)
        if first_obs.size != self.single_obs_dim:
            raise ValueError(f"Expected obs dim {self.single_obs_dim}, got {first_obs.size}")

        self._stack.clear()
        if self.stack_frames <= 1:
            self._stack.append(first_obs)
            return first_obs
        for _ in range(self.stack_frames):
            self._stack.append(first_obs)
        return np.concatenate(list(self._stack))

    def _build_observation(
        self,
        lidar_normalized: Sequence[float],
        speed_mps: float,
        servo_normalized: float,
        collision_flag: float,
        linear_accel: float = 0.0,
        angular_vel: float = 0.0,
    ) -> np.ndarray:
        lidar_arr = np.asarray(lidar_normalized, dtype=np.float32).reshape(-1)
        if lidar_arr.size != self.lidar_dim:
            raise ValueError(f"Expected lidar dim {self.lidar_dim}, got {lidar_arr.size}")
        lidar_arr = np.clip(lidar_arr, 0.0, 1.0)

        speed_norm = min(abs(float(speed_mps)) / self.max_speed_mps, 1.0)
        servo_norm = min(max(float(servo_normalized), 0.0), 1.0)
        collision = 1.0 if float(collision_flag) > 0.0 else 0.0

        scalars = [collision, speed_norm, servo_norm]
        if self.use_imu:
            max_accel = 4.0
            max_yaw_rate = 3.0
            accel_norm = max(-1.0, min(float(linear_accel) / max_accel, 1.0))
            yaw_norm = max(-1.0, min(float(angular_vel) / max_yaw_rate, 1.0))
            scalars.extend([accel_norm, yaw_norm])

        obs = np.array(
            list(lidar_arr) + scalars,
            dtype=np.float32,
        )
        return obs

    def update(
        self,
        lidar_normalized: Sequence[float],
        speed_mps: float,
        servo_normalized: float,
        collision_flag: float = 0.0,
        linear_accel: float = 0.0,
        angular_vel: float = 0.0,
    ) -> np.ndarray:
        obs = self._build_observation(
            lidar_normalized, speed_mps, servo_normalized, collision_flag,
            linear_accel, angular_vel,
        )
        if self.stack_frames <= 1:
            self._stack.clear()
            self._stack.append(obs)
            return obs
        if not self._stack:
            return self.reset(obs)
        self._stack.append(obs)
        return np.concatenate(list(self._stack))


__all__ = ["StateBuilder"]
