"""Offline end-to-end test for SAC pipeline (no ROS2)."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from sac_driver.inference_engine import InferenceEngine
from sac_driver.lidar_converter import LidarConverter
from sac_driver.state_builder import StateBuilder


# From src/racer_env.py (LIDAR_ANGLES_DEG)
LIDAR_ANGLES_DEG = [
    -15.0,
    0.0,
    15.0,
    30.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
    100.0,
    105.0,
    110.0,
    115.0,
    120.0,
    125.0,
    130.0,
    135.0,
    150.0,
    165.0,
    180.0,
    195.0,
]


@dataclass
class MockLaserScan:
    angle_min: float
    angle_increment: float
    ranges: np.ndarray


def build_scan(
    num_points: int,
    base_distance_m: float,
    obstacle_sector_deg: Tuple[float, float] | None = None,
    obstacle_distance_m: float = 0.5,
) -> MockLaserScan:
    angle_min = -math.pi
    angle_max = math.pi
    angle_increment = (angle_max - angle_min) / num_points
    ranges = np.full(num_points, base_distance_m, dtype=np.float32)

    if obstacle_sector_deg is not None:
        lo_deg, hi_deg = obstacle_sector_deg
        for idx in range(num_points):
            angle_rad = angle_min + idx * angle_increment
            angle_deg = math.degrees(angle_rad)
            if lo_deg <= angle_deg <= hi_deg:
                ranges[idx] = obstacle_distance_m

    return MockLaserScan(angle_min=angle_min, angle_increment=angle_increment, ranges=ranges)


def build_obs(
    lidar_norm: np.ndarray,
    speed_mps: float,
    servo_norm: float,
    collision_flag: float,
    max_speed_mps: float,
) -> np.ndarray:
    speed_norm = min(abs(float(speed_mps)) / max_speed_mps, 1.0)
    servo_norm = min(max(float(servo_norm), 0.0), 1.0)
    collision = 1.0 if float(collision_flag) > 0.0 else 0.0
    return np.array(list(lidar_norm) + [collision, speed_norm, servo_norm], dtype=np.float32)


def run_scenario(
    name: str,
    scan: MockLaserScan,
    converter: LidarConverter,
    engine: InferenceEngine,
    stack_frames: int,
    max_speed_mps: float,
) -> None:
    lidar_norm = converter.convert(scan)
    obs = build_obs(lidar_norm, speed_mps=2.0, servo_norm=0.5, collision_flag=0.0, max_speed_mps=max_speed_mps)
    builder = StateBuilder(stack_frames=stack_frames, lidar_dim=len(lidar_norm), max_speed_mps=max_speed_mps)
    state = builder.reset(obs)
    # fill stack with repeated frames
    for _ in range(stack_frames - 1):
        state = builder.update(lidar_norm, speed_mps=2.0, servo_normalized=0.5, collision_flag=0.0)

    steer, accel = engine.get_action(state)
    a_scale = engine.policy.action_scale.detach().cpu().numpy()
    a_bias = engine.policy.action_bias.detach().cpu().numpy()
    accel_min = a_bias[1] - a_scale[1]
    accel_max = a_bias[1] + a_scale[1]

    print(f"[{name}] steer={steer:.3f} accel={accel:.3f} (inference {engine.last_inference_ms:.3f} ms)")
    if not (-1.05 <= steer <= 1.05):
        print(f"  WARN: steer out of expected range [-1,1]: {steer:.3f}")
    if not (accel_min - 0.1 <= accel <= accel_max + 0.1):
        print(f"  WARN: accel out of expected range [{accel_min:.2f},{accel_max:.2f}]: {accel:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu/cuda/auto")
    parser.add_argument("--stack-frames", type=int, default=4)
    parser.add_argument("--max-speed", type=float, default=8.0)
    parser.add_argument("--max-range", type=float, default=20.0)
    parser.add_argument(
        "--angle-offset-deg",
        type=float,
        default=-90.0,
        help="Offset applied to training angles to map into LaserScan frame",
    )
    parser.add_argument(
        "--angle-direction",
        type=float,
        default=1.0,
        help="Direction multiplier (1 or -1) for angle mapping",
    )
    parser.add_argument("--num-points", type=int, default=360)
    args = parser.parse_args()

    engine = InferenceEngine(args.checkpoint, device=args.device, weights_only=False)
    converter = LidarConverter(
        target_angles_deg=LIDAR_ANGLES_DEG,
        max_range_m=args.max_range,
        angle_offset_deg=args.angle_offset_deg,
        angle_direction=args.angle_direction,
        use_interpolation=True,
    )

    scenarios = [
        ("clear", build_scan(args.num_points, base_distance_m=5.0)),
        ("front_obstacle", build_scan(args.num_points, base_distance_m=5.0, obstacle_sector_deg=(-10, 10))),
        ("left_obstacle", build_scan(args.num_points, base_distance_m=5.0, obstacle_sector_deg=(20, 60))),
        ("right_obstacle", build_scan(args.num_points, base_distance_m=5.0, obstacle_sector_deg=(-60, -20))),
    ]

    for name, scan in scenarios:
        run_scenario(name, scan, converter, engine, args.stack_frames, args.max_speed)


if __name__ == "__main__":
    main()
