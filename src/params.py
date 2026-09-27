from __future__ import annotations

import math

from .vehicle import MapParams, VehicleParams
from .steering import SteeringProfile


def build_vehicle_params(physics_cfg: dict) -> VehicleParams:
    vehicle_cfg = physics_cfg.get("vehicle", {})
    size_cfg = vehicle_cfg.get("size_m", {})

    length = float(size_cfg.get("length", 0.45))
    width = float(size_cfg.get("width", 0.30))
    max_steer_deg = float(vehicle_cfg.get("max_steer_angle_deg", 20.0))
    wheelbase = float(vehicle_cfg.get("wheelbase", length * 0.6))
    steering_cfg = vehicle_cfg.get("steering_profile")
    steering_profile = None
    if steering_cfg is not None:
        steering_profile = SteeringProfile(
            delay_s=float(steering_cfg["delay_s"]),
            full_travel_s=float(steering_cfg["full_travel_s"]),
            slow_diameter_m=float(steering_cfg["slow_diameter_m"]),
            fast_diameter_m=float(steering_cfg["fast_diameter_m"]),
            reference_speed_mps=float(steering_cfg["reference_speed_mps"]),
            timing_scale_range=tuple(float(v) for v in steering_cfg.get("timing_scale_range", [1.0, 1.0])),
            mid_speed_mps=(float(steering_cfg["mid_speed_mps"])
                           if "mid_speed_mps" in steering_cfg else None),
            mid_diameter_m=(float(steering_cfg["mid_diameter_m"])
                            if "mid_diameter_m" in steering_cfg else None),
            fast_diameter_range_m=(tuple(float(v) for v in steering_cfg["fast_diameter_range_m"])
                                   if "fast_diameter_range_m" in steering_cfg else None),
        )

    return VehicleParams(
        acceleration=float(vehicle_cfg.get("acceleration", 2.0)),
        brake_deceleration=float(vehicle_cfg.get("brake_deceleration", 3.5)),
        reverse_acceleration=float(vehicle_cfg.get("reverse_acceleration", 1.5)),
        max_speed=float(vehicle_cfg.get("max_speed", 8.0)),
        max_reverse_speed=float(vehicle_cfg.get("max_reverse_speed", 4.0)),
        friction=float(vehicle_cfg.get("friction", 0.6)),
        drag=float(vehicle_cfg.get("drag", 0.2)),
        max_steer_angle=math.radians(max_steer_deg),
        wheelbase=wheelbase,
        length=length,
        width=width,
        steer_speed_ref=float(vehicle_cfg.get("steer_speed_ref", 0.0)),
        steering_profile=steering_profile,
        speed_limit_choices=tuple(float(v) for v in vehicle_cfg.get("speed_limit_choices", [])),
        speed_observation_scale_mps=float(vehicle_cfg.get("speed_observation_scale_mps", 0.0)),
    )


def build_map_params(physics_cfg: dict) -> MapParams:
    map_cfg = physics_cfg.get("map", {})
    return MapParams(
        surface_friction=float(map_cfg.get("surface_friction", 1.0)),
        surface_drag=float(map_cfg.get("surface_drag", 0.0)),
    )
