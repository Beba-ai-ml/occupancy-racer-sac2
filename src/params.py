from __future__ import annotations

import math

from .vehicle import MapParams, VehicleParams


def build_vehicle_params(physics_cfg: dict) -> VehicleParams:
    vehicle_cfg = physics_cfg.get("vehicle", {})
    size_cfg = vehicle_cfg.get("size_m", {})

    length = float(size_cfg.get("length", 0.45))
    width = float(size_cfg.get("width", 0.30))
    max_steer_deg = float(vehicle_cfg.get("max_steer_angle_deg", 20.0))
    wheelbase = float(vehicle_cfg.get("wheelbase", length * 0.6))

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
    )


def build_map_params(physics_cfg: dict) -> MapParams:
    map_cfg = physics_cfg.get("map", {})
    return MapParams(
        surface_friction=float(map_cfg.get("surface_friction", 1.0)),
        surface_drag=float(map_cfg.get("surface_drag", 0.0)),
    )
