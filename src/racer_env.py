from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import os
from typing import Deque, List, Tuple

import numpy as np
import pygame

from .map_loader import MapData
from .vehicle import MapParams, Vehicle, VehicleParams

LIDAR_CENTER_DEG = 90.0
LIDAR_TOTAL_ARC_DEG = 210.0
LIDAR_DENSE_HALF_DEG = 45.0
LIDAR_DENSE_STEP_DEG = 5.0
LIDAR_SPARSE_STEP_DEG = 15.0
LIDAR_FRONT_CONE_DEG = 20.0


def _build_legacy_lidar_angles() -> list[float]:
    """Classic 27-ray layout: dense front ±45°, sparse sides, 210° arc."""
    center = int(LIDAR_CENTER_DEG)
    arc_half = int(LIDAR_TOTAL_ARC_DEG / 2)
    dense_half = int(LIDAR_DENSE_HALF_DEG)
    dense_step = int(LIDAR_DENSE_STEP_DEG)
    sparse_step = int(LIDAR_SPARSE_STEP_DEG)

    min_angle = center - arc_half
    max_angle = center + arc_half
    dense_start = center - dense_half
    dense_end = center + dense_half

    angles = list(range(min_angle, dense_start, sparse_step))
    angles += list(range(dense_start, dense_end + 1, dense_step))
    angles += list(range(dense_end + sparse_step, max_angle + 1, sparse_step))
    return [float(a) for a in angles]


def build_lidar_angles(front_step_deg: float, rear_step_deg: float) -> list[float]:
    """Build full 360° LiDAR with configurable front/rear density.

    Front hemisphere (0°–180°, centered on 90°=forward): ``front_step_deg`` spacing.
    Rear hemisphere (180°–360°): ``rear_step_deg`` spacing.
    """
    center = LIDAR_CENTER_DEG  # 90° = forward

    # Front hemisphere: 0° to 180° (inclusive on both ends)
    front_start = center - 90.0  # 0°
    front_end = center + 90.0    # 180°
    n_front = int(round((front_end - front_start) / front_step_deg)) + 1
    front_angles = [front_start + i * front_step_deg for i in range(n_front)]

    # Rear hemisphere: 180°+step to 360°-step (exclude boundaries already covered / wrap point)
    rear_start = front_end + rear_step_deg  # first rear angle after 180°
    rear_end = front_start + 360.0 - rear_step_deg  # last rear angle before 360°(=0°)
    rear_angles = []
    a = rear_start
    while a <= rear_end + 1e-9:
        rear_angles.append(a)
        a += rear_step_deg

    return front_angles + rear_angles


LIDAR_ANGLES_DEG = _build_legacy_lidar_angles()
LIDAR_MAX_RANGE_M = 20.0


def create_map_surface(
    image: np.ndarray,
    kill_mask: np.ndarray | None = None,
    spawn_mask: np.ndarray | None = None,
    lookat_mask: np.ndarray | None = None,
    raceline_mask: np.ndarray | None = None,
) -> pygame.Surface:
    if image.ndim != 2:
        raise ValueError("Map image must be a 2D array")
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    if kill_mask is not None:
        rgb[kill_mask, 0] = np.minimum(rgb[kill_mask, 0].astype(np.int16) + 160, 255).astype(np.uint8)
        rgb[kill_mask, 1] = (rgb[kill_mask, 1] * 0.3).astype(np.uint8)
        rgb[kill_mask, 2] = (rgb[kill_mask, 2] * 0.3).astype(np.uint8)
    if spawn_mask is not None:
        rgb[spawn_mask, 0] = (rgb[spawn_mask, 0] * 0.5).astype(np.uint8)
        rgb[spawn_mask, 1] = np.minimum(rgb[spawn_mask, 1].astype(np.int16) + 120, 255).astype(np.uint8)
        rgb[spawn_mask, 2] = (rgb[spawn_mask, 2] * 0.5).astype(np.uint8)
    if lookat_mask is not None:
        rgb[lookat_mask, 0] = np.minimum(rgb[lookat_mask, 0].astype(np.int16) + 140, 255).astype(np.uint8)
        rgb[lookat_mask, 1] = np.minimum(rgb[lookat_mask, 1].astype(np.int16) + 140, 255).astype(np.uint8)
        rgb[lookat_mask, 2] = (rgb[lookat_mask, 2] * 0.3).astype(np.uint8)
    if raceline_mask is not None:
        rgb[raceline_mask, 0] = np.minimum(rgb[raceline_mask, 0].astype(np.int16) + 120, 255).astype(np.uint8)
        rgb[raceline_mask, 1] = (rgb[raceline_mask, 1] * 0.3).astype(np.uint8)
        rgb[raceline_mask, 2] = np.minimum(rgb[raceline_mask, 2].astype(np.int16) + 160, 255).astype(np.uint8)
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return surface


@dataclass
class _Obstacle:
    position: pygame.Vector2
    half_size: pygame.Vector2
    dynamic: bool
    ttl: float | None
    persistent: bool
    encountered: bool = False

    @property
    def radius(self) -> float:
        return float(math.hypot(self.half_size.x, self.half_size.y))


class RacerEnv:
    def __init__(
        self,
        map_data: MapData,
        vehicle_params: VehicleParams,
        map_params: MapParams,
        steer_bins: List[int],
        accel_bins: List[float],
        fps: int = 60,
        render: bool = False,
        stack_frames: int = 1,
        episode_offset: int = 0,
        sim_cfg: dict | None = None,
    ) -> None:
        self.map_data = map_data
        self.vehicle_params = vehicle_params
        self.map_params = map_params
        self._base_vehicle_params = vehicle_params
        self._base_map_params = map_params
        self.sim_cfg = sim_cfg or {}
        self.occupied_mask = map_data.occupied_mask
        if map_data.kill_mask is not None:
            self.collision_mask = map_data.occupied_mask | map_data.kill_mask
        else:
            self.collision_mask = self.occupied_mask
        self._kill_boundary_hit = False
        self.map_height, self.map_width = map_data.image.shape
        self.map_center = pygame.Vector2(
            (self.map_width * map_data.resolution) / 2.0,
            (self.map_height * map_data.resolution) / 2.0,
        )
        lidar_cfg = self.sim_cfg.get("lidar", {})
        if "front_step_deg" in lidar_cfg:
            self.lidar_angles_deg = build_lidar_angles(
                float(lidar_cfg["front_step_deg"]),
                float(lidar_cfg.get("rear_step_deg", 2.0)),
            )
        else:
            self.lidar_angles_deg = list(LIDAR_ANGLES_DEG)  # legacy 27-ray
        self.lidar_offsets = [math.radians(90.0 - angle) for angle in self.lidar_angles_deg]
        self.lidar_max_range_m = float(LIDAR_MAX_RANGE_M)

        self.spawn_angle = 0.0
        self.spawn_clearance_m = max(self.vehicle_params.length, self.vehicle_params.width)
        self.spawn_wall_clearance_m = 1.5
        if self.spawn_clearance_m < self.spawn_wall_clearance_m:
            self.spawn_clearance_m = self.spawn_wall_clearance_m
        self.spawn_clearance_angles = [
            float(i) * (2.0 * math.pi / 16.0) for i in range(16)
        ]
        self._collision_pad = 0.5
        self.stuck_time = 0.0
        self.stuck_speed_thresh = 0.1
        self.stuck_grace_s = 0.5
        self.stuck_tau = 1.5
        self.stuck_scale = 3.0
        self.stuck_penalty_cap = 0.5
        self.reward_scale = 0.8
        self.reward_clip = 20.0
        self.reward_collision_penalty = -20.0
        self.reward_front_penalty = 0.12
        self.reward_side_penalty = 0.04
        self.reward_min_clear_penalty = 0.08
        self.reward_balance_penalty = 0.03
        self.reward_reverse_penalty = 0.05
        self.reward_alignment_bonus = 0.02
        self.reward_distance_progress_weight = 0.1  # S9: forward progress reward
        self.reward_forward_speed_weight = 0.6
        self.reward_front_speed_weight = 0.2
        self.reward_front_cone_deg = float(LIDAR_FRONT_CONE_DEG)
        self.episode_time_limit_s = 0.0
        self.episode_time_s = 0.0
        self.backward_distance_m = 0.0
        self.backward_limit_m = 4.0
        self.backward_penalized = False
        self.backward_death = False
        self.stack_frames = max(1, int(stack_frames))
        self._obs_stack: Deque[np.ndarray] = deque(maxlen=self.stack_frames)

        self.actions = [(steer, accel) for steer in steer_bins for accel in accel_bins]
        accel_values = [float(v) for v in accel_bins] if accel_bins else [-2.0, 2.0]
        self.accel_min = float(min(accel_values))
        self.accel_max = float(max(accel_values))
        if self.accel_min > self.accel_max:
            self.accel_min, self.accel_max = self.accel_max, self.accel_min
        self.fps = fps if fps > 0 else 60
        self.fixed_dt = 1.0 / self.fps
        self.last_dt = self.fixed_dt

        self.ppm = 1.0 / map_data.resolution

        # Free positions: exclude kill boundaries from free space
        base_free = map_data.free_mask
        if map_data.kill_mask is not None:
            base_free = base_free & ~map_data.kill_mask
        self.free_positions = np.argwhere(base_free)
        if self.free_positions.size == 0:
            raise ValueError("No free space found in map to place vehicle")

        # Spawn positions: restrict to spawn zones if defined
        if map_data.spawn_mask is not None:
            self.spawn_positions = np.argwhere(base_free & map_data.spawn_mask)
            if self.spawn_positions.size == 0:
                print("WARNING: spawn mask has zero overlap with free space, falling back")
                self.spawn_positions = self.free_positions
        else:
            self.spawn_positions = self.free_positions

        # Look-at targets: yellow pixels the car faces toward on spawn
        if map_data.lookat_mask is not None:
            self.lookat_positions = np.argwhere(map_data.lookat_mask)
            if self.lookat_positions.size == 0:
                self.lookat_positions = None
        else:
            self.lookat_positions = None

        # Per-zone spawn and lookat positions (zone_id -> np.ndarray of [y,x])
        self._spawn_by_zone: dict[int, np.ndarray] = {}
        self._lookat_by_zone: dict[int, np.ndarray] = {}
        self._zone_ids: list[int] = []
        if map_data.spawn_zones is not None:
            for zid in (1, 2, 3):
                zone_mask = map_data.spawn_zones == zid
                positions = np.argwhere(base_free & zone_mask)
                if positions.size > 0:
                    self._spawn_by_zone[zid] = positions
            if self._spawn_by_zone:
                self._zone_ids = sorted(self._spawn_by_zone.keys())
        if map_data.lookat_zones is not None:
            for zid in (1, 2, 3):
                zone_mask = map_data.lookat_zones == zid
                positions = np.argwhere(zone_mask)
                if positions.size > 0:
                    self._lookat_by_zone[zid] = positions
        # Warn if any spawn zone has no matching lookat zone
        if self._zone_ids and self._lookat_by_zone:
            for zid in self._zone_ids:
                if zid not in self._lookat_by_zone:
                    print(f"WARNING: spawn zone {zid} has no matching lookat zone, will use clockwise heading")
        # Backward compat: if spawn_mask exists but no zone_ids, put everything into zone 1
        if not self._zone_ids and map_data.spawn_mask is not None:
            self._spawn_by_zone[1] = self.spawn_positions.copy()
            self._zone_ids = [1]
            if self.lookat_positions is not None:
                self._lookat_by_zone[1] = self.lookat_positions

        # Pre-filter spawn positions by wall clearance so _random_spawn() can
        # skip all raycasting.  Only done when a spawn_mask restricts the set
        # and the candidate count is manageable.
        self._spawn_prefiltered = False
        if map_data.spawn_mask is not None and len(self.spawn_positions) <= 50000:
            res = map_data.resolution
            clearance = self.spawn_clearance_m
            keep = []
            for i in range(len(self.spawn_positions)):
                y, x = self.spawn_positions[i]
                origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
                if self._min_wall_distance(origin) >= clearance:
                    keep.append(i)
            if keep:
                self.spawn_positions = self.spawn_positions[np.array(keep)]
                self._spawn_prefiltered = True
            # If all filtered out: keep unfiltered, _spawn_prefiltered stays False

        # Pre-filter per-zone spawn positions by wall clearance
        self._spawn_by_zone_prefiltered: dict[int, np.ndarray] = {}
        if self._zone_ids:
            res = map_data.resolution
            clearance = self.spawn_clearance_m
            for zid, positions in self._spawn_by_zone.items():
                if len(positions) <= 50000:
                    keep = []
                    for i in range(len(positions)):
                        y, x = positions[i]
                        origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
                        if self._min_wall_distance(origin) >= clearance:
                            keep.append(i)
                    if keep:
                        self._spawn_by_zone_prefiltered[zid] = positions[np.array(keep)]

        track_cfg = self.sim_cfg.get("track", {})
        self.track_center_mode = str(track_cfg.get("center", "map")).lower()
        direction = str(track_cfg.get("direction", "clockwise")).lower()
        self.track_direction = -1.0 if direction in ("counterclockwise", "ccw", "anticlockwise") else 1.0
        self.spawn_face_forward = bool(track_cfg.get("spawn_face_forward", True))
        if self.track_center_mode == "free":
            mean_yx = self.free_positions.mean(axis=0)
            self.map_center = pygame.Vector2(
                (float(mean_yx[1]) + 0.5) * map_data.resolution,
                (float(mean_yx[0]) + 0.5) * map_data.resolution,
            )

        # Raceline waypoints for heading guidance on open maps
        self._raceline_waypoints: np.ndarray | None = None
        self._raceline_is_loop = False
        if map_data.raceline_mask is not None:
            self._build_raceline_waypoints(map_data.raceline_mask, map_data.resolution)

        self.render_requested = bool(render)
        self.render_enabled = bool(render)
        self.screen = None
        self.clock = None
        self.map_surface_base = None
        self.map_surface_scaled = None
        self.zoom = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 3.0
        self.zoom_step = 0.1
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        if self.render_requested:
            self._init_render()

        spawn_position, spawn_angle = self._random_spawn(self.spawn_clearance_m)
        self.spawn_angle = spawn_angle
        self.vehicle = Vehicle(
            vehicle_params,
            spawn_position,
            self.ppm,
            angle=spawn_angle,
            render_enabled=self.render_requested,
        )
        self.servo_value = 10.0
        self.prev_position = self.vehicle.position.copy()

        self.sim_enabled = bool(self.sim_cfg.get("enabled", False))

        reward_cfg = self.sim_cfg.get("reward", {})
        if "scale" in reward_cfg:
            self.reward_scale = float(reward_cfg["scale"])
        if "clip" in reward_cfg:
            self.reward_clip = None if reward_cfg["clip"] is None else float(reward_cfg["clip"])
        if "collision_penalty" in reward_cfg:
            self.reward_collision_penalty = float(reward_cfg["collision_penalty"])
        if "front_penalty" in reward_cfg:
            self.reward_front_penalty = float(reward_cfg["front_penalty"])
        if "side_penalty" in reward_cfg:
            self.reward_side_penalty = float(reward_cfg["side_penalty"])
        if "min_clear_penalty" in reward_cfg:
            self.reward_min_clear_penalty = float(reward_cfg["min_clear_penalty"])
        if "balance_penalty" in reward_cfg:
            self.reward_balance_penalty = float(reward_cfg["balance_penalty"])
        if "reverse_penalty" in reward_cfg:
            self.reward_reverse_penalty = float(reward_cfg["reverse_penalty"])
        if "alignment_bonus" in reward_cfg:
            self.reward_alignment_bonus = float(reward_cfg["alignment_bonus"])
        if "distance_progress_weight" in reward_cfg:
            self.reward_distance_progress_weight = float(reward_cfg["distance_progress_weight"])
        if "forward_speed_weight" in reward_cfg:
            self.reward_forward_speed_weight = float(reward_cfg["forward_speed_weight"])
        if "front_speed_weight" in reward_cfg:
            self.reward_front_speed_weight = float(reward_cfg["front_speed_weight"])
        if "front_cone_deg" in reward_cfg:
            self.reward_front_cone_deg = float(reward_cfg["front_cone_deg"])

        episode_cfg = self.sim_cfg.get("episode", {})
        if "time_limit_s" in episode_cfg:
            self.episode_time_limit_s = float(episode_cfg["time_limit_s"])

        physics_cfg = self.sim_cfg.get("physics", {})
        self.physics_enabled = self.sim_enabled and bool(physics_cfg.get("enabled", False))
        self.phys_accel_scale = self._parse_range(physics_cfg.get("accel_scale"), (1.0, 1.0))
        self.phys_brake_scale = self._parse_range(physics_cfg.get("brake_scale"), (1.0, 1.0))
        self.phys_reverse_scale = self._parse_range(physics_cfg.get("reverse_scale"), (1.0, 1.0))
        self.phys_max_speed_scale = self._parse_range(physics_cfg.get("max_speed_scale"), (1.0, 1.0))
        self.phys_max_reverse_speed_scale = self._parse_range(physics_cfg.get("max_reverse_speed_scale"), (1.0, 1.0))
        self.phys_friction_scale = self._parse_range(physics_cfg.get("friction_scale"), (1.0, 1.0))
        self.phys_drag_scale = self._parse_range(physics_cfg.get("drag_scale"), (1.0, 1.0))
        self.phys_wheelbase_scale = self._parse_range(physics_cfg.get("wheelbase_scale"), (1.0, 1.0))
        self.phys_max_steer_scale = self._parse_range(physics_cfg.get("max_steer_scale"), (1.0, 1.0))

        surface_cfg = self.sim_cfg.get("surface", {})
        self.surface_enabled = self.sim_enabled and bool(surface_cfg.get("enabled", False))
        self.surface_friction_scale = self._parse_range(surface_cfg.get("friction_scale"), (1.0, 1.0))
        self.surface_drag_range = self._parse_range(surface_cfg.get("drag_range"), (self.map_params.surface_drag, self.map_params.surface_drag))

        obs_noise_cfg = self.sim_cfg.get("observation_noise", {})
        self.obs_noise_enabled = self.sim_enabled and bool(obs_noise_cfg.get("enabled", False))
        self.lidar_noise_std = float(obs_noise_cfg.get("lidar_noise_std", 0.0))
        self.lidar_dropout_prob = float(obs_noise_cfg.get("lidar_dropout_prob", 0.0))
        self.lidar_spike_prob = float(obs_noise_cfg.get("lidar_spike_prob", 0.0))
        self.speed_noise_std = float(obs_noise_cfg.get("speed_noise_std", 0.0))
        self.servo_noise_std = float(obs_noise_cfg.get("servo_noise_std", 0.0))

        # S6: LiDAR beam divergence + ego-motion blur
        lidar_sim_cfg = self.sim_cfg.get("lidar_sim", {})
        self.lidar_beam_divergence_enabled = self.sim_enabled and bool(lidar_sim_cfg.get("beam_divergence", False))
        self.lidar_beam_noise_std = float(lidar_sim_cfg.get("beam_noise_std", 0.03))
        self.lidar_ego_motion_blur_enabled = self.sim_enabled and bool(lidar_sim_cfg.get("ego_motion_blur", False))
        self.lidar_scan_time_s = float(lidar_sim_cfg.get("scan_time_s", 0.125))  # 8 Hz

        # S7: IMU observation + encoder noise
        imu_cfg = self.sim_cfg.get("imu", {})
        self.imu_enabled = self.sim_enabled and bool(imu_cfg.get("enabled", False))
        self.imu_max_accel = float(imu_cfg.get("max_accel", 4.0))  # m/s²
        self.imu_max_yaw_rate = float(imu_cfg.get("max_yaw_rate", 3.0))  # rad/s
        self.encoder_noise_enabled = self.sim_enabled and bool(imu_cfg.get("encoder_noise", False))
        self.encoder_noise_std = float(imu_cfg.get("encoder_noise_std", 0.03))  # 3% VESC noise
        # State for IMU delta computation
        self._prev_speed = 0.0
        self._prev_angle = 0.0

        # S8: Soft collision — allow light contacts before termination
        collision_cfg = self.sim_cfg.get("soft_collision", {})
        self.soft_collision_enabled = self.sim_enabled and bool(collision_cfg.get("enabled", False))
        self.max_light_contacts = int(collision_cfg.get("max_light_contacts", 2))
        self.light_contact_penalty = float(collision_cfg.get("light_contact_penalty", -5.0))
        self._contact_count = 0

        # S9: Remove alignment reward, add distance progress
        progress_cfg = self.sim_cfg.get("distance_progress", {})
        self.distance_progress_enabled = self.sim_enabled and bool(progress_cfg.get("enabled", False))

        # S10: Async sensor delays
        sensor_delay_cfg = self.sim_cfg.get("sensor_delay", {})
        self.sensor_delay_enabled = self.sim_enabled and bool(sensor_delay_cfg.get("enabled", False))
        self.lidar_delay_range = self._parse_int_range(sensor_delay_cfg.get("lidar_delay_frames"), (3, 4))
        self.speed_delay_range = self._parse_int_range(sensor_delay_cfg.get("speed_delay_frames"), (1, 3))
        self.imu_delay_range = self._parse_int_range(sensor_delay_cfg.get("imu_delay_frames"), (0, 1))
        max_lidar_buf = max(self.lidar_delay_range[1] + 1, 10)
        self._lidar_obs_history: Deque[np.ndarray] = deque(maxlen=max_lidar_buf)
        self._speed_obs_history: Deque[float] = deque(maxlen=max(self.speed_delay_range[1] + 1, 5))
        self._imu_obs_history: Deque[Tuple[float, float]] = deque(maxlen=max(self.imu_delay_range[1] + 1, 3))
        # Per-episode delay samples (set in reset, initialized here to avoid AttributeError)
        self._ep_lidar_delay = 0
        self._ep_speed_delay = 0
        self._ep_imu_delay = 0

        # S12: Wind and slope perturbations
        wind_slope_cfg = self.sim_cfg.get("wind_slope", {})
        self.wind_enabled = self.sim_enabled and bool(wind_slope_cfg.get("wind_enabled", False))
        self.wind_force_std = float(wind_slope_cfg.get("wind_force_std", 0.3))
        self.wind_freq_hz = float(wind_slope_cfg.get("wind_freq_hz", 0.5))
        self.slope_enabled = self.sim_enabled and bool(wind_slope_cfg.get("slope_enabled", False))
        self.slope_range = self._parse_range(wind_slope_cfg.get("slope_range"), (-0.035, 0.035))
        self._slope_angle = 0.0
        self._wind_phase = 0.0
        self._wind_ou_state = 0.0  # OU process state for temporally correlated wind
        self._wind_ou_theta = 0.15  # mean-reversion rate (same as thermal drift)

        # S13: Continuous domain randomization (battery sag, friction cycle)
        continuous_dr_cfg = self.sim_cfg.get("continuous_dr", {})
        self.continuous_dr_enabled = self.sim_enabled and bool(continuous_dr_cfg.get("enabled", False))
        self.battery_sag_pct = float(continuous_dr_cfg.get("battery_sag_pct", 0.05))  # 5% drop
        self.friction_cycle_amplitude = float(continuous_dr_cfg.get("friction_cycle_amplitude", 0.02))
        self.friction_cycle_period_s = float(continuous_dr_cfg.get("friction_cycle_period_s", 30.0))

        # S14: Sensor thermal drift (Ornstein-Uhlenbeck process)
        thermal_cfg = self.sim_cfg.get("thermal_drift", {})
        self.thermal_drift_enabled = self.sim_enabled and bool(thermal_cfg.get("enabled", False))
        self._ou_theta = float(thermal_cfg.get("ou_theta", 0.15))  # mean-reversion speed
        self._ou_sigma_lidar = float(thermal_cfg.get("ou_sigma_lidar", 0.005))  # LiDAR volatility
        self._ou_sigma_speed = float(thermal_cfg.get("ou_sigma_speed", 0.005))  # speed volatility
        self._lidar_bias = 0.0
        self._speed_bias = 0.0

        control_cfg = self.sim_cfg.get("control", {})
        self.control_enabled = self.sim_enabled and bool(control_cfg.get("enabled", False))
        self.delay_steps_range = self._parse_int_range(control_cfg.get("delay_steps"), (0, 0))
        self.steer_rate_limit = float(control_cfg.get("steer_rate_limit", 0.0))
        self.accel_rate_limit = float(control_cfg.get("accel_rate_limit", 0.0))

        dt_cfg = self.sim_cfg.get("dt_jitter", {})
        self.dt_jitter_enabled = self.sim_enabled and bool(dt_cfg.get("enabled", False))
        self.dt_scale_range = self._parse_range(dt_cfg.get("dt_scale_range"), (1.0, 1.0))

        action_noise_cfg = self.sim_cfg.get("action_noise", {})
        self.action_noise_enabled = self.sim_enabled and bool(action_noise_cfg.get("enabled", False))
        self.action_steer_scale_range = self._parse_range(action_noise_cfg.get("steer_scale_range"), (1.0, 1.0))
        self.action_steer_bias_range = self._parse_range(action_noise_cfg.get("steer_bias_range"), (0.0, 0.0))
        self.action_accel_scale_range = self._parse_range(action_noise_cfg.get("accel_scale_range"), (1.0, 1.0))
        self.action_accel_bias_range = self._parse_range(action_noise_cfg.get("accel_bias_range"), (0.0, 0.0))

        perturb_cfg = self.sim_cfg.get("perturb", {})
        self.perturb_enabled = self.sim_enabled and bool(perturb_cfg.get("enabled", False))
        self.perturb_prob = float(perturb_cfg.get("prob", 0.0))
        self.perturb_yaw_rate_sigma = math.radians(float(perturb_cfg.get("yaw_rate_sigma_deg", 0.0)))
        self.perturb_speed_sigma = float(perturb_cfg.get("speed_sigma", 0.0))

        obs_delay_cfg = self.sim_cfg.get("observation_delay", {})
        self.obs_delay_enabled = self.sim_enabled and bool(obs_delay_cfg.get("enabled", False))
        self.obs_delay_p1 = float(obs_delay_cfg.get("p1", 0.0))
        self.obs_delay_p2 = float(obs_delay_cfg.get("p2", 0.0))

        action_repeat_cfg = self.sim_cfg.get("action_repeat", {})
        repeat_steps = int(action_repeat_cfg.get("steps", 1))
        self.action_repeat_steps = max(1, repeat_steps) if self.sim_enabled else 1

        obstacles_cfg = self.sim_cfg.get("obstacles", {})
        self.obstacles_enabled = self.sim_enabled and bool(obstacles_cfg.get("enabled", False))
        self.obstacle_episode_prob = float(obstacles_cfg.get("episode_prob", 0.0))
        self.obstacle_start_episode = int(obstacles_cfg.get("start_episode", 0))
        self.obstacle_max_static = int(obstacles_cfg.get("max_static", 0))
        self.obstacle_max_total = int(obstacles_cfg.get("max_total", max(self.obstacle_max_static, 0)))
        self.obstacle_min_gap = float(obstacles_cfg.get("min_gap", 0.0))
        self.obstacle_min_wall_clearance = float(obstacles_cfg.get("min_wall_clearance", 0.0))
        self.obstacle_min_separation = float(obstacles_cfg.get("min_separation", 0.0))
        self.obstacle_pass_buffer = float(obstacles_cfg.get("pass_buffer", 0.0))
        self.obstacle_pass_activate_dist = float(obstacles_cfg.get("pass_activate_distance", 8.0))
        self.obstacle_allow_wall_overlap = bool(obstacles_cfg.get("allow_wall_overlap", False))
        self.static_size_range = self._parse_range(obstacles_cfg.get("static_size_range"), (0.0, 0.0))
        self.static_radius_range = self._parse_range(obstacles_cfg.get("static_radius_range"), (0.0, 0.0))
        if self.static_size_range[1] <= 0.0 and self.static_radius_range[1] > 0.0:
            self.static_size_range = (
                self.static_radius_range[0] * 2.0,
                self.static_radius_range[1] * 2.0,
            )
        self.static_persist_prob = float(obstacles_cfg.get("static_persist_prob", 0.0))
        self.static_min_distance = float(obstacles_cfg.get("static_min_distance", 0.0))
        self.static_spawn_attempts = int(obstacles_cfg.get("static_spawn_attempts", 50))

        dynamic_cfg = obstacles_cfg.get("dynamic", {})
        self.dynamic_enabled = self.obstacles_enabled and bool(dynamic_cfg.get("enabled", False))
        self.dynamic_spawn_rate = float(dynamic_cfg.get("spawn_rate", 0.0))
        self.dynamic_distance_range = self._parse_range(dynamic_cfg.get("distance_range"), (0.0, 0.0))
        self.dynamic_size_range = self._parse_range(dynamic_cfg.get("size_range"), (0.0, 0.0))
        self.dynamic_radius_range = self._parse_range(dynamic_cfg.get("radius_range"), (0.0, 0.0))
        if self.dynamic_size_range[1] <= 0.0 and self.dynamic_radius_range[1] > 0.0:
            self.dynamic_size_range = (
                self.dynamic_radius_range[0] * 2.0,
                self.dynamic_radius_range[1] * 2.0,
            )
        self.dynamic_lateral_range = self._parse_range(dynamic_cfg.get("lateral_range"), (0.0, 0.0))
        self.dynamic_ttl_range = self._parse_range(dynamic_cfg.get("ttl_range"), (0.0, 0.0))
        self.dynamic_max_active = int(dynamic_cfg.get("max_active", 1))
        self.dynamic_spawn_attempts = int(dynamic_cfg.get("spawn_attempts", 20))

        self._obs_delay_history: Deque[np.ndarray] = deque(maxlen=3)
        self._action_delay_queue: Deque[np.ndarray] = deque()
        self._repeat_action: np.ndarray | None = None
        self._repeat_count = 0
        self._neutral_action = np.array([0.0, 0.0], dtype=np.float32)
        self._last_action = np.array([0.0, 0.0], dtype=np.float32)
        self._action_scale = np.array([1.0, 1.0], dtype=np.float32)
        self._action_bias = np.array([0.0, 0.0], dtype=np.float32)
        self.action_delay_steps = 0
        self._obstacles: list[_Obstacle] = []
        self._episode_obstacles_active = False
        self._obstacle_collision = False
        self._episode_count = max(0, int(episode_offset))

    @staticmethod
    def _parse_range(value: object, default: tuple[float, float]) -> tuple[float, float]:
        if value is None:
            return float(default[0]), float(default[1])
        if isinstance(value, (int, float)):
            v = float(value)
            return v, v
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            lo, hi = float(value[0]), float(value[1])
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
        return float(default[0]), float(default[1])

    @staticmethod
    def _parse_int_range(value: object, default: tuple[int, int]) -> tuple[int, int]:
        if value is None:
            return int(default[0]), int(default[1])
        if isinstance(value, int):
            return int(value), int(value)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            lo, hi = int(value[0]), int(value[1])
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
        return int(default[0]), int(default[1])

    @staticmethod
    def _sample_range(rng: tuple[float, float]) -> float:
        return float(np.random.uniform(rng[0], rng[1]))

    @staticmethod
    def _sample_int_range(rng: tuple[int, int]) -> int:
        lo, hi = int(rng[0]), int(rng[1])
        if lo >= hi:
            return lo
        return int(np.random.randint(lo, hi + 1))

    @staticmethod
    def _obstacle_radius(half_size: pygame.Vector2) -> float:
        return float(math.hypot(half_size.x, half_size.y))

    def _sample_obstacle_half_size(self, size_range: tuple[float, float]) -> pygame.Vector2 | None:
        if size_range[1] <= 0.0:
            return None
        width = max(0.05, self._sample_range(size_range))
        height = max(0.05, self._sample_range(size_range))
        return pygame.Vector2(width * 0.5, height * 0.5)

    def _apply_domain_randomization(self) -> None:
        if not self.sim_enabled:
            self.vehicle_params = self._base_vehicle_params
            self.map_params = self._base_map_params
            self.vehicle.params = self.vehicle_params
            self.action_repeat_steps = 1
            self._action_scale = np.array([1.0, 1.0], dtype=np.float32)
            self._action_bias = np.array([0.0, 0.0], dtype=np.float32)
            self.action_delay_steps = 0
            self._action_delay_queue = deque()
            self._repeat_action = None
            self._repeat_count = 0
            self._last_action = self._neutral_action.copy()
            self._obs_delay_history.clear()
            self._obstacles.clear()
            self._episode_obstacles_active = False
            return

        if self.physics_enabled:
            base = self._base_vehicle_params
            accel = max(0.0, base.acceleration * self._sample_range(self.phys_accel_scale))
            brake = max(0.0, base.brake_deceleration * self._sample_range(self.phys_brake_scale))
            reverse = max(0.0, base.reverse_acceleration * self._sample_range(self.phys_reverse_scale))
            max_speed = max(0.1, base.max_speed * self._sample_range(self.phys_max_speed_scale))
            max_reverse = max(0.0, base.max_reverse_speed * self._sample_range(self.phys_max_reverse_speed_scale))
            friction = max(0.0, base.friction * self._sample_range(self.phys_friction_scale))
            drag = max(0.0, base.drag * self._sample_range(self.phys_drag_scale))
            wheelbase = max(0.01, base.wheelbase * self._sample_range(self.phys_wheelbase_scale))
            max_steer = max(0.0, base.max_steer_angle * self._sample_range(self.phys_max_steer_scale))
            self.vehicle_params = replace(
                base,
                acceleration=accel,
                brake_deceleration=brake,
                reverse_acceleration=reverse,
                max_speed=max_speed,
                max_reverse_speed=max_reverse,
                friction=friction,
                drag=drag,
                wheelbase=wheelbase,
                max_steer_angle=max_steer,
            )
        else:
            self.vehicle_params = self._base_vehicle_params
        self.vehicle.params = self.vehicle_params

        if self.surface_enabled:
            base_map = self._base_map_params
            friction = max(0.0, base_map.surface_friction * self._sample_range(self.surface_friction_scale))
            drag = max(0.0, self._sample_range(self.surface_drag_range))
            self.map_params = replace(base_map, surface_friction=friction, surface_drag=drag)
        else:
            self.map_params = self._base_map_params

        if self.action_noise_enabled:
            self._action_scale = np.array(
                [
                    self._sample_range(self.action_steer_scale_range),
                    self._sample_range(self.action_accel_scale_range),
                ],
                dtype=np.float32,
            )
            self._action_bias = np.array(
                [
                    self._sample_range(self.action_steer_bias_range),
                    self._sample_range(self.action_accel_bias_range),
                ],
                dtype=np.float32,
            )
        else:
            self._action_scale = np.array([1.0, 1.0], dtype=np.float32)
            self._action_bias = np.array([0.0, 0.0], dtype=np.float32)

        if self.control_enabled:
            self.action_delay_steps = self._sample_int_range(self.delay_steps_range)
        else:
            self.action_delay_steps = 0
            self.steer_rate_limit = 0.0
            self.accel_rate_limit = 0.0
        self._action_delay_queue = deque(maxlen=self.action_delay_steps + 1)
        self._repeat_action = None
        self._repeat_count = 0
        self._last_action = self._neutral_action.copy()
        self._obs_delay_history.clear()

    def _reset_obstacles(self) -> None:
        self._obstacles.clear()
        self._obstacle_collision = False
        self._episode_obstacles_active = False
        if not self.obstacles_enabled:
            return
        if self._episode_count <= self.obstacle_start_episode:
            return
        if self.obstacle_episode_prob <= 0.0:
            return
        if np.random.random() > self.obstacle_episode_prob:
            return
        self._episode_obstacles_active = True
        if self.obstacle_max_static <= 0:
            return
        for _ in range(self.obstacle_max_static):
            obstacle = self._spawn_static_obstacle()
            if obstacle is not None:
                self._obstacles.append(obstacle)

    def _spawn_static_obstacle(self) -> _Obstacle | None:
        if self.static_size_range[1] <= 0.0:
            return None
        res = self.map_data.resolution
        attempts = max(1, int(self.static_spawn_attempts))
        for _ in range(attempts):
            idx = np.random.randint(0, len(self.free_positions))
            y, x = self.free_positions[idx]
            pos = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
            if self.static_min_distance > 0.0 and pos.distance_to(self.vehicle.position) < self.static_min_distance:
                continue
            half_size = self._sample_obstacle_half_size(self.static_size_range)
            if half_size is None:
                return None
            if not self._is_obstacle_location_valid(pos, half_size):
                continue
            if not self._is_obstacle_separated(pos, half_size):
                continue
            persistent = np.random.random() < self.static_persist_prob
            return _Obstacle(position=pos, half_size=half_size, dynamic=False, ttl=None, persistent=persistent)
        return None

    def _maybe_spawn_dynamic_obstacle(self, dt: float) -> None:
        if not self._episode_obstacles_active or not self.dynamic_enabled:
            return
        if self.dynamic_spawn_rate <= 0.0:
            return
        if self.dynamic_max_active <= 0:
            return
        if self._count_dynamic_obstacles() >= self.dynamic_max_active:
            return
        if len(self._obstacles) >= self.obstacle_max_total:
            return
        spawn_prob = 1.0 - math.exp(-self.dynamic_spawn_rate * max(0.0, dt))
        if np.random.random() >= spawn_prob:
            return
        attempts = max(1, int(self.dynamic_spawn_attempts))
        forward = pygame.Vector2(math.cos(self.vehicle.angle), math.sin(self.vehicle.angle))
        right = pygame.Vector2(-forward.y, forward.x)
        for _ in range(attempts):
            dist = max(0.0, self._sample_range(self.dynamic_distance_range))
            lateral = self._sample_range(self.dynamic_lateral_range)
            pos = self.vehicle.position + forward * dist + right * lateral
            half_size = self._sample_obstacle_half_size(self.dynamic_size_range)
            if half_size is None:
                return
            if not self._is_obstacle_location_valid(pos, half_size):
                continue
            if not self._is_obstacle_separated(pos, half_size):
                continue
            ttl_value = self._sample_range(self.dynamic_ttl_range)
            ttl = ttl_value if ttl_value > 0.0 else None
            self._obstacles.append(_Obstacle(position=pos, half_size=half_size, dynamic=True, ttl=ttl, persistent=False))
            break

    def _update_obstacles(self, dt: float) -> None:
        if not self._episode_obstacles_active:
            return
        remaining: list[_Obstacle] = []
        for obs in self._obstacles:
            if not obs.encountered and self.obstacle_pass_activate_dist > 0.0:
                if obs.position.distance_to(self.vehicle.position) <= self.obstacle_pass_activate_dist:
                    obs.encountered = True
            if obs.ttl is not None:
                obs.ttl -= dt
                if obs.ttl <= 0.0:
                    continue
            if obs.encountered and self._is_obstacle_passed(obs):
                if obs.dynamic or not obs.persistent:
                    continue
            remaining.append(obs)
        self._obstacles = remaining

    def _is_obstacle_passed(self, obs: _Obstacle) -> bool:
        forward = pygame.Vector2(math.cos(self.vehicle.angle), math.sin(self.vehicle.angle))
        rel = obs.position - self.vehicle.position
        return forward.dot(rel) < -self.obstacle_pass_buffer

    def _count_dynamic_obstacles(self) -> int:
        return sum(1 for obs in self._obstacles if obs.dynamic)

    def _is_obstacle_separated(self, pos: pygame.Vector2, half_size: pygame.Vector2) -> bool:
        if self.obstacle_min_separation <= 0.0:
            return True
        radius = self._obstacle_radius(half_size)
        for obs in self._obstacles:
            if pos.distance_to(obs.position) < (radius + obs.radius + self.obstacle_min_separation):
                return False
        return True

    def _is_obstacle_location_valid(self, pos: pygame.Vector2, half_size: pygame.Vector2) -> bool:
        if half_size.x <= 0.0 or half_size.y <= 0.0:
            return False
        res = self.map_data.resolution
        min_x = int(math.floor((pos.x - half_size.x) / res))
        max_x = int(math.floor((pos.x + half_size.x) / res))
        min_y = int(math.floor((pos.y - half_size.y) / res))
        max_y = int(math.floor((pos.y + half_size.y) / res))
        if min_x < 0 or min_y < 0 or max_x >= self.map_width or max_y >= self.map_height:
            return False
        center_x = int(math.floor(pos.x / res))
        center_y = int(math.floor(pos.y / res))
        if not self.map_data.free_mask[center_y, center_x]:
            return False
        if not self.obstacle_allow_wall_overlap:
            if np.any(self.occupied_mask[min_y : max_y + 1, min_x : max_x + 1]):
                return False
        radius = self._obstacle_radius(half_size)
        if not self.obstacle_allow_wall_overlap:
            min_wall = self._min_wall_distance_at(pos)
            if min_wall < radius + self.obstacle_min_wall_clearance:
                return False
        if self.obstacle_min_gap > 0.0 and not self._has_pass_gap(pos, radius, self.obstacle_min_gap):
            return False
        return True

    def _min_wall_distance_at(self, origin: pygame.Vector2) -> float:
        min_dist = self.lidar_max_range_m
        for angle in self.spawn_clearance_angles:
            dist, _ = self._cast_ray(origin, angle)
            if dist < min_dist:
                min_dist = dist
            if min_dist <= 0.0:
                break
        return min_dist

    def _has_pass_gap(self, origin: pygame.Vector2, radius: float, min_gap: float) -> bool:
        heading = self._clockwise_heading(origin)
        left_angle = heading + math.pi / 2.0
        right_angle = heading - math.pi / 2.0
        left_dist, _ = self._cast_ray(origin, left_angle)
        right_dist, _ = self._cast_ray(origin, right_angle)
        left_clear = left_dist - radius
        right_clear = right_dist - radius
        return left_clear >= min_gap or right_clear >= min_gap

    @staticmethod
    def _ray_aabb_intersection(
        origin: pygame.Vector2,
        direction: pygame.Vector2,
        min_pt: pygame.Vector2,
        max_pt: pygame.Vector2,
    ) -> float | None:
        eps = 1e-8
        t_min = -float("inf")
        t_max = float("inf")
        if abs(direction.x) < eps:
            if origin.x < min_pt.x or origin.x > max_pt.x:
                return None
        else:
            tx1 = (min_pt.x - origin.x) / direction.x
            tx2 = (max_pt.x - origin.x) / direction.x
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))
        if abs(direction.y) < eps:
            if origin.y < min_pt.y or origin.y > max_pt.y:
                return None
        else:
            ty1 = (min_pt.y - origin.y) / direction.y
            ty2 = (max_pt.y - origin.y) / direction.y
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))
        if t_max < max(t_min, 0.0):
            return None
        return t_min if t_min >= 0.0 else t_max

    def _ray_obstacle_intersection_cached(
        self,
        origin: pygame.Vector2,
        direction: pygame.Vector2,
        max_distance: float,
        obstacle_cache: list[Tuple[pygame.Vector2, pygame.Vector2, float]],
    ) -> Tuple[float | None, pygame.Vector2 | None]:
        best_dist: float | None = None
        best_point: pygame.Vector2 | None = None
        early_exit_dist = max_distance * 0.1
        _ray_aabb = self._ray_aabb_intersection
        ox = origin.x
        oy = origin.y
        for min_pt, max_pt, obs_radius in obstacle_cache:
            # Quick sphere pre-filter: skip if obstacle center is too far
            # The obstacle center is at midpoint of min_pt and max_pt
            cx = (min_pt.x + max_pt.x) * 0.5
            cy = (min_pt.y + max_pt.y) * 0.5
            dx = cx - ox
            dy = cy - oy
            center_dist_sq = dx * dx + dy * dy
            reach = max_distance + obs_radius
            if center_dist_sq > reach * reach:
                continue
            hit_dist = _ray_aabb(origin, direction, min_pt, max_pt)
            if hit_dist is None:
                continue
            if best_dist is not None and hit_dist >= best_dist:
                continue
            if hit_dist > max_distance:
                continue
            best_dist = hit_dist
            best_point = origin + direction * hit_dist
            if best_dist <= early_exit_dist:
                break
        return best_dist, best_point

    def _vehicle_hits_obstacle(self, obs: _Obstacle) -> bool:
        center = self.vehicle.position
        cos_a = math.cos(self.vehicle.angle)
        sin_a = math.sin(self.vehicle.angle)
        forward = pygame.Vector2(cos_a, sin_a)
        right = pygame.Vector2(-sin_a, cos_a)
        pad_m = self._collision_pad / max(self.ppm, 1e-6)
        half_len = (self.vehicle_params.length * 0.5) + pad_m
        half_w = (self.vehicle_params.width * 0.5) + pad_m
        rel = obs.position - center
        axes = (forward, right, pygame.Vector2(1.0, 0.0), pygame.Vector2(0.0, 1.0))
        for axis in axes:
            dist = abs(rel.dot(axis))
            r_vehicle = half_len * abs(axis.dot(forward)) + half_w * abs(axis.dot(right))
            r_obs = obs.half_size.x * abs(axis.x) + obs.half_size.y * abs(axis.y)
            if dist > r_vehicle + r_obs:
                return False
        return True

    def _action_to_vector(self, action: object) -> np.ndarray:
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
            if action_idx < 0 or action_idx >= len(self.actions):
                raise ValueError("Action index out of bounds")
            steer_value, accel_value = self.actions[action_idx]
            steer = (float(steer_value) - 10.0) / 10.0
            steer = max(-1.0, min(1.0, steer))
            accel_cmd = float(accel_value)
            return np.array([steer, accel_cmd], dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != 2:
            raise ValueError("Action must be an int index or a 2D continuous vector")
        steer = float(action_arr[0])
        accel_cmd = float(action_arr[1])
        return np.array([steer, accel_cmd], dtype=np.float32)

    @property
    def wants_new_action(self) -> bool:
        """True when action_repeat window expired and env needs a fresh action."""
        return self.action_repeat_steps <= 1 or self._repeat_count <= 0

    def _apply_action_repeat(self, action: np.ndarray) -> np.ndarray:
        if self.action_repeat_steps <= 1:
            return action
        if self._repeat_count <= 0 or self._repeat_action is None:
            self._repeat_action = action.copy()
            self._repeat_count = int(self.action_repeat_steps)
        self._repeat_count -= 1
        return self._repeat_action

    def _apply_action_delay(self, action: np.ndarray) -> np.ndarray:
        if self.action_delay_steps <= 0:
            return action
        self._action_delay_queue.append(action)
        if len(self._action_delay_queue) <= self.action_delay_steps:
            return self._neutral_action.copy()
        return self._action_delay_queue.popleft()

    def _apply_rate_limit(self, action: np.ndarray) -> np.ndarray:
        steer = float(action[0])
        accel = float(action[1])
        if self.steer_rate_limit > 0.0:
            steer = max(self._last_action[0] - self.steer_rate_limit, min(self._last_action[0] + self.steer_rate_limit, steer))
        if self.accel_rate_limit > 0.0:
            accel = max(self._last_action[1] - self.accel_rate_limit, min(self._last_action[1] + self.accel_rate_limit, accel))
        return np.array([steer, accel], dtype=np.float32)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        steer = float(action[0])
        accel = float(action[1])
        steer = max(-1.0, min(1.0, steer))
        accel = max(self.accel_min, min(self.accel_max, accel))
        return np.array([steer, accel], dtype=np.float32)

    def _prepare_action(self, action: object) -> Tuple[float, float]:
        action_vec = self._action_to_vector(action)
        action_vec = self._apply_action_repeat(action_vec)
        if self.action_noise_enabled:
            action_vec = action_vec * self._action_scale + self._action_bias
        action_vec = self._apply_action_delay(action_vec)
        action_vec = self._clip_action(action_vec)
        action_vec = self._apply_rate_limit(action_vec)
        action_vec = self._clip_action(action_vec)
        self._last_action = action_vec.copy()
        steer = float(action_vec[0])
        accel_cmd = float(action_vec[1])
        self.servo_value = steer * 10.0 + 10.0
        return steer, accel_cmd

    def _apply_observation_delay(self, obs: np.ndarray) -> np.ndarray:
        if not self.obs_delay_enabled:
            self._obs_delay_history.append(obs)
            return obs
        self._obs_delay_history.append(obs)
        if len(self._obs_delay_history) < 2:
            return obs
        p1 = max(0.0, min(1.0, self.obs_delay_p1))
        p2 = max(0.0, min(1.0, self.obs_delay_p2))
        if p1 + p2 > 1.0:
            p1 = max(0.0, 1.0 - p2)
        roll = float(np.random.random())
        if roll < p2 and len(self._obs_delay_history) >= 3:
            return self._obs_delay_history[-3]
        if roll < p2 + p1:
            return self._obs_delay_history[-2]
        return obs

    def _apply_perturbations(self, dt: float) -> None:
        if not self.perturb_enabled or self.perturb_prob <= 0.0:
            return
        if np.random.random() >= self.perturb_prob:
            return
        yaw_delta = float(np.random.normal(0.0, self.perturb_yaw_rate_sigma)) * dt
        speed_delta = float(np.random.normal(0.0, self.perturb_speed_sigma)) * dt
        self.vehicle.angle += yaw_delta
        self.vehicle.speed += speed_delta
        max_speed = self.vehicle_params.max_speed
        max_reverse = self.vehicle_params.max_reverse_speed
        if self.vehicle.speed > max_speed:
            self.vehicle.speed = max_speed
        elif self.vehicle.speed < -max_reverse:
            self.vehicle.speed = -max_reverse

    def _apply_wind_slope(self, dt: float) -> None:
        """S12: Apply wind gusts (lateral) and slope (longitudinal) perturbations."""
        if self.wind_enabled:
            # W2 fix: OU process for temporally correlated wind instead of i.i.d. noise
            theta = self._wind_ou_theta
            noise = float(np.random.normal(0.0, self.wind_force_std * math.sqrt(2.0 * theta * dt)))
            self._wind_ou_state += -theta * self._wind_ou_state * dt + noise
            t = self.episode_time_s
            wind_force = self._wind_ou_state * math.sin(
                2.0 * math.pi * t * self.wind_freq_hz + self._wind_phase
            )
            lateral_dir = pygame.Vector2(-math.sin(self.vehicle.angle), math.cos(self.vehicle.angle))
            self.vehicle.position += lateral_dir * (wind_force * dt)
        if self.slope_enabled and self._slope_angle != 0.0:
            gravity_accel = 9.81 * math.sin(self._slope_angle)
            self.vehicle.speed += gravity_accel * dt

    def _stuck_penalty(self, dt: float, collision: bool) -> float:
        if collision:
            self.stuck_time = 0.0
            return 0.0
        speed = abs(self.vehicle.speed)
        if speed < self.stuck_speed_thresh:
            self.stuck_time += dt
            if self.stuck_time <= self.stuck_grace_s:
                return 0.0
            t = self.stuck_time - self.stuck_grace_s
            penalty = -self.stuck_scale * (math.exp(t / self.stuck_tau) - 1.0) * dt
            if penalty < -self.stuck_penalty_cap:
                penalty = -self.stuck_penalty_cap
            return penalty
        self.stuck_time = 0.0
        return 0.0

    def _build_raceline_waypoints(self, mask: np.ndarray, resolution: float) -> None:
        """Build ordered waypoints from a painted raceline mask using double-BFS."""
        h, w = mask.shape
        rl_pixels = np.argwhere(mask)
        if len(rl_pixels) < 2:
            return

        NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # BFS 1: from arbitrary pixel to find one true endpoint
        start_y, start_x = int(rl_pixels[0, 0]), int(rl_pixels[0, 1])
        visited = np.full((h, w), -1, dtype=np.int32)
        visited[start_y, start_x] = 0
        queue = deque([(start_y, start_x)])
        farthest = (start_y, start_x)
        max_dist = 0
        while queue:
            y, x = queue.popleft()
            d = visited[y, x]
            for dy, dx in NEIGHBORS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] < 0 and mask[ny, nx]:
                    visited[ny, nx] = d + 1
                    queue.append((ny, nx))
                    if d + 1 > max_dist:
                        max_dist = d + 1
                        farthest = (ny, nx)

        # BFS 2: from true endpoint to get progress ordering
        ep_y, ep_x = farthest
        progress = np.full((h, w), -1, dtype=np.int32)
        progress[ep_y, ep_x] = 0
        queue = deque([(ep_y, ep_x)])
        max_dist2 = 0
        far2 = (ep_y, ep_x)
        while queue:
            y, x = queue.popleft()
            d = progress[y, x]
            for dy, dx in NEIGHBORS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and progress[ny, nx] < 0 and mask[ny, nx]:
                    progress[ny, nx] = d + 1
                    queue.append((ny, nx))
                    if d + 1 > max_dist2:
                        max_dist2 = d + 1
                        far2 = (ny, nx)

        if max_dist2 < 2:
            return

        # Extract centroid at each distance level → ordered waypoints
        step = max(1, max_dist2 // 300)
        waypoints = []
        for d in range(0, max_dist2 + 1, step):
            pixels_d = np.argwhere(progress == d)
            if len(pixels_d) == 0:
                continue
            cy = float(pixels_d[:, 0].mean())
            cx = float(pixels_d[:, 1].mean())
            waypoints.append(((cx + 0.5) * resolution, (cy + 0.5) * resolution))

        if len(waypoints) < 2:
            return

        self._raceline_waypoints = np.array(waypoints, dtype=np.float32)
        # Detect loop: start and end close together
        dist_se = np.linalg.norm(self._raceline_waypoints[-1] - self._raceline_waypoints[0])
        self._raceline_is_loop = dist_se < 5.0 * resolution * step
        print(f"Raceline: {len(waypoints)} waypoints, loop={self._raceline_is_loop}")

    def _clockwise_heading(self, position: pygame.Vector2) -> float:
        if self._raceline_waypoints is not None:
            pos = np.array([position.x, position.y], dtype=np.float32)
            dists_sq = np.sum((self._raceline_waypoints - pos) ** 2, axis=1)
            idx = int(np.argmin(dists_sq))
            if self._raceline_is_loop:
                next_idx = (idx + 1) % len(self._raceline_waypoints)
            else:
                next_idx = min(idx + 1, len(self._raceline_waypoints) - 1)
            d = self._raceline_waypoints[next_idx] - self._raceline_waypoints[idx]
            if d[0] ** 2 + d[1] ** 2 < 1e-12:
                return 0.0
            return math.atan2(float(d[1]), float(d[0]))
        radial = position - self.map_center
        if radial.length_squared() < 1e-6:
            return 0.0
        tangent = pygame.Vector2(-radial.y, radial.x) * self.track_direction
        if tangent.length_squared() < 1e-6:
            return 0.0
        return math.atan2(tangent.y, tangent.x)

    def _clockwise_alignment(self) -> float:
        delta = self.vehicle.position - self.prev_position
        if delta.length_squared() < 1e-6:
            return 0.0
        heading = delta.normalize()
        tangent_angle = self._clockwise_heading(self.vehicle.position)
        tangent = pygame.Vector2(math.cos(tangent_angle), math.sin(tangent_angle))
        dot = heading.dot(tangent)
        if dot <= 0.0:
            return 0.0
        return min(dot, 1.0)

    def _is_clockwise_motion(self) -> bool:
        return self._clockwise_alignment() > 0.0

    def _lookat_angle_from(self, origin: pygame.Vector2, lookat_positions: np.ndarray) -> float | None:
        """Pick a random look-at target pixel and return the angle from origin to it.

        Returns None if no valid (non-colocated) target found after retries.
        """
        res = self.map_data.resolution
        for _ in range(min(5, len(lookat_positions))):
            idx = np.random.randint(0, len(lookat_positions))
            ty, tx = lookat_positions[idx]
            target = pygame.Vector2((float(tx) + 0.5) * res, (float(ty) + 0.5) * res)
            diff = target - origin
            if diff.length_squared() >= 1e-6:
                return math.atan2(diff.y, diff.x)
        return None

    def _lookat_angle(self, origin: pygame.Vector2) -> float:
        """Pick a random look-at target pixel and return the angle from origin to it."""
        if self.lookat_positions is None:
            return self._clockwise_heading(origin)
        angle = self._lookat_angle_from(origin, self.lookat_positions)
        return angle if angle is not None else self._clockwise_heading(origin)

    def _random_spawn(self, min_clearance_m: float) -> Tuple[Tuple[float, float], float]:
        res = self.map_data.resolution

        # Zone-aware path: pick a random zone, use zone-specific positions + lookat
        if self._zone_ids:
            zone_id = self._zone_ids[np.random.randint(0, len(self._zone_ids))]
            zone_lookat = self._lookat_by_zone.get(zone_id)
            use_lookat = zone_lookat is not None and len(zone_lookat) > 0

            # Fast path: pre-filtered positions for this zone
            if zone_id in self._spawn_by_zone_prefiltered:
                positions = self._spawn_by_zone_prefiltered[zone_id]
                idx = np.random.randint(0, len(positions))
                y, x = positions[idx]
                origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
                if use_lookat:
                    angle = self._lookat_angle_from(origin, zone_lookat)
                    if angle is None:
                        angle = self._clockwise_heading(origin)
                else:
                    angle = self._clockwise_heading(origin)
                return (origin.x, origin.y), angle

            # Slow path with raycasting
            positions = self._spawn_by_zone[zone_id]
            attempts = min(1000, len(positions))
            for _ in range(attempts):
                idx = np.random.randint(0, len(positions))
                y, x = positions[idx]
                origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
                if self._min_wall_distance(origin) < self.spawn_wall_clearance_m:
                    continue
                if use_lookat:
                    angle = self._lookat_angle_from(origin, zone_lookat)
                    if angle is None:
                        angle = self._clockwise_heading(origin)
                    # With lookat, check radial clearance (not directional)
                    if self._min_wall_distance(origin) >= min_clearance_m:
                        return (origin.x, origin.y), angle
                else:
                    angle = self._clockwise_heading(origin)
                    distance_m, _ = self._cast_ray(origin, angle)
                    if self.spawn_face_forward:
                        back_dist, _ = self._cast_ray(origin, angle + math.pi)
                        if distance_m < min_clearance_m and back_dist >= min_clearance_m:
                            angle = (angle + math.pi) % (2.0 * math.pi)
                            distance_m = back_dist
                    if distance_m >= min_clearance_m:
                        return (origin.x, origin.y), angle

            # Fallback: random position from the zone
            idx = np.random.randint(0, len(positions))
            y, x = positions[idx]
            origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
            if use_lookat:
                angle = self._lookat_angle_from(origin, zone_lookat)
                if angle is None:
                    angle = self._clockwise_heading(origin)
            else:
                angle = self._clockwise_heading(origin)
            return (origin.x, origin.y), angle

        # Legacy path: no zones
        use_lookat = self.lookat_positions is not None

        # Fast path: pre-filtered positions have guaranteed clearance
        if self._spawn_prefiltered:
            idx = np.random.randint(0, len(self.spawn_positions))
            y, x = self.spawn_positions[idx]
            origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
            if use_lookat:
                angle = self._lookat_angle(origin)
            else:
                angle = self._clockwise_heading(origin)
            return (origin.x, origin.y), angle

        attempts = min(1000, len(self.spawn_positions))
        for _ in range(attempts):
            idx = np.random.randint(0, len(self.spawn_positions))
            y, x = self.spawn_positions[idx]
            origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
            if use_lookat:
                angle = self._lookat_angle(origin)
            else:
                angle = self._clockwise_heading(origin)
            distance_m, _ = self._cast_ray(origin, angle)
            if not use_lookat and self.spawn_face_forward:
                back_dist, _ = self._cast_ray(origin, angle + math.pi)
                if distance_m < min_clearance_m and back_dist >= min_clearance_m:
                    angle = (angle + math.pi) % (2.0 * math.pi)
                    distance_m = back_dist
            if distance_m >= min_clearance_m and self._min_wall_distance(origin) >= self.spawn_wall_clearance_m:
                return (origin.x, origin.y), angle

        idx = np.random.randint(0, len(self.spawn_positions))
        y, x = self.spawn_positions[idx]
        origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
        if use_lookat:
            angle = self._lookat_angle(origin)
        else:
            angle = self._clockwise_heading(origin)
            if self.spawn_face_forward:
                forward_dist, _ = self._cast_ray(origin, angle)
                back_dist, _ = self._cast_ray(origin, angle + math.pi)
                if forward_dist < min_clearance_m and back_dist > forward_dist:
                    angle = (angle + math.pi) % (2.0 * math.pi)
        return (origin.x, origin.y), angle

    def _min_wall_distance(self, origin: pygame.Vector2) -> float:
        min_dist = self.lidar_max_range_m
        for angle in self.spawn_clearance_angles:
            dist, _ = self._cast_ray(origin, angle)
            if dist < min_dist:
                min_dist = dist
            if min_dist < self.spawn_wall_clearance_m:
                break
        return min_dist

    def _cast_ray(self, origin: pygame.Vector2, angle: float) -> Tuple[float, pygame.Vector2]:
        res = self.map_data.resolution
        x0 = origin.x / res
        y0 = origin.y / res
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        max_distance_pixels = self.lidar_max_range_m / res

        if abs(dir_x) < 1e-9 and abs(dir_y) < 1e-9:
            return 0.0, origin

        cell_x = int(math.floor(x0))
        cell_y = int(math.floor(y0))
        if cell_x < 0 or cell_x >= self.map_width or cell_y < 0 or cell_y >= self.map_height:
            return 0.0, origin

        if self.occupied_mask[cell_y, cell_x]:
            return 0.0, origin

        if dir_x > 0:
            step_x = 1
            t_max_x = (cell_x + 1 - x0) / dir_x
            t_delta_x = 1.0 / dir_x
        elif dir_x < 0:
            step_x = -1
            t_max_x = (x0 - cell_x) / -dir_x
            t_delta_x = 1.0 / -dir_x
        else:
            step_x = 0
            t_max_x = float("inf")
            t_delta_x = float("inf")

        if dir_y > 0:
            step_y = 1
            t_max_y = (cell_y + 1 - y0) / dir_y
            t_delta_y = 1.0 / dir_y
        elif dir_y < 0:
            step_y = -1
            t_max_y = (y0 - cell_y) / -dir_y
            t_delta_y = 1.0 / -dir_y
        else:
            step_y = 0
            t_max_y = float("inf")
            t_delta_y = float("inf")

        distance_pixels = 0.0
        while True:
            if t_max_x < t_max_y:
                cell_x += step_x
                next_distance = t_max_x
                t_max_x += t_delta_x
            else:
                cell_y += step_y
                next_distance = t_max_y
                t_max_y += t_delta_y

            if next_distance > max_distance_pixels:
                distance_pixels = max_distance_pixels
                break

            distance_pixels = next_distance

            if cell_x < 0 or cell_x >= self.map_width or cell_y < 0 or cell_y >= self.map_height:
                distance_pixels = max_distance_pixels
                break

            if self.occupied_mask[cell_y, cell_x]:
                break

        distance_m = distance_pixels * res
        hit_point = origin + pygame.Vector2(dir_x, dir_y) * distance_m
        return distance_m, hit_point

    def _compute_lidar(self) -> list[Tuple[float, float, pygame.Vector2]]:
        origin = self.vehicle.position
        base_angle = self.vehicle.angle
        readings = []
        check_obstacles = self._episode_obstacles_active and bool(self._obstacles)

        # S6: Ego-motion blur — compute yaw rate for scan distortion
        n_rays = len(self.lidar_angles_deg)
        ego_blur = self.lidar_ego_motion_blur_enabled and n_rays > 1
        if ego_blur:
            # Compute yaw rate from bicycle model: yaw_rate = speed/wheelbase * tan(steer)
            wb = self.vehicle_params.wheelbase
            steer_angle = getattr(self.vehicle, 'servo_actual', 0.0)
            if wb > 0 and abs(steer_angle) > 1e-4 and abs(self.vehicle.speed) > 0.05:
                yaw_rate = (self.vehicle.speed / wb) * math.tan(steer_angle)
            else:
                yaw_rate = 0.0
            scan_time = self.lidar_scan_time_s

        if check_obstacles:
            # Cache obstacle AABB bounds and radii once for all rays
            obstacle_cache = [
                (obs.position - obs.half_size, obs.position + obs.half_size, obs.radius)
                for obs in self._obstacles
            ]
            _intersect = self._ray_obstacle_intersection_cached
            for i, (angle_deg, offset) in enumerate(zip(self.lidar_angles_deg, self.lidar_offsets)):
                ray_angle = base_angle + offset
                # S6: Apply ego-motion blur distortion
                if ego_blur:
                    ray_time_offset = scan_time * (i / n_rays)
                    ray_angle += yaw_rate * ray_time_offset
                distance_m, hit_point = self._cast_ray(origin, ray_angle)
                direction = pygame.Vector2(math.cos(ray_angle), math.sin(ray_angle))
                obs_dist, obs_hit = _intersect(origin, direction, distance_m, obstacle_cache)
                if obs_dist is not None and obs_hit is not None:
                    distance_m = obs_dist
                    hit_point = obs_hit
                readings.append((angle_deg, distance_m, hit_point))
        else:
            for i, (angle_deg, offset) in enumerate(zip(self.lidar_angles_deg, self.lidar_offsets)):
                ray_angle = base_angle + offset
                # S6: Apply ego-motion blur distortion
                if ego_blur:
                    ray_time_offset = scan_time * (i / n_rays)
                    ray_angle += yaw_rate * ray_time_offset
                distance_m, hit_point = self._cast_ray(origin, ray_angle)
                readings.append((angle_deg, distance_m, hit_point))
        return readings

    def _vehicle_collision(self) -> bool:
        self._obstacle_collision = False
        self._kill_boundary_hit = False
        center_px = self.vehicle.position * self.ppm
        cos_a = math.cos(self.vehicle.angle)
        sin_a = math.sin(self.vehicle.angle)
        half_len = (self.vehicle_params.length * self.ppm) / 2.0
        half_w = (self.vehicle_params.width * self.ppm) / 2.0
        pad = self._collision_pad

        exp_len = half_len + pad
        exp_w = half_w + pad
        fx = cos_a * exp_len
        fy = sin_a * exp_len
        rx = -sin_a * exp_w
        ry = cos_a * exp_w

        corners = (
            (center_px.x + fx + rx, center_px.y + fy + ry),
            (center_px.x + fx - rx, center_px.y + fy - ry),
            (center_px.x - fx + rx, center_px.y - fy + ry),
            (center_px.x - fx - rx, center_px.y - fy - ry),
        )
        min_x = int(math.floor(min(x for x, _ in corners)))
        max_x = int(math.ceil(max(x for x, _ in corners)))
        min_y = int(math.floor(min(y for _, y in corners)))
        max_y = int(math.ceil(max(y for _, y in corners)))

        if max_x < 0 or max_y < 0 or min_x >= self.map_width or min_y >= self.map_height:
            return False

        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, self.map_width - 1)
        max_y = min(max_y, self.map_height - 1)

        occ = self.collision_mask[min_y : max_y + 1, min_x : max_x + 1]
        if not np.any(occ):
            if self._episode_obstacles_active and self._obstacles:
                for obs in self._obstacles:
                    if self._vehicle_hits_obstacle(obs):
                        self._obstacle_collision = True
                        return True
            return False

        local_ys, local_xs = np.nonzero(occ)
        if local_xs.size == 0:
            return False

        xs = local_xs.astype(np.float32) + min_x + 0.5
        ys = local_ys.astype(np.float32) + min_y + 0.5
        dx = xs - center_px.x
        dy = ys - center_px.y

        proj_forward = dx * cos_a + dy * sin_a
        proj_right = -dx * sin_a + dy * cos_a
        inside = (np.abs(proj_forward) <= exp_len) & (np.abs(proj_right) <= exp_w)
        if np.any(inside):
            if self.map_data.kill_mask is not None:
                kill_sub = self.map_data.kill_mask[min_y : max_y + 1, min_x : max_x + 1]
                if np.any(kill_sub[local_ys[inside], local_xs[inside]]):
                    self._kill_boundary_hit = True
            return True
        if self._episode_obstacles_active and self._obstacles:
            for obs in self._obstacles:
                if self._vehicle_hits_obstacle(obs):
                    self._obstacle_collision = True
                    return True
        return False

    def _build_observation(self, readings: list[Tuple[float, float, pygame.Vector2]], collision: bool) -> np.ndarray:
        n_lidar = len(readings)
        # S7: obs_dim = n_lidar + 5 (collision, speed, servo, linear_accel, angular_vel)
        obs = np.empty(n_lidar + 5, dtype=np.float32)

        # Vectorized lidar normalization: extract distances in one shot
        inv_range = 1.0 / self.lidar_max_range_m
        for i in range(n_lidar):
            d = readings[i][1] * inv_range
            obs[i] = d if d < 1.0 else 1.0

        # S6: Beam divergence — distance-dependent noise (stacks with DR noise)
        if self.lidar_beam_divergence_enabled:
            lidar_slice = obs[:n_lidar]
            beam_std = self.lidar_beam_noise_std
            # Noise std scales linearly with normalized distance (0 at origin, beam_std at max range)
            noise_stds = beam_std * lidar_slice
            lidar_slice += np.random.normal(0.0, 1.0, size=n_lidar) * noise_stds
            np.clip(lidar_slice, 0.0, 1.0, out=lidar_slice)

        if self.obs_noise_enabled:
            lidar_slice = obs[:n_lidar]
            if self.lidar_noise_std > 0.0:
                lidar_slice += np.random.normal(0.0, self.lidar_noise_std, size=n_lidar)
            if self.lidar_dropout_prob > 0.0:
                drop_mask = np.random.random(size=n_lidar) < self.lidar_dropout_prob
                lidar_slice[drop_mask] = 1.0
            if self.lidar_spike_prob > 0.0:
                spike_mask = np.random.random(size=n_lidar) < self.lidar_spike_prob
                lidar_slice[spike_mask] = 0.0
            np.clip(lidar_slice, 0.0, 1.0, out=lidar_slice)

        # S14: Sensor thermal drift (Ornstein-Uhlenbeck)
        if self.thermal_drift_enabled:
            dt_ou = self.last_dt
            sqrt_dt = math.sqrt(dt_ou) if dt_ou > 0 else 0.0
            self._lidar_bias += (
                self._ou_theta * (0.0 - self._lidar_bias) * dt_ou
                + self._ou_sigma_lidar * sqrt_dt * float(np.random.normal())
            )
            self._speed_bias += (
                self._ou_theta * (0.0 - self._speed_bias) * dt_ou
                + self._ou_sigma_speed * sqrt_dt * float(np.random.normal())
            )
            lidar_slice = obs[:n_lidar]
            lidar_slice += self._lidar_bias
            np.clip(lidar_slice, 0.0, 1.0, out=lidar_slice)

        # S7: Encoder noise on speed measurement
        speed_raw = self.vehicle.speed
        if self.encoder_noise_enabled:
            noise_std = self.encoder_noise_std * max(abs(speed_raw), 0.1)
            speed_raw = speed_raw + float(np.random.normal(0.0, noise_std))
        # S14: Apply speed thermal drift bias
        if self.thermal_drift_enabled:
            speed_raw += self._speed_bias

        speed_kmh = abs(speed_raw) * 3.6
        max_speed_kmh = max(self.vehicle_params.max_speed * 3.6, 1e-3)
        speed_norm = min(speed_kmh / max_speed_kmh, 1.0)
        servo_norm = min(max(self.servo_value / 20.0, 0.0), 1.0)
        if self.obs_noise_enabled:
            if self.speed_noise_std > 0.0:
                speed_norm += float(np.random.normal(0.0, self.speed_noise_std))
            if self.servo_noise_std > 0.0:
                servo_norm += float(np.random.normal(0.0, self.servo_noise_std))
            speed_norm = min(max(speed_norm, 0.0), 1.0)
            servo_norm = min(max(servo_norm, 0.0), 1.0)

        # S7: IMU values — linear acceleration and angular velocity
        dt = self.last_dt
        if dt > 0:
            linear_accel = (self.vehicle.speed - self._prev_speed) / dt
            angular_velocity = (self.vehicle.angle - self._prev_angle) / dt
        else:
            linear_accel = 0.0
            angular_velocity = 0.0
        self._prev_speed = self.vehicle.speed
        self._prev_angle = self.vehicle.angle
        linear_accel_norm = max(-1.0, min(1.0, linear_accel / self.imu_max_accel))
        angular_vel_norm = max(-1.0, min(1.0, angular_velocity / self.imu_max_yaw_rate))

        # S10: Async sensor delays — push current values, pull delayed
        if self.sensor_delay_enabled:
            self._lidar_obs_history.append(obs[:n_lidar].copy())
            self._speed_obs_history.append(speed_norm)
            self._imu_obs_history.append((linear_accel_norm, angular_vel_norm))

            # W1 fix: use per-episode base delay with ±1 frame jitter
            lidar_delay = max(0, self._ep_lidar_delay + int(np.random.randint(-1, 2)))
            speed_delay = max(0, self._ep_speed_delay + int(np.random.randint(-1, 2)))
            imu_delay = max(0, self._ep_imu_delay + int(np.random.randint(-1, 2)))

            lidar_idx = min(lidar_delay, len(self._lidar_obs_history) - 1)
            speed_idx = min(speed_delay, len(self._speed_obs_history) - 1)
            imu_idx = min(imu_delay, len(self._imu_obs_history) - 1)

            obs[:n_lidar] = self._lidar_obs_history[-(lidar_idx + 1)]
            speed_norm = self._speed_obs_history[-(speed_idx + 1)]
            linear_accel_norm, angular_vel_norm = self._imu_obs_history[-(imu_idx + 1)]

        obs[n_lidar] = 1.0 if collision else 0.0
        obs[n_lidar + 1] = speed_norm
        obs[n_lidar + 2] = servo_norm
        obs[n_lidar + 3] = linear_accel_norm
        obs[n_lidar + 4] = angular_vel_norm
        return obs

    def _compute_reward(self, readings: list[Tuple[float, float, pygame.Vector2]], collision: bool) -> float:
        if collision:
            return self.reward_collision_penalty

        distances = []
        front_distances = []
        left_distances = []
        right_distances = []
        for angle_deg, distance_m, _ in readings:
            dist = min(distance_m / self.lidar_max_range_m, 1.0)
            distances.append(dist)
            delta = angle_deg - LIDAR_CENTER_DEG
            # Only use forward hemisphere (0°–180°) for front/left/right split
            # to avoid rear rays creating asymmetry in balance penalty.
            if angle_deg > 180.0:
                continue
            if abs(delta) <= self.reward_front_cone_deg:
                front_distances.append(dist)
            elif delta < 0:
                left_distances.append(dist)
            else:
                right_distances.append(dist)

        if not distances:
            return 0.0

        min_clear = min(distances)
        front = sum(front_distances) / max(1, len(front_distances)) if front_distances else min_clear
        left = sum(left_distances) / max(1, len(left_distances)) if left_distances else front
        right = sum(right_distances) / max(1, len(right_distances)) if right_distances else front

        max_speed = max(self.vehicle_params.max_speed, 1e-3)
        signed_speed = self.vehicle.speed
        forward_speed_norm = min(max(signed_speed, 0.0) / max_speed, 1.0)
        reverse_speed_norm = min(max(-signed_speed, 0.0) / max_speed, 1.0)
        side_clear = 0.5 * (left + right)

        if self.distance_progress_enabled:
            # S9: Distance progress reward — project onto track heading (not vehicle heading)
            delta_pos = self.vehicle.position - self.prev_position
            tangent_angle = self._clockwise_heading(self.vehicle.position)
            heading = pygame.Vector2(math.cos(tangent_angle), math.sin(tangent_angle))
            forward_progress = max(0.0, delta_pos.dot(heading))
            reward_forward = (
                self.reward_forward_speed_weight * forward_speed_norm
                + self.reward_front_speed_weight * forward_speed_norm * front
            )
            reward = reward_forward + self.reward_distance_progress_weight * forward_progress
        else:
            alignment = self._clockwise_alignment()
            reward_forward = alignment * (
                self.reward_forward_speed_weight * forward_speed_norm
                + self.reward_front_speed_weight * forward_speed_norm * front
            )
            reward = reward_forward + self.reward_alignment_bonus * alignment
        reward -= self.reward_front_penalty * forward_speed_norm * (1.0 - front)
        reward -= self.reward_side_penalty * (1.0 - side_clear)
        reward -= self.reward_min_clear_penalty * (1.0 - min_clear)
        reward -= self.reward_balance_penalty * abs(left - right)
        reward -= self.reward_reverse_penalty * reverse_speed_norm
        return reward

    def reset(self) -> np.ndarray:
        self._episode_count += 1
        self._apply_domain_randomization()
        spawn_position, spawn_angle = self._random_spawn(self.spawn_clearance_m)
        self.spawn_angle = spawn_angle
        self.vehicle.position = pygame.Vector2(spawn_position)
        self.vehicle.angle = spawn_angle
        self.vehicle.speed = 0.0
        # B1: Reset actuator lag state from previous episode
        self.vehicle.servo_actual = 0.0
        self.vehicle.accel_actual = 0.0
        self.vehicle.yaw_rate = 0.0
        self.servo_value = 10.0
        self.prev_position = self.vehicle.position.copy()
        self.stuck_time = 0.0
        self.backward_distance_m = 0.0
        self.backward_penalized = False
        self.backward_death = False
        self.episode_time_s = 0.0
        # S7: Reset IMU state
        self._prev_speed = 0.0
        self._prev_angle = spawn_angle
        # S8: Reset contact counter
        self._contact_count = 0
        # S10: Reset sensor delay buffers and sample per-episode delays
        self._lidar_obs_history.clear()
        self._speed_obs_history.clear()
        self._imu_obs_history.clear()
        if self.sensor_delay_enabled:
            self._ep_lidar_delay = self._sample_int_range(self.lidar_delay_range)
            self._ep_speed_delay = self._sample_int_range(self.speed_delay_range)
            self._ep_imu_delay = self._sample_int_range(self.imu_delay_range)
        # S14: Reset thermal drift biases
        self._lidar_bias = 0.0
        self._speed_bias = 0.0
        # S12: Randomize per-episode wind/slope
        if self.slope_enabled:
            self._slope_angle = float(np.random.uniform(self.slope_range[0], self.slope_range[1]))
        else:
            self._slope_angle = 0.0
        if self.wind_enabled:
            self._wind_phase = float(np.random.uniform(0.0, 2.0 * math.pi))
            self._wind_ou_state = 0.0
        self._reset_obstacles()
        lidar_readings = self._compute_lidar()
        collision = self._vehicle_collision()
        obs = self._build_observation(lidar_readings, collision)
        obs = self._apply_observation_delay(obs)
        return self._stack_reset(obs)

    def step(self, action: object) -> Tuple[np.ndarray, float, bool, float, str]:
        steer, accel_cmd = self._prepare_action(action)

        if self.render_requested and self.clock is not None:
            dt = self.clock.tick(self.fps) / 1000.0
            dt = min(dt, 0.05)
        else:
            dt = self.fixed_dt
        if self.dt_jitter_enabled and not self.render_requested:
            dt *= self._sample_range(self.dt_scale_range)
        self.last_dt = dt
        self.episode_time_s += dt

        self._maybe_spawn_dynamic_obstacle(dt)

        # S13: Continuous DR — battery sag and friction cycle
        step_map_params = self.map_params
        if self.continuous_dr_enabled:
            max_time = self.episode_time_limit_s if self.episode_time_limit_s > 0.0 else 60.0
            time_frac = min(self.episode_time_s / max_time, 1.0)
            battery_factor = 1.0 - self.battery_sag_pct * time_frac
            accel_cmd = accel_cmd * battery_factor
            if self.friction_cycle_period_s > 0.0:
                friction_drift = self.friction_cycle_amplitude * math.sin(
                    2.0 * math.pi * self.episode_time_s / self.friction_cycle_period_s
                )
                new_friction = max(0.0, self.map_params.surface_friction * (1.0 + friction_drift))
                step_map_params = replace(self.map_params, surface_friction=new_friction)

        self.vehicle.update(dt, False, False, steer, step_map_params, accel_cmd=accel_cmd)
        self._apply_perturbations(dt)
        self._apply_wind_slope(dt)
        lidar_readings = self._compute_lidar()
        collision = self._vehicle_collision()

        # S8: Soft collision — allow light contacts before termination
        hard_collision = collision
        if collision and self.soft_collision_enabled:
            self._contact_count += 1
            if self._contact_count <= self.max_light_contacts:
                hard_collision = False  # Treat as light contact, don't terminate

        reward = self._compute_reward(lidar_readings, hard_collision)
        # S8: Apply light contact penalty for soft collisions
        if collision and not hard_collision:
            reward += self.light_contact_penalty
        reward += self._stuck_penalty(dt, hard_collision)
        reward *= self.reward_scale
        if self.reward_clip is not None:
            reward = max(-self.reward_clip, min(self.reward_clip, reward))
        self._backward_penalty(dt)
        obs = self._build_observation(lidar_readings, collision)
        obs = self._apply_observation_delay(obs)
        obs = self._stack_append(obs)
        self._update_obstacles(dt)
        distance_delta = self.vehicle.position.distance_to(self.prev_position)
        self.prev_position = self.vehicle.position.copy()

        if self.render_requested:
            self._render(lidar_readings)

        timeout = self.episode_time_limit_s > 0.0 and self.episode_time_s >= self.episode_time_limit_s
        done = hard_collision or self.backward_death or timeout
        if hard_collision:
            if self._obstacle_collision:
                death_reason = "obstacle"
            elif self._kill_boundary_hit:
                death_reason = "kill_boundary"
            else:
                death_reason = "collision"
        elif self.backward_death:
            death_reason = "reverse_limit"
        elif timeout:
            death_reason = "timeout"
        else:
            death_reason = ""
        return obs, reward, done, distance_delta, death_reason

    def _stack_reset(self, obs: np.ndarray) -> np.ndarray:
        if self.stack_frames <= 1:
            self._obs_stack.clear()
            self._obs_stack.append(obs)
            return obs
        self._obs_stack.clear()
        for _ in range(self.stack_frames):
            self._obs_stack.append(obs)
        return np.concatenate(list(self._obs_stack))

    def _stack_append(self, obs: np.ndarray) -> np.ndarray:
        if self.stack_frames <= 1:
            self._obs_stack.clear()
            self._obs_stack.append(obs)
            return obs
        if not self._obs_stack:
            return self._stack_reset(obs)
        self._obs_stack.append(obs)
        return np.concatenate(list(self._obs_stack))

    def _backward_penalty(self, dt: float) -> None:
        if self.backward_penalized:
            return
        speed = float(self.vehicle.speed)
        if speed < 0.0:
            self.backward_distance_m += (-speed) * dt
            if self.backward_distance_m >= self.backward_limit_m:
                self.backward_penalized = True
                self.backward_death = True
        return

    def _render(self, readings: list[Tuple[float, float, pygame.Vector2]]) -> None:
        if not self.render_requested or self.screen is None:
            return
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.render_enabled = False
                elif event.type == pygame.MOUSEWHEEL:
                    self.zoom = min(self.zoom_max, max(self.zoom_min, self.zoom + event.y * self.zoom_step))
                    self._update_scaled_map_surface()
            self.screen.fill((10, 10, 10))
            offset = self._compute_camera_offset()
            if self.map_surface_scaled is not None:
                self.screen.blit(self.map_surface_scaled, (-offset.x, -offset.y))
            self._draw_obstacles(offset)
            self._draw_lidar(readings, offset)
            self.vehicle.draw(self.screen, self.zoom, offset)
            pygame.display.flip()
        else:
            pygame.event.pump()

    def _draw_lidar(self, readings: list[Tuple[float, float, pygame.Vector2]], offset: pygame.Vector2) -> None:
        if self.screen is None:
            return
        origin_px = self.vehicle.position * self.ppm * self.zoom - offset
        for _, _, hit_point in readings:
            hit_px = hit_point * self.ppm * self.zoom - offset
            pygame.draw.line(self.screen, (220, 30, 30), origin_px, hit_px, 1)

    def _draw_obstacles(self, offset: pygame.Vector2) -> None:
        if self.screen is None or not self._obstacles:
            return
        for obs in self._obstacles:
            if obs.dynamic:
                color = (230, 80, 80)
            elif obs.persistent:
                color = (240, 150, 60)
            else:
                color = (250, 200, 70)
            pos_px = obs.position * self.ppm * self.zoom - offset
            width_px = max(1, int(round(obs.half_size.x * 2.0 * self.ppm * self.zoom)))
            height_px = max(1, int(round(obs.half_size.y * 2.0 * self.ppm * self.zoom)))
            rect = pygame.Rect(
                int(pos_px.x - width_px * 0.5),
                int(pos_px.y - height_px * 0.5),
                width_px,
                height_px,
            )
            pygame.draw.rect(self.screen, color, rect, 0)

    def _init_render(self) -> None:
        if self.screen is not None:
            return
        pygame.display.init()
        pygame.display.set_caption("Occupancy Racer")
        self.map_surface_base = create_map_surface(
            self.map_data.image,
            kill_mask=self.map_data.kill_mask,
            spawn_mask=self.map_data.spawn_mask,
            lookat_mask=self.map_data.lookat_mask,
            raceline_mask=self.map_data.raceline_mask,
        )
        self._update_scaled_map_surface()
        window_size = (self.map_width, self.map_height)
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()

    def enable_render(self) -> None:
        self.render_requested = True
        self.render_enabled = True
        self.vehicle.enable_rendering()
        self._init_render()

    def disable_render(self) -> None:
        self.render_enabled = False
        self.render_requested = False
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock = None
            self.map_surface_base = None
            self.map_surface_scaled = None

    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
        pygame.quit()

    def _update_scaled_map_surface(self) -> None:
        if self.map_surface_base is None:
            return
        width = max(1, int(round(self.map_width * self.zoom)))
        height = max(1, int(round(self.map_height * self.zoom)))
        self.map_surface_scaled = pygame.transform.smoothscale(self.map_surface_base, (width, height))

    def _compute_camera_offset(self) -> pygame.Vector2:
        if self.screen is None or self.map_surface_scaled is None:
            return pygame.Vector2(0, 0)
        screen_w, screen_h = self.screen.get_size()
        scaled_w, scaled_h = self.map_surface_scaled.get_size()
        center = pygame.Vector2(screen_w / 2.0, screen_h / 2.0)
        vehicle_px = self.vehicle.position * self.ppm * self.zoom
        offset = vehicle_px - center
        max_x = max(0.0, float(scaled_w - screen_w))
        max_y = max(0.0, float(scaled_h - screen_h))
        offset.x = min(max(offset.x, 0.0), max_x)
        offset.y = min(max(offset.y, 0.0), max_y)
        return offset
