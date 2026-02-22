from __future__ import annotations

from collections import deque
import csv
import json
import math
import os
import random
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pygame

from .map_loader import MapData
from .racer_env import LIDAR_ANGLES_DEG, LIDAR_MAX_RANGE_M
from .vehicle import MapParams, Vehicle, VehicleParams


def _parse_hidden_sizes(raw: object, default: list[int]) -> list[int]:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            return default
        return [int(part) for part in parts]
    return default


def _parse_target_entropy(raw: object, action_dim: int) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ("", "auto", "none"):
            return None
        return float(value)
    return float(raw)


def _resolve_accel_range(accel_bins: object) -> tuple[float, float]:
    if accel_bins is None:
        values = [-2.0, 2.0]
    else:
        values = [float(v) for v in accel_bins]
    if not values:
        values = [-2.0, 2.0]
    accel_min = float(min(values))
    accel_max = float(max(values))
    if accel_min > accel_max:
        accel_min, accel_max = accel_max, accel_min
    return accel_min, accel_max


def _resolve_checkpoint_path(load_from: str) -> str | None:
    if os.path.isdir(load_from):
        try:
            entries = [name for name in os.listdir(load_from) if name.endswith(".pth")]
        except FileNotFoundError:
            return None
        if not entries:
            return None
        session_entries = [name for name in entries if name.startswith("session_")]
        if session_entries:
            entries = session_entries
        entries.sort(
            key=lambda name: os.path.getmtime(os.path.join(load_from, name)),
            reverse=True,
        )
        return os.path.join(load_from, entries[0])
    return load_from


def create_map_surface(image: np.ndarray) -> pygame.Surface:
    if image.ndim != 2:
        raise ValueError("Map image must be a 2D array")
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return surface


class Game:
    def __init__(
        self,
        map_data: MapData,
        vehicle_params: VehicleParams,
        map_params: MapParams,
        display_cfg: Dict[str, float],
        window_cfg: Dict[str, float],
        control_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
    ) -> None:
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        pygame.display.set_caption(window_cfg.get("caption", "Occupancy Racer"))

        self.map_data = map_data
        self.vehicle_params = vehicle_params
        self.map_params = map_params
        self.occupied_mask = map_data.occupied_mask
        self.map_height, self.map_width = map_data.image.shape
        self.map_center = pygame.Vector2(
            (self.map_width * map_data.resolution) / 2.0,
            (self.map_height * map_data.resolution) / 2.0,
        )
        self.lidar_angles_deg = list(LIDAR_ANGLES_DEG)
        self.lidar_offsets = [math.radians(90.0 - angle) for angle in self.lidar_angles_deg]
        self.lidar_max_range_m = float(LIDAR_MAX_RANGE_M)
        self.lidar_state_dim = len(self.lidar_angles_deg) + 5  # +5: collision, speed, servo, linear_accel, angular_vel
        self._lidar_line_len = 0

        self.base_scale = self._compute_base_scale(display_cfg, window_cfg)
        self.zoom_cfg = display_cfg.get("zoom", {})
        self.zoom = 1.0
        self.scale = self.base_scale * self.zoom

        self.map_surface_base = create_map_surface(map_data.image)
        self.map_surface_scaled = None
        self.map_size_scaled = pygame.Vector2(0, 0)
        self._update_scaled_map_surface()

        self.screen_size = self._compute_screen_size()
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        self.fps = int(display_cfg.get("fps", 60))

        self.ppm = 1.0 / map_data.resolution
        self.free_positions = np.argwhere(map_data.free_mask)
        if self.free_positions.size == 0:
            raise ValueError("No free space found in map to place vehicle")

        self.spawn_angle = 0.0
        self.spawn_clearance_m = max(self.vehicle_params.length, self.vehicle_params.width)
        self.spawn_wall_clearance_m = 1.5
        if self.spawn_clearance_m < self.spawn_wall_clearance_m:
            self.spawn_clearance_m = self.spawn_wall_clearance_m
        self.spawn_clearance_angles = [
            float(i) * (2.0 * math.pi / 16.0) for i in range(16)
        ]
        self.vehicle_half_length_px = (self.vehicle_params.length * self.ppm) / 2.0
        self.vehicle_half_width_px = (self.vehicle_params.width * self.ppm) / 2.0
        self._collision_pad = 0.5

        spawn_position, spawn_angle = self._random_spawn(self.spawn_clearance_m)
        self.spawn_angle = spawn_angle
        self.vehicle = Vehicle(vehicle_params, spawn_position, self.ppm, angle=spawn_angle)
        self.control_mode = str(control_cfg.get("mode", "human")).lower()
        self.rl_cfg = rl_cfg
        self.servo_value = 10.0
        self.agent = None
        self.accel_bins = rl_cfg.get("accel_bins", [-2.0, -1.0, 0.0, 1.0, 2.0])
        self.accel_min, self.accel_max = _resolve_accel_range(self.accel_bins)
        self.episode_reward = 0.0
        self.episode_penalty = 0.0
        self.episode_distance = 0.0
        self.episode_time = 0.0
        self.prev_position = self.vehicle.position.copy()
        self.stuck_time = 0.0
        self.stuck_speed_thresh = 0.1
        self.stuck_grace_s = 0.5
        self.stuck_tau = 1.5
        self.stuck_scale = 3.0
        self.stuck_penalty_cap = 0.5
        self.reward_scale = 0.8
        self.reward_clip = 20.0
        self.backward_distance_m = 0.0
        self.backward_limit_m = 4.0
        self.backward_penalized = False
        self.backward_death = False
        self.stack_frames = int(rl_cfg.get("stack_frames", 1))
        self._obs_stack: deque[np.ndarray] = deque(maxlen=max(1, self.stack_frames))
        self._stack_reset = True
        self._prev_speed = 0.0
        self._prev_angle = self.vehicle.angle
        self.distance_history = deque(maxlen=100)
        self.total_episodes = 0
        self.episodes_done = 0
        self.save_dir: str | None = None
        self.save_every = 0
        self.session_id = "default"
        self.session_ckpt: str | None = None
        self.session_meta: str | None = None
        self.session_csv: str | None = None
        self.load_from: str | None = None
        if self.control_mode == "rl":
            from .rl_agent import SACAgent

            accel_min, accel_max = _resolve_accel_range(self.accel_bins)
            action_scale = np.array([1.0, (accel_max - accel_min) * 0.5], dtype=np.float32)
            action_bias = np.array([0.0, (accel_max + accel_min) * 0.5], dtype=np.float32)
            hidden_sizes = _parse_hidden_sizes(rl_cfg.get("hidden_sizes"), [128, 128])
            target_entropy = _parse_target_entropy(rl_cfg.get("target_entropy"), 2)

            self.agent = SACAgent(
                state_dim=self.lidar_state_dim * max(1, self.stack_frames),
                action_dim=2,
                action_scale=action_scale,
                action_bias=action_bias,
                policy_lr=float(rl_cfg.get("policy_lr", 3e-4)),
                q_lr=float(rl_cfg.get("q_lr", 3e-4)),
                alpha_lr=float(rl_cfg.get("alpha_lr", 3e-4)),
                gamma=float(rl_cfg.get("gamma", 0.99)),
                tau=float(rl_cfg.get("tau", rl_cfg.get("soft_tau", 0.005))),
                batch_size=int(rl_cfg.get("batch_size", 256)),
                memory_size=int(rl_cfg.get("memory_size", 100000)),
                target_entropy=target_entropy,
                init_alpha=float(rl_cfg.get("init_alpha", 0.2)),
                start_steps=int(rl_cfg.get("start_steps", 0)),
                learn_after=int(rl_cfg.get("learn_after", 1000)),
                update_every=int(rl_cfg.get("update_every", 1)),
                updates_per_step=int(rl_cfg.get("updates_per_step", 1)),
                hidden_sizes=hidden_sizes,
                device=rl_cfg.get("device", "auto"),
            )
            print(f"Using device: {self.agent.device}")
            self.save_dir = str(rl_cfg.get("save_dir", "runs"))
            self.save_every = int(rl_cfg.get("save_every", 0))
            self.session_id = str(rl_cfg.get("session_id", "default"))
            self.load_from = rl_cfg.get("load_from")
            os.makedirs(self.save_dir, exist_ok=True)
            session_dir = os.path.join(self.save_dir, f"session_{self.session_id}")
            os.makedirs(session_dir, exist_ok=True)
            self.session_ckpt = os.path.join(session_dir, f"session_{self.session_id}.pth")
            self.session_meta = os.path.join(session_dir, f"session_{self.session_id}.json")
            self.session_csv = os.path.join(session_dir, f"session_{self.session_id}.csv")
            legacy_session_ckpt = os.path.join(self.save_dir, f"session_{self.session_id}.pth")
            episodes_completed = 0
            if self.load_from:
                resolved_load = _resolve_checkpoint_path(self.load_from)
                if resolved_load is None:
                    print(f"Failed to locate checkpoint in {self.load_from}; starting fresh.")
                else:
                    try:
                        meta = self.agent.load_checkpoint(resolved_load)
                        episodes_completed = int(meta.get("episodes_trained", 0))
                    except Exception as exc:
                        print(f"Failed to load checkpoint ({exc}); starting fresh.")
            elif os.path.exists(self.session_ckpt):
                try:
                    meta = self.agent.load_checkpoint(self.session_ckpt)
                    episodes_completed = int(meta.get("episodes_trained", 0))
                except Exception as exc:
                    print(f"Failed to load checkpoint ({exc}); starting fresh.")
            elif os.path.exists(legacy_session_ckpt):
                try:
                    meta = self.agent.load_checkpoint(legacy_session_ckpt)
                    episodes_completed = int(meta.get("episodes_trained", 0))
                except Exception as exc:
                    print(f"Failed to load checkpoint ({exc}); starting fresh.")
            self.total_episodes = episodes_completed

    def _compute_base_scale(self, display_cfg: Dict[str, float], window_cfg: Dict[str, float]) -> float:
        scale = float(display_cfg.get("scale", 1))
        if scale <= 0:
            scale = 1.0

        max_width = window_cfg.get("max_width")
        max_height = window_cfg.get("max_height")
        img_height, img_width = self.map_data.image.shape

        if max_width:
            scale = min(scale, float(max_width) / img_width)
        if max_height:
            scale = min(scale, float(max_height) / img_height)

        return max(scale, 0.1)

    def _compute_screen_size(self) -> Tuple[int, int]:
        width = max(1, int(self.map_data.image.shape[1] * self.base_scale))
        height = max(1, int(self.map_data.image.shape[0] * self.base_scale))
        return width, height

    def _update_scaled_map_surface(self) -> None:
        width = max(1, int(self.map_data.image.shape[1] * self.scale))
        height = max(1, int(self.map_data.image.shape[0] * self.scale))
        self.map_surface_scaled = pygame.transform.smoothscale(self.map_surface_base, (width, height))
        self.map_size_scaled = pygame.Vector2(width, height)

    def _clockwise_heading(self, position: pygame.Vector2) -> float:
        radial = position - self.map_center
        if radial.length_squared() < 1e-6:
            return 0.0
        tangent = pygame.Vector2(-radial.y, radial.x)
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

    def _random_spawn(self, min_clearance_m: float) -> Tuple[Tuple[float, float], float]:
        res = self.map_data.resolution
        attempts = min(1000, len(self.free_positions))
        for _ in range(attempts):
            idx = random.randrange(len(self.free_positions))
            y, x = self.free_positions[idx]
            origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
            angle = self._clockwise_heading(origin)
            distance_m, _ = self._cast_ray(origin, angle)
            if distance_m >= min_clearance_m and self._min_wall_distance(origin) >= self.spawn_wall_clearance_m:
                return (origin.x, origin.y), angle

        idx = random.randrange(len(self.free_positions))
        y, x = self.free_positions[idx]
        origin = pygame.Vector2((float(x) + 0.5) * res, (float(y) + 0.5) * res)
        angle = self._clockwise_heading(origin)
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

    def _read_controls(self) -> Tuple[bool, bool, float]:
        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_w]
        brake = keys[pygame.K_s]
        steer = 0.0
        if keys[pygame.K_a]:
            steer -= 1.0
        if keys[pygame.K_d]:
            steer += 1.0
        return throttle, brake, steer

    def _apply_zoom(self, direction: float) -> None:
        zoom_cfg = self.zoom_cfg
        min_zoom = float(zoom_cfg.get("min", 0.5))
        max_zoom = float(zoom_cfg.get("max", 3.0))
        step = float(zoom_cfg.get("step", 0.1))

        delta = step * (1 if direction > 0 else -1) * max(1, abs(direction))
        new_zoom = self.zoom + delta
        self.zoom = max(min_zoom, min(max_zoom, new_zoom))
        self.scale = self.base_scale * self.zoom
        self._update_scaled_map_surface()

    def _camera_offset(self) -> pygame.Vector2:
        screen_width, screen_height = self.screen_size
        screen_center = pygame.Vector2(screen_width / 2, screen_height / 2)
        vehicle_px = self.vehicle.position * self.ppm * self.scale
        raw_offset = vehicle_px - screen_center

        map_width = self.map_size_scaled.x
        map_height = self.map_size_scaled.y

        if map_width <= screen_width:
            offset_x = -(screen_width - map_width) / 2
        else:
            offset_x = min(max(raw_offset.x, 0), map_width - screen_width)

        if map_height <= screen_height:
            offset_y = -(screen_height - map_height) / 2
        else:
            offset_y = min(max(raw_offset.y, 0), map_height - screen_height)

        return pygame.Vector2(offset_x, offset_y)

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

    def _compute_lidar(self) -> list[Tuple[int, float, pygame.Vector2]]:
        origin = self.vehicle.position
        base_angle = self.vehicle.angle
        readings = []
        for angle_deg, offset in zip(self.lidar_angles_deg, self.lidar_offsets):
            distance_m, hit_point = self._cast_ray(origin, base_angle + offset)
            readings.append((angle_deg, distance_m, hit_point))
        return readings

    def _vehicle_collision(self) -> bool:
        center_px = self.vehicle.position * self.ppm
        cos_a = math.cos(self.vehicle.angle)
        sin_a = math.sin(self.vehicle.angle)
        half_len = self.vehicle_half_length_px
        half_w = self.vehicle_half_width_px
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

        occ = self.occupied_mask[min_y : max_y + 1, min_x : max_x + 1]
        if not np.any(occ):
            return False

        ys, xs = np.nonzero(occ)
        if xs.size == 0:
            return False

        xs = xs.astype(np.float32) + min_x + 0.5
        ys = ys.astype(np.float32) + min_y + 0.5
        dx = xs - center_px.x
        dy = ys - center_px.y

        proj_forward = dx * cos_a + dy * sin_a
        proj_right = -dx * sin_a + dy * cos_a
        inside = (np.abs(proj_forward) <= exp_len) & (np.abs(proj_right) <= exp_w)
        return bool(np.any(inside))

    def _draw_lidar(self, readings: list[Tuple[int, float, pygame.Vector2]], offset: pygame.Vector2) -> None:
        origin_px = self.vehicle.position * self.ppm * self.scale - offset
        for _, _, hit_point in readings:
            hit_px = hit_point * self.ppm * self.scale - offset
            pygame.draw.line(self.screen, (220, 30, 30), origin_px, hit_px, 1)

    def _print_lidar(self, readings: list[Tuple[int, float, pygame.Vector2]], collision: bool) -> None:
        parts = ["Stopnie:"]
        for angle_deg, distance_m, _ in readings:
            distance_cm = round(distance_m * 100.0, 2)
            parts.append(f"{angle_deg}: {distance_cm:.2f}cm")
        parts.append(f"Kolizja: {'TAK' if collision else 'NIE'}")
        line = " ".join(parts)
        if len(line) < self._lidar_line_len:
            line = line.ljust(self._lidar_line_len)
        self._lidar_line_len = len(line)
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def _build_observation(self, readings: list[Tuple[int, float, pygame.Vector2]], collision: bool) -> np.ndarray:
        lidar_norm = []
        range_mm = self.lidar_max_range_m * 1000.0
        for _, distance_m, _ in readings:
            distance_mm = distance_m * 1000.0
            lidar_norm.append(min(distance_mm / range_mm, 1.0))

        speed_kmh = abs(self.vehicle.speed) * 3.6
        max_speed_kmh = max(self.vehicle_params.max_speed * 3.6, 1e-3)
        speed_norm = min(speed_kmh / max_speed_kmh, 1.0)
        servo_norm = min(max(self.servo_value / 20.0, 0.0), 1.0)

        # IMU channels: linear_accel and angular_vel (matching racer_env +5 layout)
        dt = 1.0 / max(self.fps, 1)
        linear_accel = (self.vehicle.speed - self._prev_speed) / dt if dt > 0 else 0.0
        angular_velocity = (self.vehicle.angle - self._prev_angle) / dt if dt > 0 else 0.0
        self._prev_speed = self.vehicle.speed
        self._prev_angle = self.vehicle.angle
        max_accel = 4.0
        max_yaw_rate = 3.0
        linear_accel_norm = max(-1.0, min(1.0, linear_accel / max_accel))
        angular_vel_norm = max(-1.0, min(1.0, angular_velocity / max_yaw_rate))

        obs = np.array(
            lidar_norm + [1.0 if collision else 0.0, speed_norm, servo_norm, linear_accel_norm, angular_vel_norm],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self, readings: list[Tuple[int, float, pygame.Vector2]], collision: bool) -> float:
        if collision:
            return -5.0

        distances = [min(d / self.lidar_max_range_m, 1.0) for _, d, _ in readings]
        n = len(distances)
        mid = n // 2
        front = (distances[mid - 1] + distances[mid]) * 0.5
        left = sum(distances[: n // 3]) / max(1, n // 3)
        right = sum(distances[-(n // 3) :]) / max(1, n // 3)
        min_clear = min(distances)

        max_speed = max(self.vehicle_params.max_speed, 1e-3)
        signed_speed = self.vehicle.speed
        forward_speed_norm = min(max(signed_speed, 0.0) / max_speed, 1.0)
        reverse_speed_norm = min(max(-signed_speed, 0.0) / max_speed, 1.0)
        side_clear = 0.5 * (left + right)

        alignment = self._clockwise_alignment()
        reward_forward = alignment * (0.6 * forward_speed_norm + 0.2 * forward_speed_norm * front)
        reward = reward_forward + 0.02 * alignment
        reward -= 0.02 * (1.0 - side_clear)
        reward -= 0.05 * (1.0 - min_clear)
        reward -= 0.05 * reverse_speed_norm
        return reward

    def _reset_episode(self) -> None:
        spawn_position, spawn_angle = self._random_spawn(self.spawn_clearance_m)
        self.spawn_angle = spawn_angle
        self.vehicle.position = pygame.Vector2(spawn_position)
        self.vehicle.angle = spawn_angle
        self.vehicle.speed = 0.0
        self.vehicle.servo_actual = 0.0
        self.vehicle.accel_actual = 0.0
        self.vehicle.yaw_rate = 0.0
        self.servo_value = 10.0
        self.episode_reward = 0.0
        self.episode_penalty = 0.0
        self.episode_distance = 0.0
        self.episode_time = 0.0
        self.prev_position = self.vehicle.position.copy()
        self.backward_distance_m = 0.0
        self.backward_penalized = False
        self.backward_death = False
        self.stuck_time = 0.0
        self._stack_reset = True
        self._prev_speed = 0.0
        self._prev_angle = spawn_angle
        self.stuck_time = 0.0

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

    def _stack_reset_frames(self, obs: np.ndarray) -> np.ndarray:
        if self.stack_frames <= 1:
            self._obs_stack.clear()
            self._obs_stack.append(obs)
            return obs
        self._obs_stack.clear()
        for _ in range(self.stack_frames):
            self._obs_stack.append(obs)
        return np.concatenate(list(self._obs_stack))

    def _stack_current(self) -> np.ndarray:
        if self.stack_frames <= 1:
            return self._obs_stack[-1]
        return np.concatenate(list(self._obs_stack))

    def _stack_append(self, obs: np.ndarray) -> np.ndarray:
        if self.stack_frames <= 1:
            self._obs_stack.clear()
            self._obs_stack.append(obs)
            return obs
        if not self._obs_stack:
            return self._stack_reset_frames(obs)
        self._obs_stack.append(obs)
        return np.concatenate(list(self._obs_stack))

    def _mean_distance(self, window: int) -> float:
        if not self.distance_history:
            return 0.0
        items = list(self.distance_history)[-window:]
        return float(sum(items) / len(items)) if items else 0.0

    def _log_episode(self, death_reason: str, mean_20: float, mean_100: float) -> None:
        if not self.session_csv or self.agent is None:
            return
        csv_needs_header = not os.path.exists(self.session_csv) or os.path.getsize(self.session_csv) == 0
        with open(self.session_csv, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "total_episode",
                    "episode_in_run",
                    "distance_m",
                    "time_s",
                    "reward_sum",
                    "penalty_sum",
                    "mean_20",
                    "mean_100",
                    "alpha",
                    "q_loss",
                    "policy_loss",
                    "alpha_loss",
                    "entropy",
                    "death_reason",
                ],
            )
            if csv_needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "total_episode": self.total_episodes,
                    "episode_in_run": self.episodes_done,
                    "distance_m": f"{self.episode_distance:.4f}",
                    "time_s": f"{self.episode_time:.2f}",
                    "reward_sum": f"{self.episode_reward:.6f}",
                    "penalty_sum": f"{self.episode_penalty:.6f}",
                    "mean_20": f"{mean_20:.4f}",
                    "mean_100": f"{mean_100:.4f}",
                    "alpha": f"{self.agent.alpha.item():.6f}",
                    "q_loss": f"{self.agent.last_q_loss:.6f}" if self.agent.last_q_loss is not None else "",
                    "policy_loss": f"{self.agent.last_policy_loss:.6f}" if self.agent.last_policy_loss is not None else "",
                    "alpha_loss": f"{self.agent.last_alpha_loss:.6f}" if self.agent.last_alpha_loss is not None else "",
                    "entropy": f"{self.agent.last_entropy:.6f}" if self.agent.last_entropy is not None else "",
                    "death_reason": death_reason,
                }
            )
            csv_file.flush()

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.fps) / 1000.0
            dt = min(dt, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEWHEEL:
                    self._apply_zoom(event.y)

            accel_cmd = None
            throttle = False
            brake = False
            steer = 0.0

            if self.control_mode == "rl" and self.agent is not None:
                self.episode_time += dt
                pre_lidar = self._compute_lidar()
                pre_collision = self._vehicle_collision()
                obs_raw = self._build_observation(pre_lidar, pre_collision)
                if self.stack_frames <= 1:
                    obs = obs_raw
                elif self._stack_reset or not self._obs_stack:
                    obs = self._stack_reset_frames(obs_raw)
                    self._stack_reset = False
                else:
                    obs = self._stack_current()
                action = self.agent.select_action(obs, deterministic=False)
                steer = float(action[0])
                steer = max(-1.0, min(1.0, steer))
                self.servo_value = steer * 10.0 + 10.0
                accel_cmd = max(self.accel_min, min(self.accel_max, float(action[1])))
            else:
                throttle, brake, steer = self._read_controls()

            self.vehicle.update(dt, throttle, brake, steer, self.map_params, accel_cmd=accel_cmd)
            lidar_readings = self._compute_lidar()
            collision = self._vehicle_collision()
            if self.control_mode == "rl" and self.agent is not None:
                next_obs_raw = self._build_observation(lidar_readings, collision)
                if self.stack_frames <= 1:
                    next_obs = next_obs_raw
                else:
                    next_obs = self._stack_append(next_obs_raw)
                reward = self._compute_reward(lidar_readings, collision)
                reward += self._stuck_penalty(dt, collision)
                reward *= self.reward_scale
                if self.reward_clip is not None:
                    reward = max(-self.reward_clip, min(self.reward_clip, reward))
                self._backward_penalty(dt)
                self.episode_reward += reward
                if reward < 0:
                    self.episode_penalty += -reward
                self.episode_distance += self.vehicle.position.distance_to(self.prev_position)
                self.prev_position = self.vehicle.position.copy()
                done = collision or self.backward_death
                if collision:
                    death_reason = "collision"
                elif self.backward_death:
                    death_reason = "reverse_limit"
                else:
                    death_reason = ""
                self.agent.step(obs, action, reward, next_obs, done)
                if done:
                    self.total_episodes += 1
                    self.episodes_done += 1
                    self.distance_history.append(self.episode_distance)
                    mean_20 = self._mean_distance(20)
                    mean_100 = self._mean_distance(100)
                    print(
                        f"\nEpisode end: time={self.episode_time:.2f}s distance={self.episode_distance:.2f}m "
                        f"reward_sum={self.episode_reward:.2f} penalty_sum={self.episode_penalty:.2f} "
                        f"mean20={mean_20:.2f}m mean100={mean_100:.2f}m"
                    )
                    self._log_episode(death_reason, mean_20, mean_100)
                    if self.save_every > 0 and self.total_episodes % self.save_every == 0:
                        meta = {
                            "session_id": self.session_id,
                            "episodes_trained": self.total_episodes,
                            "alpha": float(self.agent.alpha.item()),
                            "num_envs": 1,
                        }
                        if self.session_ckpt:
                            self.agent.save_checkpoint(self.session_ckpt, meta=meta)
                        if self.session_meta:
                            with open(self.session_meta, "w", encoding="utf-8") as handle:
                                json.dump(meta, handle)
                    self._reset_episode()

            offset = self._camera_offset()
            self.screen.fill((10, 10, 10))
            self.screen.blit(self.map_surface_scaled, (-offset.x, -offset.y))
            self._draw_lidar(lidar_readings, offset)
            self.vehicle.draw(self.screen, self.scale, offset)
            if self.control_mode != "rl":
                self._print_lidar(lidar_readings, collision)

            pygame.display.flip()

        sys.stdout.write("\n")
        pygame.quit()
