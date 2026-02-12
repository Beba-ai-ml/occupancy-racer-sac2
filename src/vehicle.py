from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import pygame


@dataclass(frozen=True)
class VehicleParams:
    acceleration: float
    brake_deceleration: float
    reverse_acceleration: float
    max_speed: float
    max_reverse_speed: float
    friction: float
    drag: float
    max_steer_angle: float
    wheelbase: float
    length: float
    width: float


@dataclass(frozen=True)
class MapParams:
    surface_friction: float
    surface_drag: float


class Vehicle:
    def __init__(
        self,
        params: VehicleParams,
        position: Tuple[float, float],
        pixels_per_meter: float,
        angle: float = 0.0,
        render_enabled: bool = True,
    ) -> None:
        self.params = params
        self.position = pygame.Vector2(position)
        self.angle = angle
        self.speed = 0.0
        self.ppm = float(pixels_per_meter)
        self.render_enabled = render_enabled
        self._base_texture = self._create_texture() if render_enabled else None
        self._scaled_texture = None
        self._scaled_for = None

    def _create_texture(self) -> pygame.Surface:
        length = max(2, int(round(self.params.length * self.ppm)))
        width = max(2, int(round(self.params.width * self.ppm)))
        surface = pygame.Surface((length, width), pygame.SRCALPHA)
        body_color = (32, 120, 185)
        body_shadow = (22, 90, 140)
        outline_color = (10, 30, 50)
        glass_color = (165, 200, 220)
        glass_dark = (125, 160, 180)
        wheel_color = (20, 20, 20)
        wheel_rim = (70, 70, 70)
        headlight_color = (235, 230, 190)
        taillight_color = (210, 50, 50)
        stripe_color = (245, 245, 245)

        radius = max(2, width // 5)
        body_rect = pygame.Rect(0, 0, length, width)
        pygame.draw.rect(surface, body_color, body_rect, border_radius=radius)
        pygame.draw.rect(surface, outline_color, body_rect, width=1, border_radius=radius)

        shade_rect = pygame.Rect(1, 1, max(1, length - 2), max(1, width - 2))
        pygame.draw.rect(surface, body_shadow, shade_rect, border_radius=radius)

        roof_length = max(2, int(length * 0.55))
        roof_width = max(2, int(width * 0.65))
        roof_x = int(length * 0.22)
        roof_y = max(0, (width - roof_width) // 2)
        roof_rect = pygame.Rect(roof_x, roof_y, roof_length, roof_width)
        pygame.draw.rect(surface, glass_color, roof_rect, border_radius=max(2, roof_width // 4))

        windshield_length = max(2, int(length * 0.18))
        windshield_rect = pygame.Rect(
            roof_x + roof_length - windshield_length,
            roof_y + 2,
            windshield_length,
            max(2, roof_width - 4),
        )
        pygame.draw.rect(surface, glass_dark, windshield_rect, border_radius=max(2, roof_width // 5))

        rear_length = max(2, int(length * 0.14))
        rear_rect = pygame.Rect(
            roof_x,
            roof_y + 3,
            rear_length,
            max(2, roof_width - 6),
        )
        pygame.draw.rect(surface, glass_dark, rear_rect, border_radius=max(2, roof_width // 5))

        stripe_w = max(2, int(length * 0.12))
        stripe_h = max(2, int(width * 0.12))
        stripe_x = int(length * 0.60)
        stripe_rect = pygame.Rect(stripe_x, (width - stripe_h) // 2, stripe_w, stripe_h)
        pygame.draw.rect(surface, stripe_color, stripe_rect, border_radius=max(1, stripe_h // 2))

        wheel_length = max(2, int(length * 0.14))
        wheel_width = max(2, int(width * 0.22))
        wheel_inset_x = max(1, int(length * 0.08))
        wheel_offset_y = max(1, int(width * 0.08))
        wheel_front_x = max(0, length - wheel_inset_x - wheel_length)
        wheel_rear_x = wheel_inset_x
        wheel_top_y = wheel_offset_y
        wheel_bottom_y = max(0, width - wheel_offset_y - wheel_width)

        wheel_rects = [
            pygame.Rect(wheel_rear_x, wheel_top_y, wheel_length, wheel_width),
            pygame.Rect(wheel_rear_x, wheel_bottom_y, wheel_length, wheel_width),
            pygame.Rect(wheel_front_x, wheel_top_y, wheel_length, wheel_width),
            pygame.Rect(wheel_front_x, wheel_bottom_y, wheel_length, wheel_width),
        ]
        for rect in wheel_rects:
            pygame.draw.rect(surface, wheel_color, rect, border_radius=max(1, wheel_width // 3))
            rim_rect = rect.inflate(-max(1, wheel_length // 4), -max(1, wheel_width // 4))
            pygame.draw.rect(surface, wheel_rim, rim_rect, border_radius=max(1, wheel_width // 4))

        head_w = max(2, int(length * 0.06))
        head_h = max(2, int(width * 0.18))
        head_x = max(0, length - head_w - 1)
        head_y_top = max(0, int(width * 0.1))
        head_y_bottom = max(0, width - head_h - int(width * 0.1))
        pygame.draw.rect(
            surface,
            headlight_color,
            pygame.Rect(head_x, head_y_top, head_w, head_h),
            border_radius=2,
        )
        pygame.draw.rect(
            surface,
            headlight_color,
            pygame.Rect(head_x, head_y_bottom, head_w, head_h),
            border_radius=2,
        )

        tail_w = max(2, int(length * 0.05))
        tail_h = max(2, int(width * 0.18))
        tail_x = 1
        tail_y_top = max(0, int(width * 0.12))
        tail_y_bottom = max(0, width - tail_h - int(width * 0.12))
        pygame.draw.rect(
            surface,
            taillight_color,
            pygame.Rect(tail_x, tail_y_top, tail_w, tail_h),
            border_radius=2,
        )
        pygame.draw.rect(
            surface,
            taillight_color,
            pygame.Rect(tail_x, tail_y_bottom, tail_w, tail_h),
            border_radius=2,
        )

        return surface

    def _get_scaled_texture(self, scale: float) -> pygame.Surface:
        if self._base_texture is None:
            raise RuntimeError("Vehicle texture requested with rendering disabled")
        if self._scaled_for == scale and self._scaled_texture is not None:
            return self._scaled_texture

        length = max(2, int(round(self.params.length * self.ppm * scale)))
        width = max(2, int(round(self.params.width * self.ppm * scale)))
        self._scaled_texture = pygame.transform.smoothscale(self._base_texture, (length, width))
        self._scaled_for = scale
        return self._scaled_texture

    def enable_rendering(self) -> None:
        if self.render_enabled:
            return
        self.render_enabled = True
        self._base_texture = self._create_texture()
        self._scaled_texture = None
        self._scaled_for = None

    def update(
        self,
        dt: float,
        throttle: bool,
        brake: bool,
        steer: float,
        map_params: MapParams,
        accel_cmd: float | None = None,
    ) -> None:
        params = self.params
        accel = 0.0
        if accel_cmd is None:
            if throttle and not brake:
                accel += params.acceleration
            if brake:
                if self.speed > 0:
                    accel -= params.brake_deceleration
                else:
                    accel -= params.reverse_acceleration
        else:
            accel = float(accel_cmd)

        if accel == 0.0:
            friction = params.friction * map_params.surface_friction
            drag = params.drag + map_params.surface_drag
            decel = friction + drag * abs(self.speed)
            if decel > 0.0 and self.speed != 0.0:
                sign = 1.0 if self.speed > 0 else -1.0
                self.speed -= sign * decel * dt
                if self.speed * sign < 0:
                    self.speed = 0.0
        else:
            self.speed += accel * dt

        if self.speed > params.max_speed:
            self.speed = params.max_speed
        elif self.speed < -params.max_reverse_speed:
            self.speed = -params.max_reverse_speed

        if steer != 0.0 and abs(self.speed) > 0.05 and params.wheelbase > 0:
            max_steer_angle = params.max_steer_angle
            min_steer_angle = math.radians(5.0)
            if max_steer_angle < min_steer_angle:
                min_steer_angle = max_steer_angle

            if params.max_speed > 0:
                speed_ratio = min(abs(self.speed) / params.max_speed, 1.0)
            else:
                speed_ratio = 0.0

            steer_limit = max_steer_angle + (min_steer_angle - max_steer_angle) * speed_ratio
            steer_angle = steer * steer_limit
            if abs(steer_angle) > 1e-4:
                yaw_rate = (self.speed / params.wheelbase) * math.tan(steer_angle)
                self.angle += yaw_rate * dt

        direction = pygame.Vector2(math.cos(self.angle), math.sin(self.angle))
        self.position += direction * self.speed * dt

    def draw(self, surface: pygame.Surface, scale: float, offset: pygame.Vector2) -> None:
        if not self.render_enabled:
            return
        texture = self._get_scaled_texture(scale)
        rotated = pygame.transform.rotate(texture, -math.degrees(self.angle))
        pos_px = self.position * self.ppm * scale - offset
        rect = rotated.get_rect(center=(pos_px.x, pos_px.y))
        surface.blit(rotated, rect)
