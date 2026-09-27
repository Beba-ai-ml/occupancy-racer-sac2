"""Time-based steering transport delay and finite servo travel, independent of FPS."""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SteeringProfile:
    delay_s: float
    full_travel_s: float
    slow_diameter_m: float
    fast_diameter_m: float
    reference_speed_mps: float
    timing_scale_range: tuple[float, float] = (1.0, 1.0)
    mid_speed_mps: float | None = None
    mid_diameter_m: float | None = None
    fast_diameter_range_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        values = (self.full_travel_s, self.slow_diameter_m, self.fast_diameter_m,
                  self.reference_speed_mps, *self.timing_scale_range)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Steering times, diameters, reference speed and timing scales must be positive and finite")
        if not math.isfinite(self.delay_s) or self.delay_s < 0:
            raise ValueError("Steering delay must be finite and nonnegative")
        if self.fast_diameter_m < self.slow_diameter_m:
            raise ValueError("Fast turning diameter must not be smaller than slow diameter")
        if (self.mid_speed_mps is None) != (self.mid_diameter_m is None):
            raise ValueError("Intermediate speed and diameter must be provided together")
        if self.mid_speed_mps is not None:
            if not (math.isfinite(self.mid_speed_mps) and 0 < self.mid_speed_mps < self.reference_speed_mps):
                raise ValueError("Intermediate speed must be between zero and reference speed")
            if not (math.isfinite(self.mid_diameter_m)
                    and self.slow_diameter_m < self.mid_diameter_m < self.fast_diameter_m):
                raise ValueError("Intermediate diameter must be between slow and fast diameters")
        if self.fast_diameter_range_m is not None:
            if (len(self.fast_diameter_range_m) != 2
                    or any(not math.isfinite(v) for v in self.fast_diameter_range_m)
                    or not self.slow_diameter_m < self.fast_diameter_range_m[0] <= self.fast_diameter_range_m[1]):
                raise ValueError("Fast diameter range must be ordered and above slow diameter")
        if len(self.timing_scale_range) != 2 or self.timing_scale_range[0] > self.timing_scale_range[1]:
            raise ValueError("Timing scales must contain ordered min/max values")

    def turning_radius(self, speed_mps: float) -> float:
        # Empirical understeer fit, not a tire-force model. Low-speed asymptote
        # and the diameter at reference_speed are user-supplied estimates.
        ratio = abs(speed_mps) / self.reference_speed_mps
        exponent = 2.0
        if self.mid_speed_mps is not None:
            diameter_fraction = ((self.mid_diameter_m - self.slow_diameter_m)
                                 / (self.fast_diameter_m - self.slow_diameter_m))
            exponent = math.log(diameter_fraction) / math.log(self.mid_speed_mps / self.reference_speed_mps)
        return 0.5 * (self.slow_diameter_m
                      + (self.fast_diameter_m - self.slow_diameter_m) * ratio ** exponent)


class SteeringActuator:
    def __init__(self, profile: SteeringProfile) -> None:
        self.profile = profile
        self.reset()

    def reset(self, delay_scale: float = 1.0, travel_scale: float = 1.0) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (delay_scale, travel_scale)):
            raise ValueError("Timing scales must be positive and finite")
        self.delay_s = self.profile.delay_s * delay_scale
        self.rate = 2.0 / (self.profile.full_travel_s * travel_scale)
        self.time_s = 0.0
        self.actual = 0.0
        self.target = 0.0
        self._submitted = 0.0
        self._pending: deque[tuple[float, float]] = deque()

    def _move(self, duration: float) -> None:
        change = self.target - self.actual
        limit = self.rate * max(0.0, duration)
        self.actual += max(-limit, min(limit, change))

    def step(self, command: float, dt: float) -> float:
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(command):
            raise ValueError("Steering command must be finite and dt positive and finite")
        command = max(-1.0, min(1.0, command))
        if command != self._submitted:
            self._pending.append((self.time_s + self.delay_s, command))
            self._submitted = command
        end = self.time_s + dt
        # Split at arrival times: no movement before the transport delay, and
        # no extra one-frame lag or dependence on how dt is partitioned.
        cursor = self.time_s
        while self._pending and self._pending[0][0] <= end + 1e-12:
            arrival, target = self._pending.popleft()
            arrival = min(end, max(cursor, arrival))
            self._move(arrival - cursor)
            self.target = target
            cursor = arrival
        self._move(end - cursor)
        self.time_s = end
        return self.actual
