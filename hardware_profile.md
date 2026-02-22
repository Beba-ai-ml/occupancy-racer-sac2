# F1Tenth Hardware Profile — Real Vehicle Parameters

> Created: 2026-02-15
> Purpose: Reference for sim-to-real calibration. All values from owner interview.

---

## Platform

| Parameter | Value | Notes |
|-----------|-------|-------|
| Platform | F1Tenth | AliExpress F1Tenth kit |
| Compute | Jetson Nano Orin Dev Kit 8GB | On-vehicle inference |
| Middleware | ROS2 Foxy | Topics for LiDAR, VESC, IMU |
| Nawierzchnia | Gładka podłoga (hala) | Low grip — significant tire slip |

## Sensors

| Sensor | Rate | Noise/Accuracy | Notes |
|--------|------|----------------|-------|
| LiDAR | **8 Hz** | Standard RPLidar noise | 125ms between scans |
| VESC speed | ~50 Hz (default) | ~3-5% estimated | Telemetry from ESC |
| IMU (VESC built-in) | Unknown (likely 50-100 Hz) | Standard MEMS | Accel + gyro available via ROS2 |

## Actuators

| Actuator | Response Time | Notes |
|----------|---------------|-------|
| Steering servo | **~50ms** (estimated) | F1Tenth standard servo, has deadband on small commands (plan to eliminate) |
| ESC/Motor | **~100ms** | From command to speed change |
| Servo deadband | Present | Small steer commands (<2-3°) ignored; user plans to fix mechanically |

## Dynamics

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max speed | **7-8 m/s** | ~25-29 km/h |
| Tire slip | **Heavy** | At >4-5 m/s turning radius 2-3x larger than expected |
| Slip onset | ~4-5 m/s | Below this speed, grip is OK |
| Understeer/oversteer | Understeer dominant | On smooth floor, front tires lose grip first |
| Grip level | Low | Smooth floor + rubber tires = limited traction |

## Timing

| Metric | Value | Notes |
|--------|-------|-------|
| LiDAR scan period | **125ms** (8 Hz) | |
| Total closed-loop latency | **80-100ms** | LiDAR scan → inference → servo response |
| Inference time (Jetson) | ~5-10ms | SAC policy is small MLP — fast |
| Servo lag | ~50ms | From PWM command to physical movement |
| Motor lag | ~100ms | From throttle command to wheel speed change |

---

## Sim-to-Real Changes — Calibrated Values

### S1. Servo/Motor Lag (`vehicle.py`)
```
servo_tau = 0.050   # 50ms first-order filter
motor_tau = 0.100   # 100ms first-order filter

# Implementation:
servo_actual += (servo_target - servo_actual) * (dt / servo_tau)
accel_actual += (accel_cmd - accel_actual) * (dt / motor_tau)
```

### S2. Control Frequency (config YAML)
```
# LiDAR = 8 Hz → 125ms per scan
# At 60 FPS sim: 125ms / 16.7ms = 7.5 → round to 8
action_repeat: 8    # hold each action for 8 sim frames = 133ms ≈ 125ms
```

### S3. Action-to-Effect Latency (config YAML)
```
# Total closed-loop: 80-100ms
# At 60 FPS: 80ms = 4.8 frames, 100ms = 6 frames
dr_delay_steps: [4, 6]    # 67-100ms delay range
```

### S4. Tire Slip Model (`vehicle.py`)
```
# Heavy slip on smooth floor at >4-5 m/s
# Turning radius 2-3x expected → grip_factor = 0.33-0.50

slip_speed_threshold = 4.0          # m/s — slip onset
critical_slip_angle = 0.175         # ~10 degrees (low grip floor)
min_grip_factor = 0.4               # At max slip: 40% grip remaining
grip_reduction_speed_scale = 0.1    # How fast grip drops with speed

# Implementation:
if abs(speed) > slip_speed_threshold:
    speed_factor = (abs(speed) - slip_speed_threshold) / slip_speed_threshold
    slip_angle = steer_angle - atan2(lateral_vel, max(forward_vel, 0.1))
    if abs(slip_angle) > critical_slip_angle:
        grip = max(min_grip_factor, 1.0 - speed_factor * grip_reduction_speed_scale)
        yaw_rate *= grip
```

### S5. Quadratic Drag (`vehicle.py`)
```
# Replace linear drag with v² drag
# Old: drag_force = drag_coeff * speed
# New: drag_force = drag_coeff * speed * abs(speed)

drag_force = self.drag * self.speed * abs(self.speed)
```

### S6. LiDAR Beam Divergence + Ego-Motion (`racer_env.py`)
```
# Beam divergence: noise scales with distance
beam_noise_std = 0.03  # base noise
distance_noise = beam_noise_std * (distance / max_range)  # 0 at origin, 3% at max

# Ego-motion blur: 125ms scan time (8 Hz)
scan_time = 0.125  # seconds
yaw_rate = vehicle.yaw_rate  # rad/s
for i, ray in enumerate(rays):
    ray_time_offset = scan_time * (i / n_rays)  # 0 to 125ms
    ray_angle_distortion = yaw_rate * ray_time_offset
    actual_ray_angle = nominal_angle + ray_angle_distortion
```

### S7. IMU in Observation + Encoder Noise (`racer_env.py`)
```
# Observation: 30-dim → 32-dim per frame
# Add: linear_acceleration (normalized), angular_velocity (normalized)

# VESC speed noise: ~3%
speed_noise = np.random.normal(0, 0.03 * abs(speed))
speed_measured = speed + speed_noise

# IMU values:
linear_accel = (speed_new - speed_old) / dt  # m/s²
angular_velocity = (angle_new - angle_old) / dt  # rad/s

# Normalize:
linear_accel_norm = clip(linear_accel / max_accel, -1, 1)
angular_vel_norm = clip(angular_velocity / max_yaw_rate, -1, 1)

# New obs: [27 lidar, collision, speed, servo, linear_accel, angular_vel]
# obs_dim = 32, stacked = 32 * 4 = 128
```

### S8. Soft Collision (`racer_env.py`)
```
# Allow 2 light contacts before termination
max_light_contacts = 2
light_contact_penalty = -5.0    # vs -20.0 for hard termination

if collision:
    contact_count += 1
    if contact_count <= max_light_contacts:
        reward += light_contact_penalty  # -5 penalty, continue
        # Push vehicle back slightly
    else:
        reward += collision_penalty      # -20, done=True
        done = True
```

### S9. Remove Alignment Reward (`racer_env.py`)
```
# REMOVE these lines from reward function:
# reward += alignment * alignment_bonus        # DELETE
# reward += alignment * forward_speed_weight    # KEEP but remove alignment multiplier

# REPLACE with distance progress:
distance_delta = position.distance_to(prev_position)
forward_progress = dot(delta_vector, heading_vector)
reward += distance_progress_weight * forward_progress  # Reward moving forward
```

### S10. Async Sensor Delays (`racer_env.py`)
```
# Per-sensor observation delays (at 60 FPS):
lidar_delay_frames = random.randint(7, 8)    # 117-133ms (8 Hz LiDAR)
vesc_delay_frames = random.randint(1, 3)     # 17-50ms
imu_delay_frames = random.randint(0, 1)      # 0-17ms

# Store sensor history, assemble obs from different timestamps:
obs_lidar = lidar_history[-lidar_delay_frames]
obs_speed = speed_history[-vesc_delay_frames]
obs_imu = imu_history[-imu_delay_frames]
```

### S11. Yaw Inertia (`vehicle.py`)
```
# Low-pass filter on yaw rate changes
yaw_inertia_tau = 0.100  # 100ms time constant

target_yaw_rate = (speed / wheelbase) * tan(steer_angle)
yaw_rate += (target_yaw_rate - yaw_rate) * (dt / yaw_inertia_tau)
```

### S12. Wind / Slope (`vehicle.py` or `racer_env.py`)
```
# Random wind gusts (lateral force, 0.5 Hz)
wind_force = np.random.normal(0, 0.5) * sin(2 * pi * t * 0.5)
lateral_accel += wind_force / vehicle_mass

# Random slope (per-episode, small)
slope_angle = np.random.uniform(-0.035, 0.035)  # ±2 degrees in radians
gravity_component = 9.81 * sin(slope_angle)
longitudinal_accel += gravity_component
```

### S13. Continuous DR (`racer_env.py`)
```
# Step-level physics variation (battery sag)
time_in_episode = step_count * dt
battery_factor = 1.0 - 0.05 * (time_in_episode / max_episode_time)  # 5% drop over episode
available_accel = base_accel * battery_factor

# Step-level friction variation (tire temperature)
friction_drift = 0.02 * sin(2 * pi * time_in_episode / 30.0)  # 30s cycle
current_friction = base_friction + friction_drift
```

### S14. Sensor Thermal Drift (`racer_env.py`)
```
# Ornstein-Uhlenbeck process for LiDAR bias drift
theta = 0.15      # mean-reversion speed
sigma = 0.005     # volatility
mu = 0.0          # long-term mean

# Per step:
lidar_bias += theta * (mu - lidar_bias) * dt + sigma * sqrt(dt) * np.random.normal()
# Apply to all lidar readings:
lidar_readings += lidar_bias
```

---

## Validation Checklist (post-implementation)

After implementing all changes, verify on REAL vehicle:

- [ ] **S1** Servo responds with ~50ms lag (scope test PWM vs servo position)
- [ ] **S2** Agent holds actions for ~125ms (matches 8Hz LiDAR)
- [ ] **S3** Total latency matches real pipeline (~80-100ms)
- [ ] **S4** Sim turning radius matches real at 5+ m/s (measure both)
- [ ] **S5** Braking distance at max speed is realistic
- [ ] **S6** LiDAR readings at max range are appropriately noisy
- [ ] **S7** Speed reading matches VESC telemetry noise level
- [ ] **S8** Light wall touches don't crash training
- [ ] **S9** Agent discovers track direction without oracle
- [ ] **S10** Agent handles stale LiDAR data gracefully
- [ ] **S11** Yaw response matches real vehicle inertia
- [ ] **S12** Agent handles lateral disturbances
- [ ] **S13** Agent maintains performance over long episodes
- [ ] **S14** Agent handles slow sensor drift

## Training Notes

- **S7 changes obs_dim**: 30 → 32 per frame, 120 → 128 stacked
- **Must train from scratch** after S7 (checkpoint incompatible)
- **Config files to update**: game.yaml, config_sac_*.yaml (state_dim, obs_dim)
- **sac_driver/ must be updated**: state_builder.py, lidar_converter.py (for real vehicle inference)
- **Recommended training order**: S1-S3 first (config/vehicle only), validate, then S4-S7, then S8-S14
