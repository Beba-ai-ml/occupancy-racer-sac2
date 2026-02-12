# Runtime Configuration Guide

Runtime configuration for the trained SAC model on f1tenth.

---

## Table of Contents

1. [Parameter Summary](#parameter-summary)
2. [LIDAR](#1-lidar)
3. [Servo (Steering)](#2-servo-steering)
4. [Controls](#3-controls)
5. [Speed](#4-speed)
6. [Troubleshooting](#5-troubleshooting)
7. [Pre-launch Checklist](#6-pre-launch-checklist)

---

## Parameter Summary

| Parameter | Training Value | Runtime Setting |
|-----------|---------------|-----------------|
| LIDAR rays | 27 | 27 (same angles) |
| LIDAR center | 90° | `angle_offset_deg = -90.0` |
| LIDAR max range | 20m* | `max_range_m = 20.0`* |
| stack_frames | 4 | 4 |
| state_dim | 120 | 120 |
| Frequency | 30 Hz* | 30 Hz |
| servo neutral | 0.5 | see Servo section |
| steer sign | +1 = right | verify direction |
| speed sign | +1 = forward | verify direction |

*\* = values after retraining per Tasks.md*

---

## 1. LIDAR

### Training LIDAR Angles

Training uses 27 rays over a 210-degree arc:

```
Angles (degrees): -15, 0, 15, 30, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
                  95, 100, 105, 110, 115, 120, 125, 130, 135, 150, 165, 180, 195

Where: 90° = vehicle FRONT
       <90° = LEFT side
       >90° = RIGHT side
```

### Runtime Offset

Since training uses **front = 90°**, at runtime you must shift the angles:

```yaml
lidar:
  angle_offset_deg: -90.0    # shifts ROS angles so that 0° -> 90°
  angle_direction: 1.0       # 1.0 = CCW, -1.0 = CW (depends on mounting)
```

### Offset Verification Test

1. Place an obstacle **IN FRONT** of the vehicle
2. Check which rays drop (have low values)
3. **Correct:** center rays (indices ~10-16) drop
4. **Wrong:** side rays drop -> incorrect offset

```
Visualization (top-down view):

         OBSTACLE
              v
        [ ======= ]

    <------- 90° -------->  (center rays)
   /                       \
  /                         \
 /                           \
45°                         135°
```

### Range Normalization

```python
# At runtime:
lidar_normalized = min(distance_m / max_range_m, 1.0)

# max_range_m MUST match the training value (default 20.0)
```

---

## 2. Servo (Steering)

### How Training Normalizes Servo

```python
# In training:
steer in [-1, 1]                    # action from policy
servo_value = steer * 10.0 + 10.0   # transform to [0, 20]
servo_norm = servo_value / 20.0      # normalize to [0, 1]
```

### Value Table

| Position | steer | servo_value | servo_norm |
|----------|-------|-------------|------------|
| Max left | -1.0 | 0.0 | **0.00** |
| Slight left | -0.5 | 5.0 | 0.25 |
| Neutral | 0.0 | 10.0 | **0.50** |
| Slight right | +0.5 | 15.0 | 0.75 |
| Max right | +1.0 | 20.0 | **1.00** |

### Runtime Configuration

If VESC outputs servo in range `[0.0, 1.0]`:
```yaml
state:
  servo_source: /commands/servo/position
  servo_norm_offset: 0.0
  servo_norm_divisor: 1.0
```

If VESC outputs servo in range `[0.05, 0.95]`:
```yaml
state:
  servo_norm_offset: -0.05
  servo_norm_divisor: 0.9
  # result: (servo - 0.05) / 0.9 -> [0, 1]
```

---

## 3. Controls

### Steering Sign (steer_sign)

In training: **steer > 0 = turn RIGHT**

```yaml
control:
  steer_sign: 1.0    # default
```

**Test:**
1. Force a constant action `steer = +0.5`
2. Observe whether the robot turns right
3. If it turns left -> set `steer_sign: -1.0`

### Rate Limiting

Training uses rate limiting on actions:
```yaml
control:
  steer_rate_limit: 0.15   # max steer change per step
  accel_rate_limit: 0.30   # max accel change per step
```

You can implement this at runtime for smoother control.

---

## 4. Speed

### Speed Sign (speed_sign)

In training: **speed > 0 = driving FORWARD**

```yaml
control:
  speed_sign: 1.0    # default
```

**Test:**
1. Force a constant action `accel = +1.0`
2. Observe whether the robot drives forward
3. If it drives backward -> set `speed_sign: -1.0`

### Speed Normalization in Observation

```python
# In training:
speed_norm = abs(vehicle_speed) / max_speed  # always [0, 1]
speed_norm = min(speed_norm, 1.0)            # clamp to 1.0
```

**Important:** Training uses the **absolute value** of speed, so there is no direction information in the observation.

### VESC ERPM

If `speed_to_erpm_gain` in VESC is negative:
```yaml
control:
  speed_sign: -1.0   # compensate for negative gain
```

---

## 5. Troubleshooting

### Problem: Robot drives into walls

**Possible causes:**

| Cause | How to check | Solution |
|-------|-------------|----------|
| Wrong LIDAR offset | Obstacle in front doesn't affect center rays | Fix `angle_offset_deg` |
| Inverted steer | Robot steers in opposite direction | `steer_sign: -1.0` |
| Wrong servo normalization | Servo values outside [0,1] | Fix `servo_norm_*` |
| Wrong max_range | Obstacles "disappear" too early | Match `max_range_m` |

### Problem: Robot unresponsive / delayed reaction

**Possible causes:**

| Cause | How to check | Solution |
|-------|-------------|----------|
| Too low frequency | Measure actual loop Hz | Optimize to 30 Hz |
| Observation delay | Log timestamps | Check message queue |
| Wrong speed integrator | Speed doesn't ramp up | Check implementation |

### Problem: Robot drives backward

**Solution:**
```yaml
control:
  speed_sign: -1.0
```

### Problem: Robot spins in circles

**Possible causes:**
- Inverted steer sign
- Servo stuck at one value
- LIDAR only sees one side (wrong offset)

---

## 6. Pre-launch Checklist

### Configuration

- [ ] `angle_offset_deg` set to `-90.0`
- [ ] `max_range_m` matches training value
- [ ] `stack_frames = 4`
- [ ] Loop frequency ~30 Hz
- [ ] Servo normalization yields values in [0, 1]

### Static Tests (robot stationary)

- [ ] Obstacle in front -> center LIDAR rays drop
- [ ] Obstacle on left -> left rays drop
- [ ] Obstacle on right -> right rays drop
- [ ] `servo_norm` = 0.5 with neutral steering

### Dynamic Tests (slow speed!)

- [ ] Command steer=+0.5 -> robot turns right
- [ ] Command steer=-0.5 -> robot turns left
- [ ] Command accel=+1.0 -> robot drives forward
- [ ] Robot avoids a simple obstacle

---

## Example YAML Configuration

```yaml
# sac_runtime_config.yaml

model:
  checkpoint: runs/session_name/session_name.pth
  state_dim: 120
  action_dim: 2

lidar:
  angles_count: 27
  angle_offset_deg: -90.0
  angle_direction: 1.0
  max_range_m: 20.0

state:
  stack_frames: 4
  max_speed_mps: 8.0
  servo_norm_offset: 0.0
  servo_norm_divisor: 1.0

control:
  steer_sign: 1.0
  speed_sign: 1.0
  steer_rate_limit: 0.15
  accel_rate_limit: 0.30
  max_steering_angle_deg: 20.0
  wheelbase_m: 0.27

topics:
  lidar: /scan
  odom: /odom
  servo: /commands/servo/position
  drive: /drive
```

---

## History

| Date | Change |
|------|--------|
| 2026-02-03 | Document created |
| 2026-02-12 | Translated to English |
