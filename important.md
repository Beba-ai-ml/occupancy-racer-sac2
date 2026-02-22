# SAC v2 — Performance & Sim-to-Real Analysis

> Generated: 2026-02-15

---

## PART 1: PERFORMANCE OPTIMIZATIONS

### P1. [CRITICAL] CSV I/O Blocks Training Loop
- **File:** `src/train.py:289-343`
- **Problem:** Synchronous file write with `flush()` called every episode completion
- **Cost:** 5-50ms per episode (disk-dependent)
- **Fix:** Background thread with buffered queue:
  ```python
  import queue, threading

  class LogBuffer:
      def __init__(self, filepath, flush_interval=10):
          self.queue = queue.Queue()
          self.filepath = filepath
          self.thread = threading.Thread(target=self._writer_loop, daemon=True)
          self.thread.start()

      def log(self, row_dict):
          self.queue.put(row_dict)

      def _writer_loop(self):
          with open(self.filepath, "a", newline="") as f:
              writer = None
              count = 0
              while True:
                  row = self.queue.get()
                  if row is None: break
                  if writer is None:
                      writer = csv.DictWriter(f, fieldnames=row.keys())
                      writer.writeheader()
                  writer.writerow(row)
                  count += 1
                  if count % 10 == 0:
                      f.flush()
  ```

### P2. [CRITICAL] 5 Separate GPU Transfers Per Step
- **File:** `src/rl_agent.py:314-327`
- **Problem:** 5 individual `.pin_memory().to(device, non_blocking=True)` calls per gradient step
- **Cost:** ~2.5ms per step (5 transfers x ~500us each)
- **Fix:** Batch all conversions, pre-allocate pinned buffers:
  ```python
  # Convert all at once
  tensors = {k: torch.from_numpy(v).pin_memory() for k, v in batch.items()}
  tensors = {k: v.to(self.device, non_blocking=True) for k, v in tensors.items()}
  ```

### P3. [HIGH] Replay Buffer: Expensive Numpy Indexing
- **File:** `src/rl_agent.py:92-129`
- **Problem:** `np.random.randint()` + advanced indexing creates copies every sample call
- **Cost:** 5 array copies x batch_size=1024 per step
- **Fix:** Pre-allocate index buffer:
  ```python
  self._idx_buf = np.empty(batch_size, dtype=np.int32)
  # In sample():
  np.random.randint(0, self.size, size=batch_size, out=self._idx_buf)
  ```

### P4. [HIGH] VecEnv: Sequential Send/Recv
- **File:** `src/vec_env.py:158-172`
- **Problem:** Pipes send/recv sequentially for each environment
- **Cost:** Latency = N_envs x (step + IPC overhead) instead of max(step times)
- **Fix:** Use `select.select()` for parallel recv:
  ```python
  # Send all at once
  for remote, action in zip(self.remotes, actions):
      remote.send(("step", action))
  # Recv in any-ready order
  remaining = set(range(len(self.remotes)))
  while remaining:
      ready = select.select(self.remotes, [], [])[0]
      for remote in ready:
          idx = self.remotes.index(remote)
          if idx in remaining:
              results[idx] = remote.recv()
              remaining.discard(idx)
  ```

### P5. [MEDIUM] Action Selection: Unnecessary List Conversions
- **File:** `src/rl_agent.py:282-294`
- **Problem:** `np.array(states)` copies list of arrays, then list comprehension on output
- **Fix:** Use `np.stack(states)`, return ndarray directly instead of list

### P6. [MEDIUM] Observation Building: Python Loop + Useless Rounding
- **File:** `src/racer_env.py:1115-1150`
- **Problem:** List append loop -> array -> `np.round(obs, 3)` (rounding is pointless for neural net)
- **Fix:** Pre-allocate `np.empty(obs_size)`, vectorize ops, remove `np.round`:
  ```python
  obs = np.empty(obs_size, dtype=np.float32)
  distances = np.array([d for _, d, _ in readings], dtype=np.float32)
  obs[:n_rays] = np.clip(distances / max_range, 0.0, 1.0)
  obs[-3] = float(collision)
  obs[-2] = speed_norm
  obs[-1] = servo_norm
  ```

### P7. [MEDIUM] Lidar Ray-Obstacle: O(rays x obstacles)
- **File:** `src/racer_env.py:1030-1043`, `711-728`
- **Problem:** 27 rays x N obstacles AABB checks every frame per env
- **Cost:** Scales badly with obstacle count (36K checks/sec at 4 envs, 5 obstacles)
- **Fix:** Early exit on close hit + spatial partitioning (quadtree) for N > 5

### P8. [LOW] Grad Clip: Parameter List Rebuilt Every Step
- **File:** `src/rl_agent.py:345-361`
- **Problem:** `list(critic1.parameters()) + list(critic2.parameters())` every backward pass
- **Fix:** Cache in `__init__`:
  ```python
  self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
  ```

### P9. [LOW] torch.compile Already Implemented
- **File:** `src/rl_agent.py:222-233`
- **Status:** Present but verify it's actually activating (mode='default')
- **Potential:** 10-30% speedup on forward/backward passes if working

**Estimated total speedup from all fixes: 2-4x faster training**

---

## PART 2: SIM-TO-REAL GAPS

### S1. [CRITICAL] No Servo/Motor Lag
- **File:** `src/vehicle.py:182-240`
- **Sim:** Steering and acceleration applied instantly — zero actuator delay
- **Real:** Servo response: 40-60ms, ESC/motor response: 50-100ms
- **Impact:** Agent learns micro-corrections impossible on real hardware
- **Fix:** First-order low-pass filter:
  ```python
  # In vehicle.update():
  servo_tau = 0.05   # 50ms
  servo_actual += (servo_target - servo_actual) * (dt / servo_tau)
  accel_tau = 0.075  # 75ms
  accel_actual += (accel_cmd - accel_actual) * (dt / accel_tau)
  ```

### S2. [CRITICAL] Control Frequency 6x Too Fast
- **Config:** `fps: 60, action_repeat: 1` → agent acts every 16.7ms
- **Real:** LiDAR at 10Hz → real control at ~100ms intervals
- **Impact:** Agent doesn't learn to hold actions; overfits to 16.7ms cadence
- **Fix:**
  ```yaml
  action_repeat: 6   # hold action 100ms (matches real 10Hz LiDAR)
  # OR
  fps: 20             # reduce simulation frequency
  ```

### S3. [CRITICAL] Action-to-Effect Latency Too Low
- **Config:** `dr_delay_steps: [1, 3]` → 17-50ms delay
- **Real:** Total closed-loop latency: 100-150ms (LiDAR scan + inference + servo + motor)
- **Fix:**
  ```yaml
  dr_delay_steps: [3, 5]  # 50-83ms at 60 FPS
  # Better with action_repeat: [4, 8] steps total latency
  ```

### S4. [HIGH] No Tire Slip Model
- **File:** `src/vehicle.py` — kinematic bicycle model
- **Sim:** Perfect traction at all speeds and steer angles
- **Real:** Tires slip (Pacejka model), understeer at high speed, oversteer on hard braking
- **Impact:** Agent turns at speeds that cause real vehicle to slide out
- **Fix:** Add simplified Pacejka slip:
  ```python
  slip_angle = steer_angle - atan2(lateral_vel, forward_vel)
  if abs(slip_angle) > critical_angle:
      yaw_rate *= grip_reduction_factor  # e.g., 0.6-0.8
  ```

### S5. [HIGH] Linear Drag Instead of Quadratic
- **File:** `src/vehicle.py:204-212`
- **Sim:** `drag_force = drag_coeff * speed` (linear)
- **Real:** `drag_force = 0.5 * rho * Cd * A * v^2` (quadratic — much stronger at high speed)
- **Impact:** Agent misjudges braking distance at high speeds
- **Fix:** `drag_force = drag_coeff * speed * abs(speed)`

### S6. [HIGH] LiDAR: No Beam Divergence or Ego-Motion Blur
- **File:** `src/racer_env.py:957-1028`
- **Sim:** Infinitely thin rays, instantaneous scan, perfect geometry
- **Real:** Beam divergence ±0.3-1deg, ego-motion distorts scan during high-speed turns, 30ms scan time
- **Impact:** Agent relies on precision that doesn't exist
- **Fix:**
  ```python
  # Range-dependent noise (further = less precise)
  beam_noise = lidar_noise_std * (distance / max_range)
  # Ego-motion blur
  yaw_distortion = yaw_rate * scan_time / 2.0
  ray_angle += yaw_distortion * ray_index / total_rays
  ```

### S7. [HIGH] Perfect State Observation (Speed, Servo)
- **File:** `src/racer_env.py:1146-1150`
- **Sim:** Exact vehicle speed and servo angle — zero noise, zero latency
- **Real:** Wheel encoders: +-3-5% noise + drift. Servo potentiometer: coarse resolution. No IMU in obs.
- **Impact:** Agent expects perfect state information unavailable on real hardware
- **Fix:**
  - Replace perfect speed with encoder estimate + noise
  - Add IMU (linear accel + angular velocity) to obs space: 30-dim → 32-dim per frame
  - Add encoder quantization noise

### S8. [MEDIUM] Binary Collision = Instant Episode End
- **File:** `src/racer_env.py:1255, 1262`
- **Sim:** Any wall contact → done=True, reward=-20
- **Real:** Light scrapes don't stop vehicle; only hard impacts do
- **Fix:** Allow 2-3 light contacts with penalty before termination:
  ```python
  contact_count += 1
  if contact_count < 3:
      reward -= 5.0   # penalty but continue
  else:
      done = True      # hard stop
  ```

### S9. [MEDIUM] Alignment Reward Uses Oracle Track Knowledge
- **File:** `src/racer_env.py:1152-1196`
- **Sim:** Reward includes alignment to pre-computed track center direction
- **Real:** Agent doesn't know track center — must discover it from LiDAR
- **Fix:** Replace alignment bonus with pure distance-progress reward

### S10. [MEDIUM] Synchronized Observations
- **Sim:** All sensors read at same instant (60 Hz)
- **Real:** LiDAR 10Hz, IMU 100Hz, encoders 50Hz — each with independent latency
- **Fix:** Add per-sensor observation delays (LiDAR: 5-7 frames, encoder: 2-3 frames)

### S11. [MEDIUM] No Yaw Inertia
- **File:** `src/vehicle.py`
- **Sim:** Yaw rate changes instantly with steering
- **Real:** Rotational inertia limits yaw acceleration; can't snap turns
- **Fix:** Low-pass filter yaw rate changes:
  ```python
  yaw_rate += (target_yaw_rate - yaw_rate) * (dt / 0.1)  # 100ms time constant
  ```

### S12. [LOW] No Wind / Slope / Terrain
- **Sim:** Flat, uniform surface, no external forces
- **Real:** Wind gusts, surface slopes, uneven terrain affect dynamics
- **Fix:** Random lateral force + gravity slope component

### S13. [LOW] DR is Per-Episode, Not Continuous
- **Sim:** Physics params randomized once at reset, constant during episode
- **Real:** Battery voltage drops over time, tires heat up, conditions change continuously
- **Fix:** Add step-level variation (battery sag, gradual friction changes)

### S14. [LOW] No Sensor Thermal Drift
- **Sim:** Sensor noise is white (uncorrelated frame-to-frame)
- **Real:** Sensor bias drifts with temperature over minutes
- **Fix:** Add low-frequency correlated noise (Ornstein-Uhlenbeck process)

---

## ESTIMATED SIM-TO-REAL TRANSFER

| Scenario | Estimated Success Rate |
|----------|----------------------|
| Current config (as-is) | 40-60% |
| After S1-S3 (actuator lag + control freq) | 60-75% |
| After S1-S7 (all critical + high) | 70-85% |
| After all mitigations | 85-95% |

---

## PRIORITY ACTION LIST

### Performance (training speed):
1. `[CRITICAL]` P1 — Background CSV writer
2. `[CRITICAL]` P2 — Batch GPU transfers
3. `[HIGH]` P3 — Pre-allocated replay buffer indexing
4. `[HIGH]` P4 — Parallel VecEnv recv
5. `[MEDIUM]` P5+P6 — Vectorize action selection + observation building

### Sim-to-Real (transfer quality):
1. `[CRITICAL]` S1 — Servo/motor lag (first-order filter)
2. `[CRITICAL]` S2 — action_repeat: 6 (10Hz control)
3. `[CRITICAL]` S3 — Increase dr_delay_steps: [3, 5]
4. `[HIGH]` S4 — Tire slip model
5. `[HIGH]` S5 — Quadratic drag
6. `[HIGH]` S6 — LiDAR beam divergence + ego-motion
7. `[HIGH]` S7 — Encoder noise + IMU in obs
8. `[MEDIUM]` S8-S11 — Collision, alignment, sync, yaw inertia
