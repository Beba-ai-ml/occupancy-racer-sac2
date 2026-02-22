# SAC v2 - Knowledge Base

## Architecture

Async SAC with 64 CPU actor processes + 1 GPU learner. Actors send single transitions through a shared mp.Queue (config-driven size, natural backpressure when full). Main loop pulls transitions one at a time via `queue.get()` / `get_nowait()`, inserts into replay buffer via `memory.add()`, and runs proportional gradient updates (UTD ≈ 1.0). Weights synced to actors via per-actor `mp.Queue(maxsize=1)`.

```
[64 Actor Processes] --single transition--> [mp.Queue] --get/get_nowait--> [memory.add()] --> [Replay Buffer]
     |                     ↑ blocks when full                                                        |
     |                     (backpressure!)                                                     sample()
     |                                                                                    pin_memory()
     |                                                                                               |
     |                                                                                     [GPU Learner]
     |                                                                                  (N learns per N transitions)
     |                                                                                               |
     |<----------------------- weights_queues (mp.Queue per actor, maxsize=1) -----------------------|
```

### Observation Space
- 27 LiDAR rays + 5 state values (collision, speed, servo, linear_accel, angular_vel) = **32-dim per frame**
- Frame stacking x4 = **128-dim state** input to networks
- IMU channels (linear_accel, angular_vel) always present in obs (S7 addition)

### Action Space
- Continuous 2D: [steer, accel] normalized to [-1, 1]
- Mapped through accel_bins to physical acceleration values
- **action_repeat=8**: policy queried every 8 sim frames (8Hz at 60fps). Actor loop checks `env.wants_new_action` to skip redundant policy queries during repeat window.

### Networks
- GaussianPolicy MLP [256, 256, 128] with log_std output
- Twin Q-networks (critic1, critic2) same architecture
- Target networks with soft update (tau=0.005)
- torch.compile with mode='default' (NOT 'reduce-overhead' — causes CUDA graph deadlocks)
- Checkpoints strip `_orig_mod.` prefix from torch.compile keys

### Domain Randomization
- Per-episode: physics scales, surface friction, observation noise, control delays, action noise, perturbations, **sensor delay base values** (W1 fix)
- Per-step (S12): wind gusts via Ornstein-Uhlenbeck process (W2 fix — not i.i.d.)
- Per-step (S13): battery sag, friction cycle
- Per-step (S14): sensor thermal drift (Ornstein-Uhlenbeck)

### Sim-to-Real Pipeline
vehicle.py models physical actuator lag (servo 50ms, motor 100ms, tire slip, yaw inertia, quadratic drag). racer_env.py models sensor realism (LiDAR beam divergence, ego-motion blur, async sensor delays with per-episode base + ±1 jitter, encoder noise, IMU). Config controls decision rate (action_repeat=8 for 8Hz).

**Three separate latency phenomena — never double-count:**
1. `action_repeat=8` → decision frequency (8Hz LiDAR)
2. `delay_steps=[0,1]` → inference pipeline delay only (~15ms)
3. `vehicle.py servo_tau/motor_tau` → physical actuator response (50ms/100ms)

## Key Files

### Training
- `src/train_ssac.py` — **Primary training script** (async, 64 actors). Entry point: `python -m src.train_ssac`
  - **Main loop**: blocks on `transition_queue.get(timeout=0.1)`, drains burst up to 512 via `get_nowait()`, `memory.add()` per transition, proportional learning, weight sync via `weights_queues`
  - **Weight sync**: per-actor `mp.Queue(maxsize=1)` — drain stale then put fresh state_dict
  - **Actor loop**: checks `env.wants_new_action` before querying policy (C1 fix). Single `transition_queue.put()` per decision window.
  - **Warmup**: `base_warmup_steps=37500`, scaled by `num_actors/4`
  - **Process context**: `mp.get_context("spawn")`
  - **CSV logging**: direct file writes via `_write_csv_row()` helper
  - Stats queue bounded: `maxsize=num_actors*100` (W8 fix)
  - `_bump_nofile_limit()`: targets 65536 when hard==INFINITY (W5 fix)
- `src/train.py` — Sync training script (secondary, legacy)
  - Now passes `grad_clip` and `alpha_min` to SACAgent (B5 fix)
  - Saves `distance_history` in checkpoint meta (W11 fix)
  - Direct CSV writes (no background thread)
- `src/sim_config.py` — Builds nested sim_cfg dict from flat YAML keys + CLI args

### Agent
- `src/rl_agent.py` — SACAgent class, GaussianPolicy, twin Q-networks, ReplayBuffer, learn()
  - ReplayBuffer: numpy-based, circular buffer with `add()` per transition
  - `sample()` returns random batch, `learn()` does `pin_memory()` per call then async `non_blocking .to(device)`
  - For-loop soft update (`_soft_update`) with tau
  - torch.compile mode='default'
  - Alpha always auto-tuned (dead fixed_alpha code removed, W6 fix)
  - save_checkpoint strips `_orig_mod.` prefix (W7 fix)

### Environment
- `src/racer_env.py` — RacerEnv: physics step, LiDAR raycasting, reward, observation building, all S6-S14 features
  - `wants_new_action` property: True when action_repeat window expired
  - `_build_observation()`: builds 32-dim obs every sim frame
  - `_compute_reward()`: S9 branches on distance_progress_enabled
  - `step()`: full sim step including S8 soft collision, S13 continuous DR, S12 wind/slope
  - S10 sensor delays: per-episode base with ±1 frame jitter per step (W1 fix)

### Vehicle Physics
- `src/vehicle.py` — Ackermann bicycle model with:
  - S1: servo_tau=0.050 (50ms first-order filter)
  - S1: motor_tau=0.100 (100ms first-order filter)
  - S4: tire slip model (onset >4 m/s, min_grip=0.4 for smooth floor)
  - S5: quadratic drag (v^2 instead of linear)
  - S11: yaw inertia (tau=100ms low-pass)
  - All three states (servo_actual, accel_actual, yaw_rate) zeroed on episode reset (B1 fix)

### Interactive Mode
- `src/game.py` — Interactive game with human/RL control
  - obs_dim now +5 matching training (C2 fix)
  - _build_observation includes IMU channels
  - _reset_episode resets all state properly (B3 fix)
  - reward_clip=20.0, no np.round (B4 fix)
  - Note: reward function still uses hardcoded values, not configurable like racer_env.py

### Infrastructure
- `src/vec_env.py` — VecEnv with sequential recv: `[remote.recv() for remote in self.remotes]`
- `sac_driver/` — Real vehicle inference
  - `state_builder.py`: always uses 5 extra dims (B6 fix), use_imu flag only controls noise not dim count
  - `policy_loader.py`: loads GaussianPolicy for deployment

### Config
- `config/game.yaml` — Base config with sim_randomization section (S6-S14 feature toggles live here)
- `config/config_sac_13.yaml` — Reference config (Session 53 baseline, batch=256)
- `config/config_sac_20.yaml` — Current primary config (batch=256, learn_after=5000, memory_size=8M)
- `config/config_sac_8/10/11/12.yaml` — Historical configs
- `config/physics.yaml` — Vehicle physics parameters

### Documentation
- `hardware_profile.md` — Real F1Tenth hardware specs + calibrated sim values + validation checklist
- `important.md` — Performance (P1-P9) + sim-to-real (S1-S14) analysis

## Tech Stack
- Python 3.10+, PyTorch (CUDA), pygame (rendering/physics), numpy
- multiprocessing (async actors, spawn context), direct mp.Queue for weight sync and transitions
- Hardware target: Jetson Nano Orin 8GB, ROS2 Foxy, VESC ESC, RPLidar (8Hz)

## How to Run

```bash
cd /home/beba/occupancy_racer/Soft_Actor_Critic_2
source .venv/bin/activate
python -m src.train_ssac --config-file config/config_sac_20.yaml --session-id <NAME> --no-resume
```

Key CLI flags:
- `--config-file` — which config YAML to use
- `--session-id` — name for checkpoints/logs in runs/
- `--no-resume` — start fresh (required after env behavior changes)
- `--max-episodes N` — stop after N episodes (0 = unlimited)
- All `dr_*` and `reward_*` flat keys can be overridden via CLI

Outputs go to `runs/session_<ID>/`:
- `.csv` — episode log (distance, reward, losses, alpha, etc.)
- `.pth` — latest checkpoint
- `.pth.bak` — previous checkpoint
- `_Backup_N.pth` — periodic backups every save_every episodes

## Hardware (Training Machine)
- CPU: AMD Ryzen 7 5700X 8-Core (16 threads)
- RAM: 46GB (~35GB used during training with 64 actors)
- GPU: NVIDIA RTX 3080 10GB VRAM
- Sweet spot: **64 actors** on this CPU. ~128-160 max before context-switching overhead.
- Per-actor memory: ~440MB (map + physics + GaussianPolicy on CPU)
- After replay buffer fills: CPU ~30%, GPU ~30%. Bottleneck is single-threaded main process.

## Conventions

### Config priority
CLI args > config_sac_*.yaml flat keys > game.yaml sim_randomization > defaults in racer_env.py

### sim_cfg structure
game.yaml has nested `sim_randomization:` section. S6-S14 features (lidar_sim, imu, soft_collision, distance_progress, sensor_delay, wind_slope, continuous_dr, thermal_drift) are ONLY configurable here — no CLI arg equivalents. The flat `dr_*` keys in config_sac_*.yaml map to the older DR features (physics, surface, obs_noise, control, dt_jitter, action_noise, perturb, obs_delay).

### obs_dim
Always 32 per frame (27 lidar + 5 state). Stacked x4 = 128. Hardcoded as `(len(LIDAR_ANGLES_DEG) + 5)` in train_ssac.py, train.py, and game.py. If obs format changes, ALL THREE files must be updated.

### torch.compile
MUST use mode='default'. mode='reduce-overhead' uses CUDA graphs that deadlock with pin_memory() + non_blocking transfers. Checkpoint keys are stripped of `_orig_mod.` prefix in save_checkpoint.

### Scaling actors
More actors does NOT increase throughput past CPU saturation. With 16 threads, ~128-160 actors maximizes throughput. Beyond that, actors block on queue.put() (queue full) and context-switch overhead kills per-actor speed. Adding RAM only helps for bigger replay buffer, NOT more actors.

### Main loop and UTD
**CRITICAL**: The main loop must maintain UTD (Updates-to-Data ratio) ≈ 1.0. This is achieved by:
1. Config-driven queue size → actors block on `put()` when full (backpressure)
2. Proportional learning: N `learn()` calls per N transitions drained
3. Direct queue access — no intermediate buffer/drainer

### np.random.randint
Does NOT support `out=` parameter (unlike np.random.default_rng().integers()). This caused a silent learner crash in the past.

### Action repeat + delay_steps
action_repeat models DECISION FREQUENCY (8Hz LiDAR rate). delay_steps models PIPELINE DELAY (inference time only). vehicle.py models ACTUATOR LAG (servo/motor response). These are three SEPARATE physical phenomena — do not double-count. Actor loop only queries policy when `env.wants_new_action` is True.

### Sensor delays
Per-episode base delay sampled in reset(), with ±1 frame jitter per step. This models real hardware where delay is determined by wiring/buffering and stays approximately constant within a run.

Last updated: 2026-02-22
