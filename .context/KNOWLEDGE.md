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
- **Configurable LiDAR** (2026-03-05): `build_lidar_angles(front_step, rear_step)` generates full 360° coverage
  - Default high-res: 0.5° front (361 rays) + 2° rear (89 rays) = **450 rays** + 5 state = **455-dim per frame**
  - Legacy: 27 rays (210° arc, dense front ±45°, sparse sides) — used when no lidar config in sim_cfg
  - Config: `lidar_front_step_deg` and `lidar_rear_step_deg` in YAML/CLI
- Frame stacking x4 = **1820-dim state** input (high-res) or 128-dim (legacy)
- IMU channels (linear_accel, angular_vel) always present in obs (S7 addition)
- obs_dim computed dynamically in train_ssac.py from `build_lidar_angles()` output length

### Action Space
- Continuous 2D: [steer, accel] normalized to [-1, 1]
- Mapped through accel_bins to physical acceleration values
- **action_repeat=8**: policy queried every 8 sim frames (8Hz at 60fps). Actor loop checks `env.wants_new_action` to skip redundant policy queries during repeat window.

### Networks
- GaussianPolicy MLP — configurable `hidden_sizes` in YAML (current: [512, 512, 256] for 450-ray, was [256, 256, 128] for 27-ray)
- log_std output (NO LayerNorm — tanh already bounds output, LN could interfere with entropy auto-tuning)
- Twin Q-networks (critic1, critic2) with **LayerNorm** after each hidden Linear layer: Linear→LayerNorm→ReLU. Prevents Q-value divergence on diverse map pools. Added 2026-02-24.
- Target networks with soft update (tau=0.005). LayerNorm weight/bias included in `parameters()` — soft update handles them correctly. LayerNorm has NO running stats (unlike BatchNorm).
- torch.compile with mode='default' (NOT 'reduce-overhead' — causes CUDA graph deadlocks)
- Checkpoints strip `_orig_mod.` prefix from torch.compile keys
- **Checkpoint incompatibility**: Adding/removing LayerNorm changes state_dict keys. Must use `--no-resume` after architecture changes. The `_adapt_keys()` method only handles `_orig_mod.` prefix, NOT structural key changes.

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
  - **CSV logging**: `LogBuffer` class (P1) — background thread writes CSV rows via unbounded `queue.Queue`, `_write_csv_row()` enqueues non-blocking. `shutdown()` called in finally block.
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
  - `map_ids`: int32 array tracking which map each transition came from (-1 = unknown)
  - `_stratified_indices()`: generates batch indices with equal samples per map. Uses cached map→indices dict, rebuilt every 1000 adds or on buffer wrap. Falls back to uniform if <2 maps.
  - Cache: `_cached_map_indices` dict rebuilt via `_build_map_index_cache()` (np.unique + np.where per map). Invalidated every 1000 `add()` calls or when buffer is full (wrapping). ~363μs/sample with 40 maps, 2M buffer.
  - `_rng = np.random.default_rng()` for sampling (P3) — faster than `np.random.randint`
  - `sample()` returns random batch via `_rng.integers()` (NO `out=` parameter — see Conventions). `stratified=True` routes to `_stratified_indices()`.
  - `sample_into()` writes directly into pre-allocated numpy views of pinned memory via `np.take(..., out=)` (P11). Also supports `stratified=True`.
  - `_critic_params` cached list for `clip_grad_norm_` (P8) — avoid rebuilding every `learn()` call
  - Fused in-place soft update: `t.data.mul_(1-tau).add_(s.data, alpha=tau)` (P2) — no temporary tensor
  - **Pinned memory pool** (P11): pre-allocated pinned tensors + numpy views in `__init__()`, `learn()` fills via `sample_into()` then async `.to(device, non_blocking=True)`. Eliminates 5x `cudaHostRegister` per learn call.
  - torch.compile mode='default'
  - Alpha always auto-tuned (dead fixed_alpha code removed, W6 fix)
  - save_checkpoint strips `_orig_mod.` prefix (W7 fix)

### Environment
- `src/racer_env.py` — RacerEnv: physics step, LiDAR raycasting, reward, observation building, all S6-S14 features
  - `build_lidar_angles(front_step, rear_step)`: public function for 360° ray generation, used by both RacerEnv and train_ssac.py
  - `_build_legacy_lidar_angles()`: old 27-ray layout, used when no lidar config in sim_cfg
  - `wants_new_action` property: True when action_repeat window expired
  - `_build_observation()`: builds (n_lidar + 5)-dim obs every sim frame
  - `_compute_reward()`: S9 branches on distance_progress_enabled. Left/right/front groups filtered to forward hemisphere (0°-180°) to avoid rear rays creating balance asymmetry with 360° LiDAR.
  - `step()`: full sim step including S8 soft collision, S13 continuous DR, S12 wind/slope
  - S10 sensor delays: per-episode base with ±1 frame jitter per step (W1 fix)
  - **Multi-zone spawn** (2026-03-02): `_spawn_by_zone` / `_lookat_by_zone` dicts (zone_id→positions). `_random_spawn()` picks random zone, uses zone-specific spawn positions + lookat targets. Pre-filters wall clearance per zone. Falls back to legacy path if no zones defined.

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

### Map Zones
- `src/map_loader.py` — `decode_zone_channel()` / `encode_zone_channel()` for PNG RGBA multi-zone encoding
  - G channel (spawn): zone 1=255, zone 2=170, zone 3=85, none=0
  - B channel (lookat): same encoding
  - R channel (kill): unchanged binary 255/0
  - A channel (raceline): unchanged binary 255/0
  - Decode thresholds (midpoints): >=213→1, >=128→2, >=43→3, else→0
  - `MapData` has `spawn_zones` / `lookat_zones` (uint8) + boolean `spawn_mask` / `lookat_mask` (derived)
  - Backward compatible: old 0/255 zone files decode as zone 1
- `tools/map_zone_painter.py` — Tkinter painter with zone_id selector (1-3), per-zone colors
  - spawn_layer/lookat_layer are uint8 (0=none, 1-3=zone), kill/raceline remain bool
  - Per-zone spawn colors: green (1), cyan (2), teal (3)
  - Per-zone lookat colors: yellow (1), orange (2), pink (3)
  - **Eraser mode** (2026-03-05): erases from ALL layers at once, undo restores all layers

### Infrastructure
- `src/vec_env.py` — VecEnv with sequential recv: `[remote.recv() for remote in self.remotes]`
- `sac_driver/` — Real vehicle inference
  - `state_builder.py`: always uses 5 extra dims (B6 fix), use_imu flag only controls noise not dim count
  - `policy_loader.py`: loads GaussianPolicy for deployment

### Config
- `config/game.yaml` — Base config with sim_randomization section (S6-S14 feature toggles live here)
- `config/config_sac_13.yaml` — Reference config (Session 53 baseline, batch=256)
- `config/config_sac_20.yaml` — Current primary config (batch=256, learn_after=5000, memory_size=1M, alpha_min=0.03, alpha_max=0.3, map_switch_every=15, stratified_sampling=false, target_entropy=-1.5, 40 maps, lidar_front_step=0.5, lidar_rear_step=2.0, hidden_sizes=[512,512,256], num_actors=32)
- `config/config_sac_8/10/11/12.yaml` — Historical configs
- `config/physics.yaml` — Vehicle physics parameters (max_speed=4.0 m/s, was 8.0)

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
- `--lidar-front-step-deg F` — front hemisphere LiDAR step (0.5° = 361 rays)
- `--lidar-rear-step-deg F` — rear hemisphere LiDAR step (2.0° = 89 rays)
- `--stratified-sampling` / `--no-stratified-sampling` — toggle stratified replay (BooleanOptionalAction)
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
- Sweet spot: **64 actors** for 27-ray, **32 actors** for 450-ray (RAM constraint). ~128-160 max before context-switching overhead.
- Per-actor memory: ~200-440MB (map + physics + GaussianPolicy on CPU, varies by lidar ray count)
- After replay buffer fills: CPU ~30%, GPU ~30%. Bottleneck is single-threaded main process.
- **450-ray RAM budget** (32 actors): buffer 14.6 GB + actors ~12.8 GB + queue ~0.5 GB = ~28 GB of 46 GB.

## Conventions

### Config priority
CLI args > config_sac_*.yaml flat keys > game.yaml sim_randomization > defaults in racer_env.py

### sim_cfg structure
game.yaml has nested `sim_randomization:` section. S6-S14 features (lidar_sim, imu, soft_collision, distance_progress, sensor_delay, wind_slope, continuous_dr, thermal_drift) are ONLY configurable here — no CLI arg equivalents. The flat `dr_*` keys in config_sac_*.yaml map to the older DR features (physics, surface, obs_noise, control, dt_jitter, action_noise, perturb, obs_delay).

### obs_dim
**Dynamic** since 2026-03-05. train_ssac.py computes `(num_lidar_rays + 5) * stack_frames` from `build_lidar_angles()`. RacerEnv reads lidar config from `sim_cfg["lidar"]` and builds matching angles. Legacy train.py and game.py still use `LIDAR_ANGLES_DEG` (27 rays, 128-dim stacked). When using config_sac_20.yaml: 450+5=455 per frame, ×4 = 1820-dim.

### torch.compile
MUST use mode='default'. mode='reduce-overhead' uses CUDA graphs that deadlock with pin_memory() + non_blocking transfers. Checkpoint keys are stripped of `_orig_mod.` prefix in save_checkpoint.

### Scaling actors
More actors does NOT increase throughput past CPU saturation. With 16 threads, ~128-160 actors maximizes throughput. Beyond that, actors block on queue.put() (queue full) and context-switch overhead kills per-actor speed. Adding RAM only helps for bigger replay buffer, NOT more actors.

### Main loop and UTD
**CRITICAL**: The main loop must maintain UTD (Updates-to-Data ratio) ≈ 1.0. This is achieved by:
1. Config-driven queue size → actors block on `put()` when full (backpressure)
2. Proportional learning: N `learn()` calls per N transitions drained
3. Direct queue access — no intermediate buffer/drainer

### np.random.randint AND np.random.Generator.integers — NO `out=`
**NEITHER** supports `out=` parameter:
- `np.random.randint()` — no `out=`
- `np.random.default_rng().integers()` — also **no `out=`** despite some docs suggesting otherwise
- This caused a `TypeError` crash on first `learn()` call, which looked like a hang (64 actor join timeouts in finally block = ~64s silence)
- **Solution**: call `integers()` without `out=`, let numpy allocate the index array. Still faster than `np.random.randint`.

### LogBuffer queue must be unbounded
`LogBuffer` in train_ssac.py uses `queue.Queue()` (unbounded). A bounded queue (e.g., maxsize=256) causes deadlock: during checkpoint saves, actors keep producing episodes, `_drain_stats()` tries to write 300+ CSV rows, the queue fills, `put()` blocks, main loop freezes, actors block on transition_queue → complete deadlock. The writer thread must also have try/except to prevent silent crashes.

### Action repeat + delay_steps
action_repeat models DECISION FREQUENCY (8Hz LiDAR rate). delay_steps models PIPELINE DELAY (inference time only). vehicle.py models ACTUATOR LAG (servo/motor response). These are three SEPARATE physical phenomena — do not double-count. Actor loop only queries policy when `env.wants_new_action` is True.

### Sensor delays
Per-episode base delay sampled in reset(), with ±1 frame jitter per step. This models real hardware where delay is determined by wiring/buffering and stays approximately constant within a run.

### LayerNorm in critics — design rationale
QNetwork uses `Linear→LayerNorm→ReLU` (pre-activation norm) in hidden layers. Output layer has NO LayerNorm — Q-values must be unbounded. GaussianPolicy does NOT use LayerNorm because: (1) tanh already bounds output, (2) LN would change feature scale before mean/log_std heads, potentially interacting with entropy auto-tuning, (3) policy is synced to 64 actors — architecture mismatch risk. The collapse problem was Q-value divergence, not policy instability. Fixing critics fixes the gradient source.

### 40-map training stability
**Two distinct collapse mechanisms identified (2026-02-25):**

**Type 1 — Q-divergence (sudden, Mapa_1_1):** Without LayerNorm, Q-loss explodes 5x in ~200 episodes, alpha crashes, performance drops instantly from 172m to 43m. Fixed by: (1) LayerNorm in critics, (2) alpha_min≥0.03, (3) memory_size≤2M.

**Type 2 — Catastrophic forgetting (gradual, Mapa_1_2):** With LayerNorm fixes, training peaks at 307m then slowly decays to 119m over 13K episodes. Q-loss stable, alpha RISING (0.15→0.23, policy losing plasticity), entropy frozen. Root causes: (a) 2M buffer evicts map data too fast with map_switch=50, (b) rising alpha dilutes Q-value signal in policy gradient. Fixed by: (1) alpha_max=0.3 (caps entropy domination), (2) map_switch_every=15 (3.3x more map diversity in buffer).

**Type 3 — Curriculum destruction (plateau, Mapa_1_4):** Stratified sampling forced equal representation of all maps including earliest (worst) experiences in every batch. Destroyed the natural curriculum where the buffer fills with progressively better experiences. Result: peak only 131m (vs 284m in 1_3), plateau at 60m for 20K+ episodes. The WORST 40-map run. Fixed by: reverting stratified_sampling=false.

**Current recipe for 40 maps:** LayerNorm in critics + alpha_min=0.03 + alpha_max=0.3 + memory=1M + map_switch_every=15 + stratified_sampling=**false** + target_entropy=-1.5. Uniform sampling preserves natural curriculum. target_entropy=-1.5 (vs default -2.0) gives wider action distributions for multi-map coverage.

### Alpha bounds (alpha_min / alpha_max)
`alpha_min` prevents entropy collapse (floor). `alpha_max` prevents entropy from dominating policy gradient (ceiling). Both use `log_alpha.data.clamp_()` after the alpha optimizer step. Value of 0.0 = disabled. Added 2026-02-25.

### Config key validation
`_validate_config_keys()` in train_ssac.py checks ALL YAML keys against argparse destinations. Any new config parameter MUST have a corresponding `parser.add_argument()` or the training script exits with "Unknown config keys". Use `is not None` pattern (NOT `or`) when reading args — `0.0` is falsy in Python.

### Stratified replay sampling
Actor workers tag each transition with `map_id` (integer index into `map_pool`). `_map_name_to_id` dict built once at actor start. Main loop unpacks 6-element tuples `(state, action, reward, next_state, done, map_id)` and passes `map_id` to `memory.add()`. `learn(stratified=True)` routes to `_stratified_indices()` which divides batch evenly across all maps. Cache is rebuilt every 1000 adds. `map_ids_list` is shuffled before remainder allocation to prevent systematic bias toward low map IDs. `to_list()`/`load_list()` do NOT include map_ids (memory serialization is disabled anyway). `step()` API also accepts `map_id` for backward compatibility.

### target_entropy for multi-task SAC
Standard SAC uses `target_entropy = -dim(A) = -2.0` for 2D actions. This is calibrated for SINGLE-TASK. For 40-map multi-task, the policy needs wider action distributions to cover diverse track geometries. Entropy frozen at -2.0 in ALL runs (1_1 through 1_4) — this is the auto-tuner correctly reaching the wrong target.

| target_entropy | std/dim | Use case |
|---|---|---|
| -2.0 | 0.18 | Single-task (too narrow for 40 maps) |
| -1.5 | 0.22 | Conservative multi-task (current Mapa_1_5 setting) |
| -1.0 | 0.30 | Aggressive multi-task (try if -1.5 insufficient) |
| -0.5 | 0.39 | Too exploratory — slow convergence risk |

Implementation: `_parse_target_entropy()` in train_ssac.py:127 accepts float or "auto". Set via YAML `target_entropy: -1.5` or CLI `--target-entropy -1.5`.

### Network capacity for 40 maps
With 450-ray LiDAR and [512,512,256]: ~6.65M total params (policy ~1.33M, 2x critic ~1.33M each, 2x target ~1.33M each). Previous 27-ray [256,256,128]: ~662K total. Multi-task RL literature (MT-SAC, Meta-World) uses [400,400] for 10-50 tasks. With 450-ray input, first layer alone is 455×512 = 233K params — the capacity increase is proportional to the information increase.

### Pinned memory numpy views (P11)
`self._pin_rewards` shape `(bs, 1)` → `squeeze(1).numpy()` → numpy view `(bs,)`. The intermediate squeezed tensor can be GC'd safely because the storage is kept alive by `self._pin_rewards`. `np.take(..., out=np_view)` writes directly into pinned memory. `.to(device, non_blocking=True)` then does DMA from pinned buffer. `q_loss.item()` at end of learn() syncs CUDA, ensuring the H2D copy completes before next iteration overwrites the buffer.

### Configurable LiDAR architecture (2026-03-05)
`build_lidar_angles(front_step_deg, rear_step_deg)` in racer_env.py generates full 360° coverage:
- Front hemisphere (0°-180°): `front_step_deg` spacing → e.g. 0.5° = 361 rays
- Rear hemisphere (180°-360°): `rear_step_deg` spacing → e.g. 2° = 89 rays
- Boundary handling: 180° included in front, rear starts at 180+step, ends at 360-step (no overlap at 0°/360°)
- Config injection: train_ssac.py sets `sim_cfg["lidar"]["front_step_deg"]` only when CLI/YAML provides values. Otherwise RacerEnv falls back to legacy `LIDAR_ANGLES_DEG` (27 rays).
- Reward function: `_compute_reward()` filters rear rays (>180°) from left/right/front groups to maintain symmetric balance penalty. `min_clear` still uses all 450 rays (including rear) — low severity, 0.04 coefficient.
- Checkpoint stores lidar config inside `sim_cfg` in config_snapshot metadata.

Last updated: 2026-03-05
