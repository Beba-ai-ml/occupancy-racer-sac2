# SAC v2 - Current State

## What Works
- Async training with 32 actors on GPU — stable, no crashes (reduced from 64 for 450-ray RAM budget)
- All 14 sim-to-real features (S1-S14) implemented, configured, and enabled
- S1-S5, S11 active in vehicle.py (hardcoded defaults, always on)
- S6-S14 configured in game.yaml sim_randomization (added 2026-02-16)
- sac_driver updated for 32-dim obs + IMU (backward compatible)
- **Session 53**: best run — **24K episodes, mean(100)=250m**, config_sac_13, 10 maps, all fixes applied
- game.py interactive mode synchronized with training obs format (+5 dim)
- Replay buffer stores correct actions during action_repeat windows (C1 fix)
- **Safe perf optimizations** (2026-02-22): P1, P2, P3, P8, P11 — pure computational shortcuts, same math
- **LayerNorm in QNetwork critics** (2026-02-24): Linear→LayerNorm→ReLU, prevents Q-value divergence on 40 maps
- **Alpha bounds** (2026-02-25): alpha_min=0.03 (floor) + alpha_max=0.3 (ceiling) in rl_agent.py
- **Stratified replay sampling code** (2026-02-26): implemented and validated (10/10 PASS), but DISABLED after Mapa_1_4 failure
- **Configurable 360° LiDAR** (2026-03-05): 450 rays (0.5° front + 2° rear), config-driven, reviewed by 2 Opus critics x2 rounds

## Work in Progress
- **Mapa_3_1 ready to launch** — config_sac_20 with 450-ray LiDAR, [512,512,256] network, 32 actors, 1M buffer
- **Multi-zone spawn/lookat** (2026-03-02) — implemented, reviewed by 2 Opus critics, bugs fixed

## Recent Changes (2026-03-05)

### max_speed reduced to 4.0 m/s
`config/physics.yaml` — `max_speed: 8.0` → `4.0` (was ~29 km/h, now ~14.4 km/h). DR range: ~3.4–4.6 m/s.

### Eraser mode in map_zone_painter
`tools/map_zone_painter.py` — new "Eraser" radiobutton. LMB in eraser mode clears ALL layers (kill, spawn, lookat, raceline) at once. Undo restores all layers from pre-stroke snapshot.

### Configurable 360° LiDAR (Mapa_3_1 architecture)
4 files changed: `src/racer_env.py`, `src/train_ssac.py`, `config/config_sac_20.yaml`, `src/game.py` (import only).
- **Purpose**: Increase LiDAR from 27 rays (210° arc) to 450 rays (full 360°) for richer spatial awareness
- **Architecture**: `build_lidar_angles(front_step, rear_step)` — front hemisphere 0.5° (361 rays) + rear hemisphere 2° (89 rays)
- **Config-driven**: `lidar_front_step_deg` and `lidar_rear_step_deg` in YAML/CLI. Legacy 27-ray when not configured.
- **Network scaled**: hidden_sizes [256,256,128] → [512,512,256] (~6.65M params vs ~662K)
- **RAM adjustments**: memory_size 2M→1M, num_actors 64→32, queue_size 96K→32K, sync_every 1600→800
- **Reward fix**: left/right/front groups filtered to forward hemisphere (0°-180°) to avoid rear-ray asymmetry
- **Reviewed by 2 Opus critics x2 rounds**: all PASS in round 2, one low-severity WARN (min_clear includes rear)

## Recent Changes (2026-03-02)

### Multi-Zone Spawn/Lookat Support
3 files changed: `src/map_loader.py`, `tools/map_zone_painter.py`, `src/racer_env.py`.
- **Purpose**: Open maps with gaps confuse AI — paint multiple spawn+lookat zone pairs (matched by ID 1-3) so AI spawns in specific areas facing the right direction
- **Encoding**: PNG RGBA G/B channels encode zone IDs via intensity levels (255/170/85/0)
- **Backward compatible**: old binary 0/255 zone files decode as zone 1
- **Zone painter**: zone_id selector (1-3), per-zone overlay colors, encode/decode for save/load
- **racer_env**: per-zone spawn/lookat position dicts, per-zone wall clearance prefiltering, zone-aware `_random_spawn()` picks random zone then zone-specific positions
- **Reviewed by 2 Opus critics**: fixed `_lookat_angle_from` colocated target fallback, zone slow path clearance check, missing lookat warning, backward-compat reference stale after prefilter
- **Zone 3 color fix**: original zone 3 lookat (pink with all 1.0 multipliers) rendered as pure white on gray maps — invisible. Fixed to magenta (160, 0, 100, 1.0, 0.3, 1.0). Zone 3 spawn also made more distinct (dark green).

## Recent Changes (2026-03-01)

### Mapa_1_4 Analysis — Stratified Sampling FAILED

**Mapa_1_4 results (41,222 episodes):**
- Peak mean_100: **131.6m @ ep 19,910** (WORST of all 40-map runs)
- Final mean_100: **60.2m** (hard plateau for 20K+ episodes)
- Q-loss: stable at ~69 (no divergence)
- Alpha: dropped to 0.063 (near-deterministic policy)
- Entropy: frozen at -2.0 from ep 5K

**Root cause:** Stratified sampling forced equal representation of all 40 maps in every batch, including the earliest (worst) crash episodes. This destroyed the natural curriculum where the replay buffer gradually fills with better experiences. The agent couldn't "graduate" past its own worst history.

**Comparison:**
| Run | Peak m100 | Final m100 | Pattern |
|---|---|---|---|
| 1_1 | 181m | 49m | Q-divergence collapse |
| **1_2** | **307m** | 119m | Gradual forgetting |
| 1_3 | 284m | 210m | Slow decline |
| **1_4** | **131m** | **60m** | **Curriculum destruction (WORST)** |

### Config Changes for Mapa_1_5
1. `stratified_sampling: true` → **`false`** — reverts to uniform sampling, restores natural curriculum
2. `target_entropy: auto` → **`-1.5`** — multi-task entropy target (default -2.0 is for single-task SAC)

### Adversarial Review Process (2026-03-01)
Two Opus 4.6 critics (theory + engineering) debated 4 proposals over 3 rounds:
- **Both agreed:** revert stratified, defer PER (too complex ~300 LOC, stale priorities in async)
- **Both agreed:** target_entropy -2.0 is wrong for 40-map multi-task
- **Converged on -1.5** (theory wanted -1.0, engineering wanted -1.5 as safer compromise)
- **Deferred:** PER, per-map reward normalization, network capacity increase, map-ID embedding

## Recent Changes (2026-02-26)

### Stratified Replay Sampling — Implemented (later disabled)
Implementation across 3 files (rl_agent.py, train_ssac.py, config_sac_20.yaml). Code remains in place but config sets `stratified_sampling: false`. The feature is toggleable via CLI `--stratified-sampling` / `--no-stratified-sampling`.

## Earlier Changes
- (2026-02-25) Deep analysis of 1_1 + 1_2 collapse patterns, alpha_max=0.3, map_switch=15
- (2026-02-24) LayerNorm in critics, alpha_min=0.03, memory=2M
- (2026-02-22) Safe perf optimizations re-added (P1-P11), Phase 1/2 reverted
- (2026-02-16) Critical bug fixes C1-C2, B1-B6, W1-W8, W11, S6-S14

## Decisions

0. **Phase 1/2 reverted, then 5 safe opts re-added** — the dangerous optimizations broke UTD. P1/P2/P3/P8/P11 are safe.
1. **batch_size=256** — Session 53 (250m) used 256. 1024 was slower to climb. 512 not tried but GPU throughput concern (doubles forward/backward time).
2. **memory_size** — 8M→2M (anti Q-poisoning), then 2M→1M (450-ray RAM constraint). With 1M + 32 actors at ~240 trans/sec, buffer fills in ~70 min.
3. **learn_after=5000** — Session 54 proved faster ramp-up than 40K.
4. **8Hz control rate (action_repeat=8)** — matches real LiDAR. Non-negotiable.
5. **delay_steps=[0,1] not [4,6]** — delay_steps only models inference pipeline. Actuator lag is in vehicle.py.
6. **IMU always in obs (32-dim)** — even when imu sim_cfg not enabled, IMU channels exist.
7. **Policy only queried when wants_new_action=True** — saves GPU compute, ensures replay buffer integrity.
8. **Sensor delays per-episode with ±1 jitter** — matches real hardware.
9. **Wind as OU process** — temporally correlated, matches real sustained gusts.
10. **LayerNorm ONLY in critics, NOT policy** — policy has tanh squashing, critics have unbounded Q-values that diverge.
11. **alpha_min=0.03** — prevents entropy collapse. Session 53 never went below 0.076.
12. **map_switch_every=15 (was 50, originally 10)** — 50 caused buffer data distribution collapse. 15 gives nearly all 40 maps active simultaneously with 64 actors.
13. **No Q-target clamp for now** — LayerNorm prevents the extreme cases.
14. **No LR scheduling** — Session 53 proved fixed LR=0.0003 works.
15. **alpha_max=0.3** — prevents alpha from rising past 0.3 and diluting Q-value signal in policy gradient.
16. **stratified_sampling=false** (REVERTED from true) — stratified destroyed natural curriculum in Mapa_1_4. Forced equal sampling from all maps including earliest garbage data. Peak regressed from 284m (1_3) to 131m (1_4). Code remains but disabled.
17. **target_entropy=-1.5** (NEW) — default -2.0 (= -dim(A)) is calibrated for single-task SAC. For 40-map multi-task, policy needs wider action distributions. -1.5 gives std ~0.22/dim vs 0.18/dim. If insufficient, try -1.0 (std ~0.30). Two Opus critics converged on -1.5 as safe compromise.
18. **40 maps stays** — reducing to 20 would dodge the generalization goal. Fix the algorithm, not the problem scope.
19. **PER deferred** — ~300 LOC, breaks sample_into() pinned memory path, stale priorities with 64 async actors. Try simpler fixes first.
20. **450-ray LiDAR with 360° coverage** — front hemisphere at 0.5° (361 rays, high-res for navigation) + rear at 2° (89 rays, awareness). Total 450 rays vs original 27. Matches real RPLidar capability (~720-1000 pts/scan).
21. **Network [512,512,256] for 450-ray input** — proportional to information increase. First layer: 455×512 = 233K params (vs 32×256 = 8K). ~6.65M total params.
22. **memory_size=1M (was 2M)** — 450-ray obs are 14.2× larger per entry. 1M buffer = 14.6 GB, fits in 46 GB with 32 actors.
23. **num_actors=32 (was 64)** — RAM constraint. 32 actors × ~400 MB + 14.6 GB buffer + queue = ~28 GB total.

## Known Issues

### Open concerns
1. **game.py reward function diverges from racer_env.py** — hardcoded weights, different structure. Low priority.
2. **Duplicated utility functions** — `_parse_hidden_sizes`, `_parse_target_entropy`, etc. copy-pasted across train.py, train_ssac.py, game.py.
3. **GaussianPolicy duplicated** in rl_agent.py and sac_driver/policy_loader.py with different method names.
4. **S11 yaw_inertia_tau=0.100** — estimated, not measured on real vehicle.
5. **reward_alignment_bonus: 0.02 still in configs** — harmless but confusing.
6. **train.py (sync) still has action_repeat issue** — train_ssac.py (primary) is fixed.
7. **Missing CLI mappings for S6-S14** — can only configure via game.yaml.
8. **config_sac_13.yaml has learn_after=40000** — but Session 53 actually ran with 5000.
9. **train.py missing alpha_max parameter** — uses default=0.0, no crash but silent behavioral difference.
10. **game.py missing alpha_min, alpha_max, grad_clip** — uses defaults, no crash but inconsistent.
11. **init_alpha not clamped to [alpha_min, alpha_max] at construction** — only clamped on first learn() call. Current config init_alpha=0.3 == alpha_max=0.3 so not broken now.
12. **Reward scale heterogeneity across maps** — different maps produce different reward magnitudes (wider tracks → higher speeds → higher rewards). Could cause gradient dominance. Potential fixes: reward_clip reduction (20→10-15) or per-map normalization.
13. **Spawn-in-collision instant deaths (0.05s / 0.00m)** — `_random_spawn()` fallback path (line 1048-1060 in racer_env.py) picks a random position WITHOUT clearance guarantee when 1000 attempts fail on narrow maps. `reset()` (line 1475) calls `_vehicle_collision()` but only feeds result to observation — does NOT re-spawn if already in collision. Soft collision (max_light_contacts=2) then kills the car in exactly 3 frames: frame 1-2 soft contact (-4.0 each), frame 3 hard collision (-16.0) = reward_sum ≈ -24.09, time=0.050s, distance=0.00m. Rare but stochastic — appeared ~161 times in warmup, then sporadically after ep 15K in Mapa_1_5. Fix: add retry loop in `reset()` that re-spawns if `_vehicle_collision()` returns True (5 retries max, ~5 lines of code).

## Next Steps

### Tier 1 — Mapa_3_1 (450-ray LiDAR, new architecture)
1. **Run Mapa_3_1** — `--no-resume --config-file config/config_sac_20.yaml --session-id Mapa_3_1`
2. **Monitor** (expect slower per-episode due to 16.7× more raycasting):
   - Raycasting throughput — is it a bottleneck? Check steps/sec
   - RAM usage — should be ~28 GB of 46 GB
   - Entropy should stabilize ~-1.5
   - mean_100 target: comparable to 27-ray runs, potentially better due to richer spatial info
3. Expect slower initial climb (more params, more exploration needed)

### Tier 2 — If Mapa_3_1 is too slow
4. Numba JIT `_cast_ray` for 10-50× speedup on raycasting
5. Reduce to 1° front step (181 front + 89 rear = 270 rays) as compromise
6. Increase num_actors back if RAM permits

### Tier 3 — If performance plateaus
7. `reward_clip: 20` → `15`
8. `alpha_min: 0.03` → `0.05`
9. Per-map reward normalization

### Tier 4 — Last resort
10. Rank-based PER with IS weights
11. Map-ID embedding in critic
12. Reduce map pool to 20 (diagnostic only)

## Training Session History

| Session | Episodes | Config | batch | learn_after | Mean(100) | Notes |
|---------|----------|--------|-------|-------------|-----------|-------|
| 33 | 11,475 | sac_13 | 256 | 5K | 800m+ | Pre-sim2real best. Completes laps. |
| 39 | 8,131 | sac_13 | 256 | 5K | — | delay_steps double-counted, 20x slower |
| **53** | **24,000** | **sac_13** | **256** | **5K** | **250m** | **Pre-Phase-2 baseline. Best post-sim2real.** |
| 54 | 14,700 | sac_20 | 1024 | 5K | 112m | config_sac_20, cut short |
| diag | ? | sac_13 | 256 | 5K | ~50m | Phase 2 + debt v1. Broken UTD. |
| diag_2 | 20,000 | sac_13 | 256 | 5K | ~25m | Debt tuning. Worse. |
| diag_5 | 26,500 | sac_13 | 256 | 5K | ~56m | MIN_DEBT/MAX_BURST. UTD=0.016. |
| diag_10 | 29,750 | sac_20 | 1024 | 5K | ~56m | Simple loop. UTD ~0.02. |
| diag_11 | 21,750 | sac_20 | 1024 | 5K | ~34m | Proportional + drainer cap. Clogged. |
| diag_25 | 8,500 | sac_20 | 1024 | 40K | ~49m | Direct queue, cut short at 8.5K eps |
| **Mapa_1_1** | **46,526** | **sac_20** | **256** | **5K** | **49m** | **40 maps, 8M buf, NO LayerNorm. Peaked 181m@15812, Q-divergence collapse.** |
| **Mapa_1_2** | **31,384** | **sac_20** | **256** | **5K** | **119m** | **40 maps, 2M buf, LayerNorm, alpha_min=0.03, map_switch=50. Peaked 307m@17918, gradual decay.** |
| **Mapa_1_3** | **26K+** | **sac_20** | **256** | **5K** | **~210m** | **40 maps, +alpha_max=0.3, map_switch=15. Peaked 284m, slow decline (-26%).** |
| **Mapa_1_4** | **41,222** | **sac_20** | **256** | **5K** | **60m** | **40 maps, +stratified=true. Peaked 131m@19910, plateau 60m. WORST RUN. Curriculum destroyed.** |
| Mapa_1_5 | pending | sac_20 | 256 | 5K | ? | 40 maps, stratified=false, target_entropy=-1.5. |
| **Mapa_3_1** | **pending** | **sac_20** | **256** | **5K** | **?** | **450-ray LiDAR, [512,512,256], 32 actors, 1M buffer.** |

### Config Comparison
| Param | **13 (S53)** | **20 (1_1)** | **20 (1_2)** | **20 (1_3)** | **20 (1_4)** | **20 (1_5)** | **20 (3_1)** |
|-------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 | **256** |
| memory_size | 1.6M | 8M | 2M | 2M | 2M | 2M | **1M** |
| alpha_min | 0.005 | 0.005 | 0.03 | 0.03 | 0.03 | 0.03 | **0.03** |
| alpha_max | — | — | — | 0.3 | 0.3 | 0.3 | **0.3** |
| map_pool | 10 | 40 | 40 | 40 | 40 | 40 | **40** |
| map_switch | 10 | 10 | 50 | 15 | 15 | 15 | **15** |
| LayerNorm | No | No | Yes | Yes | Yes | Yes | **Yes** |
| stratified | — | — | — | — | true | false | **false** |
| target_entropy | auto | auto | auto | auto | auto | -1.5 | **-1.5** |
| lidar_rays | 27 | 27 | 27 | 27 | 27 | 27 | **450** |
| hidden_sizes | [256,256,128] | [256,256,128] | [256,256,128] | [256,256,128] | [256,256,128] | [256,256,128] | **[512,512,256]** |
| num_actors | 64 | 64 | 64 | 64 | 64 | 64 | **32** |

Last updated: 2026-03-05 (360° LiDAR, max_speed=4.0, eraser mode in zone painter)
