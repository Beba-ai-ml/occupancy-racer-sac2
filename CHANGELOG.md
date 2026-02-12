# Changelog

All notable changes to this project are documented in this file.

---

## [2026-02-10] Fix UTD Ratio Bug with Configurable utd_ratio

### Problem
The `// num_actors` formula divided gradient updates by 64, producing only ~8 `learn()` calls per cycle instead of a proper UTD ratio. This caused training plateaus at 30-40m instead of reaching 257m+.

### Changed
- **src/train_ssac.py**: Added `--utd-ratio` CLI argument and config key. Replaced broken `// num_actors` formula with `max(1, int(transitions_received * utd_ratio * agent.updates_per_step))`. Added UTD ratio to config snapshot and startup diagnostics.

### Added
- **config/config_sac_12.yaml**: 64 actors, 10 maps (K_01-K_10), `utd_ratio: 0.25`, `batch_size: 256`, `stack_frames: 8`, `grad_clip: 1.0`, `alpha_min: 0.01`.

---

## [2026-02-09] Learner Throughput: Drain More, Learn Less Per Cycle

### Problem
With 64 actors, the learner drained max 64 transitions per cycle then performed 64x `learn()` calls (~300ms GPU time). Meanwhile actors produced ~800 transitions, filling the queue to 48K and blocking on `put()`.

### Changed
- **src/train_ssac.py**: Increased drain limit from 63 to 512. Decoupled gradient update count from raw transition count (later superseded by the UTD ratio fix above).

---

## [2026-02-08] Fix SAC Training Degradation (grad_clip + alpha_min)

### Problem
Session `Mapa_64_22` showed performance degradation across 47K episodes: alpha collapsed to 0.002 (exploration died), Q-loss diverging (+96%), no gradient clipping or alpha floor.

### Changed
- **src/rl_agent.py**: Added `grad_clip` and `alpha_min` parameters to `SACAgent`. Gradient clipping applied after critic and policy backward passes. Alpha floor enforced after alpha optimizer step.
- **src/train_ssac.py**: Added `--grad-clip` and `--alpha-min` CLI arguments. Passed both to `SACAgent` constructor with config file fallback.

### Added
- **config/config_sac_10.yaml**: 64 actors, 40 maps, `grad_clip: 1.0`, `alpha_min: 0.01`, `target_entropy: -1.0`.
- **config/config_sac_11.yaml**: 128 actors, Dom_01 only, `grad_clip: 1.0`, `alpha_min: 0.005`.

---

## [2026-02-08] Speed Up Spawn with Zone Masks

### Changed
- **src/racer_env.py**: Added spawn position pre-filtering by wall clearance at init time. Added fast path in `_random_spawn()` that skips raycasting when positions are pre-filtered (eliminates 17 raycasts x up to 1000 attempts per spawn).
- **src/map_loader.py**: Moved PIL import to module level with `try/except`. Avoids repeated imports during map switching across 64 actors x 10+ maps. Gracefully handles missing PIL.
