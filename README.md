# arm_farm

SO-ARM101 data collection and behavior cloning, built on
[lerobot](https://github.com/huggingface/lerobot) (tracked from `main`).

## Setup

```bash
uv sync
cp .env.example .env
# edit .env — fill in serial ports and camera indices
```

The scripts in `scripts/` auto-load `.env` from the repo root, so you don't
need to `source` it. For ad-hoc Python (e.g. `uv run lerobot-find-port`,
`uv run lerobot-setup-motors`) load it into the current shell with
`set -a; source .env; set +a` so the vars are exported.

To discover unknown ports or camera indices:

```bash
uv run lerobot-find-port
uv run lerobot-find-cameras
```

Brand-new arms also need a one-time motor ID assignment via
`uv run lerobot-setup-motors`.

## First-time per arm: calibrate

```bash
./scripts/calibrate.sh
```

Calibration files land under `data/calibration/`, keyed by the IDs in
`.env`, and are reused on every subsequent run.

## Daily loop

```bash
./scripts/teleoperate.sh                                  # sanity check
./scripts/record.sh "Pick up the red cube" pickplace 50   # 50 episodes
./scripts/train.sh act pickplace                          # train ACT
./scripts/train.sh diffusion pickplace                    # or diffusion
```

## Storage

Recorded data and training artifacts live inside the repo and are
gitignored:

- `data/<HF_USER>/<dataset_name>/` — recorded episodes (parquet + mp4)
- `data/calibration/` — per-arm calibration JSON
- `data/hub/` — lerobot's dataset-revision cache (not model weights)
- `outputs/train/<run_id>/` — checkpoints, configs, logs
- `outputs/sim/<run_id>/` — mjlab training run artifacts

The scripts export `HF_LEROBOT_HOME=<repo>/data` automatically.

Pretrained policy weights downloaded from the HF Hub (used by SmolVLA,
Pi0, etc.) live in the system-wide `~/.cache/huggingface/hub/` and are
shared across projects. Override with `HF_HOME` to keep them in-repo.

## Sim

Optional simulation path on top of [mjlab](https://github.com/mujocolab/mjlab).
Train RL policies on a Linux + NVIDIA CUDA desktop, play and replay
recorded datasets on macOS without code changes.

```bash
uv sync --extra sim    # mjlab + mujoco-warp; CPU torch on macOS, cu128 torch on Linux
```

Registered tasks (mjlab IDs):

- `Play` — SO-ARM101 idle on a plane next to a free cube. No rewards.
- `Cube` — state-only lift task (joint pos/vel + ee→cube + cube→goal).
- `Cube-Rgb` — front camera (64×64 RGB) with cube-color domain randomization.
- `Cube-Depth` — wrist camera (32×32 depth, 0.5 m cutoff).

Daily commands:

```bash
./scripts/sim/preview.sh                  # Play with --agent zero (smoke test)
./scripts/sim/play.sh Cube --agent zero   # blind lift task with random actions
./scripts/sim/play_native.sh              # native mujoco preview (fast on Mac)
./scripts/sim/train.sh Cube --env.scene.num-envs 4096   # Linux + CUDA only
./scripts/sim/replay.sh \
    --dataset.repo_id=arm_farm/pickplace --dataset.episode=0   # replay recorded teleop
```

Replay routes through a custom lerobot `Robot` (`mujoco_so101`) that wraps
an mjlab env; recorded SO-ARM101 datasets feed it directly without remap.

Two viewer backends are wired up:

- **Viser** (default) — web viewer via `mjviser.ViserMujocoScene`. The URL is
  printed in the terminal on startup. No local display or special launcher
  needed; works the same on macOS and Linux, headed or headless.
- **Native MuJoCo** — OpenGL window via `mujoco.viewer.launch_passive`. On
  macOS the script automatically re-execs under `mjpython` (mujoco's
  Cocoa-aware launcher) with `DYLD_FALLBACK_LIBRARY_PATH` set so it picks up
  uv's `libpython3.12.dylib`.

For mjlab-driven playback (`play.sh`, `preview.sh`, `train.sh`):

```bash
./scripts/sim/play.sh Cube                  # default: viser
./scripts/sim/play.sh Cube --viewer=native  # OpenGL window
```

For dataset replay through the `mujoco_so101` wrapper, set
`ARM_FARM_SIM_VIEWER` (`viser` | `native` | `none`):

```bash
./scripts/sim/replay.sh \
    --dataset.repo_id=arm_farm/pickplace --dataset.episode=0
ARM_FARM_SIM_VIEWER=native ./scripts/sim/replay.sh \
    --dataset.repo_id=arm_farm/pickplace --dataset.episode=0
```

### Fast Mac preview (native MuJoCo)

mjwarp's CPU backend is slow per-step on Apple silicon — even with one env
the per-frame kernel dispatch dominates. `./scripts/sim/play_native.sh`
bypasses mjwarp entirely: it reuses mjlab's spec composition (which is
mjwarp-free) but drives the rollout under native `mujoco.mj_step`, with
viser as the visualizer. Roughly 10x faster wall-clock for previewing the
SO-ARM101 + cube scene on a Mac.

Scripted action sources (no checkpoint needed):

```bash
./scripts/sim/play_native.sh                            # Cube task, zero actions
./scripts/sim/play_native.sh --task=Play                # idle scene
./scripts/sim/play_native.sh --task=Cube --action=sine  # sinusoidal joint sweep
./scripts/sim/play_native.sh --action=random            # random actuator targets
```

Trained-policy playback (`--checkpoint <path.onnx>`):
mjlab's `ManipulationOnPolicyRunner` auto-exports an ONNX policy
alongside each checkpoint save with `joint_names`, `observation_names`,
`action_scale`, and `default_joint_pos` metadata embedded. The native
runner reads that metadata to wire up the obs pipeline and action
mapping without ever instantiating mjlab's mjwarp-backed env, so
inference runs at native MuJoCo speed.

```bash
./scripts/sim/play_native.sh --task=Cube \
    --checkpoint=outputs/sim/<run_id>-Cube/<run_id>-Cube.onnx
```

The script reimplements the actor obs against `mj_data` to match
mjlab's manipulation manager term-for-term:

- `joint_pos` / `joint_vel`: relative to `default_joint_pos` (read from
  the ONNX metadata).
- `ee_to_cube` / `cube_to_goal`: world-frame deltas anchored at the
  `ee_site` site and the cube freejoint.
- `goal_position` (vision tasks): target rotated into the EE frame via
  the `ee_site` rotation matrix.
- `actions`: previous raw policy output, cached on the wrapper.
- Camera obs (`Cube-Rgb` / `Cube-Depth`): rendered per step with
  `mujoco.Renderer`, normalized identically to mjlab —
  RGB → `(3, H, W)` float32 in `[0, 1]`, depth →
  `clip(depth / cutoff_distance, 0, 1)` shaped `(1, H, W)` float32.

`LiftingCommand` goal sampling is replicated locally with the same
default ranges mjlab ships (`target_position_range`, `object_pose_range`,
4 s resampling period). Vision policies use mjlab's multi-input ONNX
export (`["obs", "camera"]`); the native runner detects the camera setup
from the registered `env_cfg` (sensor name, height/width, RGB vs depth)
so the same shell command works for blind and vision checkpoints
without extra flags.

Reference repos vendored locally for browsing (gitignored under
`src/arm_farm/sim/reference/`): mjlab, mjlab_playground,
anymal_c_velocity, lerobot-sim2real, leisaac, SO-ARM100. The SO-ARM101
MJCF + meshes are vendored from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache-2.0) under `src/arm_farm/sim/assets/so101/xmls/`.

## Layout

- `pyproject.toml` — `lerobot[feetech,core_scripts]` from git main; ruff
  and ty in `[dependency-groups].dev`.
- `src/arm_farm/hardware.py` — typed presets that read the same env vars
  as the shell scripts. Suitable for notebooks or sweep scripts.
- `scripts/` — bash wrappers for the lerobot CLIs with project defaults
  baked in.

## Lint and type-check

```bash
uv run ruff check
uv run ruff format
uv run ty check
```
