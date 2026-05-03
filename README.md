# arm_farm

SO-ARM101 data collection and behavior cloning, built on
[lerobot](https://github.com/huggingface/lerobot).

## Setup

```bash
uv sync
cp .env.example .env
# edit .env — fill in serial ports and camera indices
```

Scripts under `scripts/` auto-load `.env`. For ad-hoc Python, source it:
`set -a; source .env; set +a`.

Discover ports / cameras / motor IDs (one-time per arm):

```bash
uv run lerobot-find-port
uv run lerobot-find-cameras
uv run lerobot-setup-motors  # brand-new arms only
```

## First-time per arm: calibrate

```bash
./scripts/calibrate.sh
```

Calibration JSON lands in `data/calibration/`, keyed by the IDs in `.env`.

## Daily loop

```bash
./scripts/teleoperate.sh                                  # sanity check
./scripts/record.sh "Pick up the red cube" pickplace 50   # 50 episodes
./scripts/train.sh act pickplace                          # train ACT
./scripts/train.sh diffusion pickplace                    # or diffusion
```

## Storage

Recorded data and training artifacts live in-repo (gitignored):

- `data/<HF_USER>/<dataset_name>/` — recorded episodes (parquet + mp4)
- `data/calibration/` — per-arm calibration JSON
- `data/hub/` — lerobot's dataset-revision cache
- `outputs/train/<run_id>/` — checkpoints, configs, logs
- `outputs/sim/<run_id>/` — mjlab training run artifacts

Scripts set `HF_LEROBOT_HOME=<repo>/data` automatically. Pretrained policy
weights (SmolVLA, Pi0, …) cache to `~/.cache/huggingface/hub/`; override
with `HF_HOME` to keep them in-repo.

## Sim

Optional [mjlab](https://github.com/mujocolab/mjlab)-based simulation. Train
RL on Linux + CUDA, preview and replay datasets on macOS.

```bash
uv sync --extra sim
```

Registered tasks:

- `Play` — robot + cube + plane, no rewards. Smoke test.
- `Cube` — state-only lift (joint pos/vel + ee→cube + cube→goal).
- `Cube-Rgb` — front 32×32 RGB camera + cube-color DR.
- `Cube-Depth` — wrist 32×32 depth, 0.5 m cutoff.

Daily commands:

```bash
uv run --extra sim preview --task=Play                    # idle smoke test
uv run --extra sim preview                                # Cube, native mujoco (fast on Mac)
./scripts/sim/play.sh Cube --agent zero                   # via mjlab + mjwarp
./scripts/sim/train.sh Cube --env.scene.num-envs 4096     # Linux + CUDA only
./scripts/sim/replay.sh \
    --dataset.repo_id=arm_farm/pickplace --dataset.episode=0
```

Replay routes through a custom lerobot Robot (`mujoco_so101`) so recorded
SO-ARM101 datasets feed the sim without remap.

### Viewers

Two backends:

- **Viser** (default) — web viewer; URL prints on startup. No display
  required, works headed or headless on macOS and Linux.
- **Native MuJoCo** — OpenGL window via `launch_passive`. macOS scripts
  re-exec under `mjpython` with `DYLD_FALLBACK_LIBRARY_PATH` pointed at
  uv's `libpython` so Cocoa hosting works.

`play.sh`/`train.sh` accept `--viewer=native|viser`. `preview` is viser-only
— its win is the in-viewer GUI: per-camera panels, wandb checkpoint
dropdown, live per-term reward plot. For dataset replay through
`mujoco_so101`, set `ARM_FARM_SIM_VIEWER` to `viser`/`native`/`none`.

### Fast Mac preview

mjwarp's CPU backend is slow on Apple silicon (kernel dispatch dominates).
`preview` reuses mjlab's spec composition (mjwarp-free) and drives rollouts
under native `mujoco.mj_step`. Roughly 10x faster.

```bash
uv run --extra sim preview                            # zero actions
uv run --extra sim preview --task=Cube --action=sine  # sinusoidal sweep
uv run --extra sim preview --action=random
```

Pass `--checkpoint <path.onnx>` to roll out a trained policy. mjlab's
`ManipulationOnPolicyRunner` auto-exports ONNX with `joint_names`,
`observation_names`, `action_scale`, `default_joint_pos` metadata; `preview`
reads it to wire up obs and action mapping without instantiating
mjlab's mjwarp-backed env.

```bash
uv run --extra sim preview --task=Cube \
    --checkpoint=outputs/sim/<run_id>-Cube/<run_id>-Cube.onnx
```

Vision tasks use multi-input ONNX (`["obs", "camera"]`); the camera setup
is read from the registered `env_cfg`, so the same flag works for blind
and vision checkpoints. `LiftingCommand` goal sampling is replicated
locally against the same `target_position_range` / `object_pose_range`
mjlab ships.

The SO-ARM101 MJCF + meshes are vendored from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache-2.0) under `src/arm_farm/sim/assets/so101/xmls/`. Local
read-only checkouts of upstream repos (mjlab, mjlab_playground,
anymal_c_velocity, lerobot-sim2real, leisaac, SO-ARM100) currently sit
under `src/arm_farm/sim/reference/`; they're untracked and will be
removed from the working tree in a future commit.

## Layout

- `pyproject.toml` — deps and `[project.scripts]` entry points.
- `src/arm_farm/hardware.py` — typed lerobot presets driven by env vars.
- `scripts/` — bash wrappers for lerobot/mjlab CLIs with project defaults.

## Lint and type-check

```bash
uv run ruff check
uv run ruff format
uv run ty check
```
