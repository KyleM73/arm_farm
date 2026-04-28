# arm_farm

SO-ARM101 data collection and behavior cloning, built on
[lerobot](https://github.com/huggingface/lerobot) (tracked from `main`).

## Setup

```bash
uv sync
cp .env.example .env
# edit .env — fill in serial ports and camera indices
source .env
```

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

The scripts export `HF_LEROBOT_HOME=<repo>/data` automatically.

Pretrained policy weights downloaded from the HF Hub (used by SmolVLA,
Pi0, etc.) live in the system-wide `~/.cache/huggingface/hub/` and are
shared across projects. Override with `HF_HOME` to keep them in-repo.

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
