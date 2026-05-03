"""Convert ``.pt`` checkpoints in a wandb run to ``.onnx``.

For runs that only have ``.pt`` files in wandb. The env is built once
(mjwarp ~30 s on Mac CPU) and reused for every ``.pt``; each produced
``.onnx`` is uploaded back to the run by default. Task is auto-detected
from the run config; pass ``--task`` to override.

Usage::

    uv run --extra sim convert2onnx <entity>/<project>/<run_id>
    uv run --extra sim convert2onnx <entity>/<project>/<run_id> --names model_500.pt
    uv run --extra sim convert2onnx <entity>/<project>/<run_id> --task Cube-Rgb
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import tyro

import arm_farm.sim  # noqa: F401  (registers tasks)
from arm_farm.sim.preview import (
    _WANDB_CACHE_DIR,
    _download_wandb_file,
    _list_wandb_checkpoint_files,
)

logger = logging.getLogger(__name__)


def _detect_task_from_run(run_path: str) -> str | None:
    """Best-effort task-ID lookup from the wandb run config."""
    import wandb

    try:
        api = wandb.Api()
        run = api.run(run_path)
    except Exception as e:
        logger.warning("Could not read wandb run config: %s", e)
        return None
    cfg = dict(run.config or {})
    # mjlab versions log the task under different keys.
    for key in ("task", "task_id", "env_task", "env.task"):
        v = cfg.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _convert_one(pt_path: Path, env_w, runner, *, device: str) -> Path:
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists():
        logger.info("Skipping %s: %s already exists", pt_path.name, onnx_path.name)
        return onnx_path

    runner.load(str(pt_path), load_cfg={"actor": True}, strict=True, map_location=device)
    runner.export_policy_to_onnx(str(pt_path.parent), onnx_path.name)
    metadata = get_base_metadata(env_w.unwrapped, "local-conversion")
    attach_metadata_to_onnx(str(onnx_path), metadata)
    logger.info("Wrote %s", onnx_path)
    return onnx_path


def main(
    wandb_run_path: tyro.conf.Positional[str],
    /,
    task: str | None = None,
    names: tuple[str, ...] = (),
    device: str = "cpu",
    upload: bool = True,
    cache_dir: Path = _WANDB_CACHE_DIR,
) -> None:
    """Convert ``.pt`` checkpoints in a wandb run to ``.onnx``.

    Args:
        wandb_run_path: ``entity/project/run_id``.
        task: mjlab task ID; auto-detected from the run config when omitted.
        names: subset of ``.pt`` filenames; empty means all.
        device: CPU is safest — CUDA can differ in ONNX trace fidelity.
        upload: if True, attach each produced ``.onnx`` back to the run.
        cache_dir: download cache.
    """
    files = _list_wandb_checkpoint_files(wandb_run_path)
    pts = sorted(f for f in files if f.endswith(".pt"))
    if not pts:
        raise SystemExit(f"Run {wandb_run_path!r} has no .pt files to convert.")
    if names:
        pts = [p for p in pts if Path(p).name in names or p in names]
        if not pts:
            raise SystemExit(f"None of {names!r} are .pt files in run {wandb_run_path!r}.")

    if task is None:
        task = _detect_task_from_run(wandb_run_path)
        if task is None:
            raise SystemExit("Could not auto-detect task from wandb run config. Pass --task explicitly.")
    logger.info("Converting %d .pt file(s) from %s (task=%s)...", len(pts), wandb_run_path, task)

    # Local imports defer mjwarp until a conversion is actually requested.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from arm_farm.sim.runner import ArmFarmManipulationRunner

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or ArmFarmManipulationRunner
    runner = runner_cls(env_w, asdict(agent_cfg), device=device)

    converted: list[Path] = []
    try:
        for pt_name in pts:
            local_pt = _download_wandb_file(wandb_run_path, pt_name, cache_dir=cache_dir)
            converted.append(_convert_one(local_pt, env_w, runner, device=device))
    finally:
        env.close()

    if upload and converted:
        _upload_onnx_to_wandb(wandb_run_path, converted)


def _upload_onnx_to_wandb(run_path: str, onnx_paths: list[Path]) -> None:
    """Attach ONNX files to the run via ``wandb.Run.upload_file``
    (no ``wandb.init`` required). Upload failures are warned, not fatal."""
    import wandb

    try:
        api = wandb.Api()
        run = api.run(run_path)
    except Exception as e:
        logger.warning("Skipping wandb upload: could not open run %s: %s", run_path, e)
        return
    for path in onnx_paths:
        try:
            run.upload_file(str(path), root=str(path.parent))
            logger.info("Uploaded %s to wandb run %s", path.name, run_path)
        except Exception as e:
            logger.warning("Could not upload %s to wandb: %s", path.name, e)


def cli() -> None:
    from arm_farm.sim._env import load_env

    load_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(main)


if __name__ == "__main__":
    cli()
