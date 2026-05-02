"""Native MuJoCo + mjviser preview path that bypasses mjwarp.

mjwarp's CPU backend is slow per-step on Apple silicon (kernel dispatch
dominates), even with one env. Native ``mujoco.mj_step`` is ~10x faster.
This entry point reuses the same scene composition as a registered task
(so the SO-ARM101 + cube + plane look identical to ``play.sh``) but
drives the rollout under native MuJoCo with ``mjviser.Viewer``.

Reuses mjlab's ``Scene`` class for spec composition — that path only
manipulates ``mujoco.MjSpec`` and never touches mjwarp, so building the
model under ``device="cpu"`` is fast.

Action sources
--------------
Two families of action source are supported:

- **Scripted** (``zero``/``random``/``sine``): deterministic, useful for
  inspecting the scene without a checkpoint.
- **Policy** (``--checkpoint <path.onnx>``): runs a trained policy
  exported by mjlab's ``ManipulationOnPolicyRunner``. mjlab auto-exports
  the policy as ONNX alongside each checkpoint save with metadata
  embedded; this script reads that metadata to wire up the obs pipeline
  and action mapping correctly.

Three task variants are supported under policy mode: the blind ``Cube``
task and the two visual variants (``Cube-Rgb``, ``Cube-Depth``). The
blind obs (``joint_pos_rel``, ``joint_vel_rel``,
``ee_to_object_distance``, ``object_to_goal_distance``, ``last_action``)
and the vision obs (``target_position`` in EE frame + per-step
``camera_rgb``/``camera_depth`` rendered with ``mujoco.Renderer``) are
reimplemented term-for-term against ``mj_data`` to avoid pulling
mjlab's manager-based env (which would re-introduce mjwarp).
``LiftingCommand`` goal sampling is replicated locally with the same
defaults as ``mjlab.tasks.manipulation.mdp.commands.LiftingCommandCfg``.

Usage
-----
::

    uv run --extra sim arm-farm-play-native --task=Cube
    uv run --extra sim arm-farm-play-native --task=Cube --action=sine
    uv run --extra sim arm-farm-play-native --task=Cube --checkpoint <run>.onnx
    uv run --extra sim arm-farm-play-native --task=Cube-Rgb --checkpoint <run>.onnx
    uv run --extra sim arm-farm-play-native --task=Cube-Depth --checkpoint <run>.onnx
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import onnxruntime as ort
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import Scene
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjviser import Viewer

import arm_farm.sim  # noqa: F401  (registers tasks)

logger = logging.getLogger(__name__)

ActionSource = Literal["zero", "random", "sine"]
INIT_STATE_KEYFRAME = "init_state"
StepFn = Callable[[mujoco.MjModel, mujoco.MjData], None]
ResetFn = Callable[[mujoco.MjModel, mujoco.MjData], None]

# Match mjlab.tasks.manipulation.mdp.commands.LiftingCommandCfg defaults.
# Keep these in sync with the upstream cfg if mjlab bumps them.
TARGET_RANGE_LO = np.array([0.3, -0.2, 0.2], dtype=np.float64)
TARGET_RANGE_HI = np.array([0.5, 0.2, 0.4], dtype=np.float64)
OBJECT_RANGE_LO = np.array([0.25, -0.15, 0.02], dtype=np.float64)
OBJECT_RANGE_HI = np.array([0.40, 0.15, 0.05], dtype=np.float64)
COMMAND_RESAMPLE_PERIOD_S = 4.0  # play_env_cfg's `resampling_time_range = (4.0, 4.0)`.


# ---------------------------------------------------------------------------
# Scripted action sources (no checkpoint).
# ---------------------------------------------------------------------------


def _make_scripted_step_fn(action: ActionSource, sine_period_s: float) -> StepFn:
    if action == "zero":
        return mujoco.mj_step

    if action == "random":

        def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            data.ctrl[:] = np.random.uniform(lo, hi)
            mujoco.mj_step(model, data)

        return step

    if action == "sine":
        omega = 2.0 * math.pi / sine_period_s

        def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            phase = data.time * omega
            mid = 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])
            half = 0.5 * (model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0])
            data.ctrl[:] = mid + half * math.sin(phase)
            mujoco.mj_step(model, data)

        return step

    raise ValueError(f"Unknown action source: {action!r}")


# ---------------------------------------------------------------------------
# ONNX policy inference for the Cube / Cube-Rgb / Cube-Depth tasks.
# ---------------------------------------------------------------------------


def _parse_csv_metadata(value: str, *, dtype: type = float) -> list:
    """Parse mjlab's ``list_to_csv_str``-encoded metadata back into a list."""
    if not value:
        return []
    return [dtype(x) for x in value.split(",")]


def _resolve_name(model: mujoco.MjModel, obj: int, candidates: list[str]) -> int:
    """Return the first matching id among `candidates`, or raise."""
    for name in candidates:
        idx = mujoco.mj_name2id(model, obj, name)
        if idx >= 0:
            return idx
    raise KeyError(f"None of {candidates!r} found in model (obj_type={obj})")


def _find_cube_freejoint(model: mujoco.MjModel) -> int:
    """Return the joint id of the cube's freejoint (the only freejoint)."""
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name and "cube" in name:
                return jid
    raise RuntimeError("No cube freejoint found in compiled model")


def _find_cube_body(model: mujoco.MjModel) -> int:
    """Return the body id of the cube (matches the cube freejoint's parent)."""
    cube_jid = _find_cube_freejoint(model)
    return int(model.jnt_bodyid[cube_jid])


# ---------------------------------------------------------------------------
# Vision config + camera rendering.
# ---------------------------------------------------------------------------


CameraDataType = Literal["rgb", "depth"]


@dataclass
class _VisionCfg:
    """Per-task camera config inferred from the registered env_cfg."""

    obs_input_name: str  # ONNX input tensor name, e.g. "camera"
    obs_term_name: str  # term key inside the obs group, e.g. "front" / "wrist"
    sensor_name: str  # CameraSensorCfg.name (matches MJCF camera name suffix)
    camera_name_in_model: str  # mujoco camera name including entity prefix, e.g. "robot/front"
    data_type: CameraDataType  # "rgb" or "depth"
    height: int
    width: int
    cutoff_distance: float | None  # None for RGB; clamp for depth
    enabled_geom_groups: tuple[int, ...]


def _resolve_obs_term_func_name(func: Any) -> str:
    """Return the original function name even after module re-exports."""
    return getattr(func, "__name__", repr(func))


def _vision_cfg_from_env_cfg(env_cfg: ManagerBasedRlEnvCfg, ort_input_names: list[str]) -> _VisionCfg | None:
    """Inspect the registered env_cfg to figure out the camera setup, if any.

    Returns ``None`` for blind tasks (no ``camera`` obs group). Otherwise
    returns the camera shape, mujoco name, and rgb/depth data type so
    the step_fn can render and feed the ONNX policy correctly.
    """
    if "camera" not in env_cfg.observations:
        return None
    if len(ort_input_names) < 2:
        # Blind ONNX, but the env_cfg would feed it a camera. Most likely
        # the user passed the wrong checkpoint for the task. Surface a
        # clear error rather than running with garbage inputs.
        raise RuntimeError(
            f"Task has a 'camera' obs group but the ONNX has only {len(ort_input_names)} "
            "input(s); pass an ONNX exported from the matching vision task."
        )
    if "camera" not in ort_input_names:
        raise RuntimeError(
            f"ONNX inputs {ort_input_names!r} do not include 'camera'; expected mjlab >= 1.3 vision export."
        )

    camera_group = env_cfg.observations["camera"]
    if len(camera_group.terms) != 1:
        raise NotImplementedError(
            f"Native vision path only supports a single camera term; got {list(camera_group.terms)}."
        )
    obs_term_name, term = next(iter(camera_group.terms.items()))
    sensor_name = term.params["sensor_name"]
    func_name = _resolve_obs_term_func_name(term.func)
    if func_name == "camera_rgb":
        data_type: CameraDataType = "rgb"
        cutoff = None
    elif func_name == "camera_depth":
        data_type = "depth"
        cutoff = float(term.params.get("cutoff_distance", 1.0))
    else:
        raise NotImplementedError(
            f"Native vision path does not know how to render camera obs func {func_name!r}."
        )

    sensors = env_cfg.scene.sensors or ()
    cam_sensor: CameraSensorCfg | None = next(
        (s for s in sensors if isinstance(s, CameraSensorCfg) and s.name == sensor_name),
        None,
    )
    if cam_sensor is None:
        raise RuntimeError(
            f"CameraSensorCfg for sensor_name={sensor_name!r} not found in scene.sensors; "
            f"available: {[s.name for s in sensors]}."
        )

    return _VisionCfg(
        obs_input_name="camera",
        obs_term_name=obs_term_name,
        sensor_name=sensor_name,
        camera_name_in_model=cam_sensor.camera_name or sensor_name,
        data_type=data_type,
        height=int(cam_sensor.height),
        width=int(cam_sensor.width),
        cutoff_distance=cutoff,
        enabled_geom_groups=tuple(cam_sensor.enabled_geom_groups),
    )


def _build_render_scene_option(enabled_geom_groups: tuple[int, ...]) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in enabled_geom_groups:
        if 0 <= g < len(opt.geomgroup):
            opt.geomgroup[g] = 1
    return opt


def _make_renderer(model: mujoco.MjModel, vcfg: _VisionCfg) -> mujoco.Renderer:
    renderer = mujoco.Renderer(model, height=vcfg.height, width=vcfg.width)
    if vcfg.data_type == "depth":
        renderer.enable_depth_rendering()
    return renderer


def _render_camera_obs(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    vcfg: _VisionCfg,
    scene_option: mujoco.MjvOption,
) -> np.ndarray:
    """Render the camera and apply mjlab's per-data-type normalization.

    RGB matches ``manipulation_mdp.camera_rgb``: ``(3, H, W) float32`` in
    ``[0, 1]``. Depth matches ``manipulation_mdp.camera_depth``:
    ``(1, H, W) float32`` clamped to ``[0, 1]`` after dividing by the
    configured ``cutoff_distance``.
    """
    renderer.update_scene(data, camera=vcfg.camera_name_in_model, scene_option=scene_option)
    frame = renderer.render()  # rgb: (H, W, 3) uint8; depth: (H, W) float32
    if vcfg.data_type == "rgb":
        return frame.transpose(2, 0, 1).astype(np.float32) / 255.0
    cutoff = vcfg.cutoff_distance or 1.0
    return np.clip(frame / cutoff, 0.0, 1.0).astype(np.float32)[None, :, :]


# ---------------------------------------------------------------------------
# Policy context + obs computation.
# ---------------------------------------------------------------------------


@dataclass
class _PolicyCtx:
    """Indices and state needed to run an ONNX Cube-task policy each step."""

    session: ort.InferenceSession
    obs_names: list[str]
    state_input_name: str  # ONNX input name for the 1D state, e.g. "obs"
    action_scale: np.ndarray  # (6,)
    default_joint_pos: np.ndarray  # (6,)
    joint_qpos_adrs: np.ndarray  # (6,) qpos addresses for actuated joints
    joint_qvel_adrs: np.ndarray  # (6,) qvel addresses
    actuator_ids: np.ndarray  # (6,) ctrl indices
    ee_site_id: int
    cube_body_id: int
    cube_qpos_adr: int  # qpos address for cube freejoint (7 fields: x,y,z,qw,qx,qy,qz)
    cube_qvel_adr: int  # qvel address for cube freejoint (6 fields: vx,vy,vz,wx,wy,wz)
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    next_resample_time: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    vision: _VisionCfg | None = None
    renderer: mujoco.Renderer | None = None
    scene_option: mujoco.MjvOption | None = None


def _build_policy_context(
    onnx_path: Path,
    model: mujoco.MjModel,
    seed: int,
    env_cfg: ManagerBasedRlEnvCfg,
) -> _PolicyCtx:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Pull the metadata mjlab embedded at export time.
    md = session.get_modelmeta().custom_metadata_map
    obs_names = _parse_csv_metadata(md.get("observation_names", ""), dtype=str)
    joint_names = _parse_csv_metadata(md.get("joint_names", ""), dtype=str)
    action_scale = np.array(_parse_csv_metadata(md.get("action_scale", "")), dtype=np.float32)
    default_joint_pos = np.array(_parse_csv_metadata(md.get("default_joint_pos", "")), dtype=np.float64)

    if not obs_names:
        raise RuntimeError(
            f"ONNX at {onnx_path} has no `observation_names` metadata; expected mjlab>=1.3 export."
        )
    if action_scale.shape != (len(joint_names),):
        raise RuntimeError(
            f"action_scale shape {action_scale.shape} does not match joint_names length {len(joint_names)}."
        )

    # Resolve indices in the compiled model. mjlab attaches the robot under
    # the "robot/" prefix; raw joint names also resolve when no prefix is
    # used (e.g. when the model was compiled from so101.xml directly).
    joint_qpos_adrs: list[int] = []
    joint_qvel_adrs: list[int] = []
    actuator_ids: list[int] = []
    for j in joint_names:
        jid = _resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, [f"robot/{j}", j])
        joint_qpos_adrs.append(int(model.jnt_qposadr[jid]))
        joint_qvel_adrs.append(int(model.jnt_dofadr[jid]))
        aid = _resolve_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"robot/{j}", j])
        actuator_ids.append(aid)

    ee_site_id = _resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, ["robot/ee_site", "ee_site"])
    cube_body_id = _find_cube_body(model)
    cube_jid = _find_cube_freejoint(model)
    cube_qpos_adr = int(model.jnt_qposadr[cube_jid])
    cube_qvel_adr = int(model.jnt_dofadr[cube_jid])

    ort_input_names = [inp.name for inp in session.get_inputs()]
    vision = _vision_cfg_from_env_cfg(env_cfg, ort_input_names)
    renderer: mujoco.Renderer | None = None
    scene_option: mujoco.MjvOption | None = None
    if vision is not None:
        renderer = _make_renderer(model, vision)
        scene_option = _build_render_scene_option(vision.enabled_geom_groups)

    # The state input is whichever ONNX input is NOT the camera input.
    state_input_name = next(
        (n for n in ort_input_names if vision is None or n != vision.obs_input_name),
        ort_input_names[0],
    )

    return _PolicyCtx(
        session=session,
        obs_names=obs_names,
        state_input_name=state_input_name,
        action_scale=action_scale,
        default_joint_pos=default_joint_pos,
        joint_qpos_adrs=np.array(joint_qpos_adrs, dtype=np.int32),
        joint_qvel_adrs=np.array(joint_qvel_adrs, dtype=np.int32),
        actuator_ids=np.array(actuator_ids, dtype=np.int32),
        ee_site_id=ee_site_id,
        cube_body_id=cube_body_id,
        cube_qpos_adr=cube_qpos_adr,
        cube_qvel_adr=cube_qvel_adr,
        rng=np.random.default_rng(seed),
        vision=vision,
        renderer=renderer,
        scene_option=scene_option,
    )


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    """Return the unit quaternion (w, x, y, z) for a rotation about z by ``yaw``."""
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _resample_command(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> None:
    """Replicate ``LiftingCommand._resample_command`` against native ``mj_data``.

    Samples a new target position from ``target_position_range`` and
    re-spawns the cube uniformly in ``object_pose_range`` with a random yaw
    (matches mjlab's defaults; see TARGET_RANGE_* / OBJECT_RANGE_* up top).
    """
    ctx.target_pos[:] = ctx.rng.uniform(TARGET_RANGE_LO, TARGET_RANGE_HI)

    cube_xy_z = ctx.rng.uniform(OBJECT_RANGE_LO, OBJECT_RANGE_HI)
    cube_yaw = ctx.rng.uniform(-math.pi, math.pi)
    quat_wxyz = _yaw_quat_wxyz(cube_yaw)
    qpos_slice = slice(ctx.cube_qpos_adr, ctx.cube_qpos_adr + 7)
    qvel_slice = slice(ctx.cube_qvel_adr, ctx.cube_qvel_adr + 6)
    data.qpos[qpos_slice] = np.concatenate([cube_xy_z, quat_wxyz])
    data.qvel[qvel_slice] = 0.0
    mujoco.mj_forward(model, data)


def _ee_frame(data: mujoco.MjData, ee_site_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ee_pos_w, ee_xmat)`` for the grasp site."""
    ee_pos = np.asarray(data.site_xpos[ee_site_id], dtype=np.float64)
    ee_xmat = np.asarray(data.site_xmat[ee_site_id], dtype=np.float64).reshape(3, 3)
    return ee_pos, ee_xmat


def _compute_actor_obs(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> np.ndarray:
    """Build the actor obs vector matching mjlab's Cube task obs manager.

    Order is dictated by ``ctx.obs_names`` (read from the ONNX metadata),
    so the term layout in the obs vector tracks whatever the task registered.
    ``goal_position`` matches ``manipulation_mdp.target_position`` (target in
    the EE frame).
    """
    parts: list[np.ndarray] = []
    for name in ctx.obs_names:
        if name == "joint_pos":
            parts.append(data.qpos[ctx.joint_qpos_adrs] - ctx.default_joint_pos)
        elif name == "joint_vel":
            parts.append(data.qvel[ctx.joint_qvel_adrs])
        elif name == "ee_to_cube":
            ee_pos, _ = _ee_frame(data, ctx.ee_site_id)
            cube_pos = np.asarray(data.xpos[ctx.cube_body_id], dtype=np.float64)
            parts.append(cube_pos - ee_pos)
        elif name == "cube_to_goal":
            cube_pos = np.asarray(data.xpos[ctx.cube_body_id], dtype=np.float64)
            parts.append(ctx.target_pos - cube_pos)
        elif name == "goal_position":
            ee_pos, ee_xmat = _ee_frame(data, ctx.ee_site_id)
            parts.append(ee_xmat.T @ (ctx.target_pos - ee_pos))
        elif name == "actions":
            parts.append(ctx.last_action.astype(np.float64))
        else:
            raise NotImplementedError(
                f"Native obs term {name!r} not implemented. "
                "If the policy expects it, extend `_compute_actor_obs`."
            )
    return np.concatenate(parts).astype(np.float32)


def _make_policy_step_fn(ctx: _PolicyCtx) -> StepFn:
    """Wrap ONNX inference + the JointPositionAction mapping into a step_fn."""

    def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if data.time >= ctx.next_resample_time:
            _resample_command(model, data, ctx)
            ctx.next_resample_time = data.time + COMMAND_RESAMPLE_PERIOD_S

        inputs: dict[str, np.ndarray] = {
            ctx.state_input_name: _compute_actor_obs(model, data, ctx)[None, :],
        }
        if ctx.vision is not None:
            assert ctx.renderer is not None and ctx.scene_option is not None
            cam = _render_camera_obs(ctx.renderer, data, ctx.vision, ctx.scene_option)
            inputs[ctx.vision.obs_input_name] = cam[None, :]  # (1, C, H, W)

        action = ctx.session.run(None, inputs)[0][0]  # ty: ignore[not-subscriptable]
        ctx.last_action[:] = action

        # JointPositionAction: target = raw_action * scale + default_joint_pos.
        target = action.astype(np.float64) * ctx.action_scale + ctx.default_joint_pos
        data.ctrl[ctx.actuator_ids] = target

        mujoco.mj_step(model, data)

    return step


# ---------------------------------------------------------------------------
# Scene + viewer plumbing.
# ---------------------------------------------------------------------------


def _reset_to_init_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset ``data`` to the ``init_state`` keyframe mjlab.Scene composes."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, INIT_STATE_KEYFRAME)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


_WANDB_CACHE_DIR = Path.home() / ".cache" / "arm_farm" / "wandb_onnx"


def _resolve_wandb_onnx(
    run_path: str, onnx_name: str | None = None, cache_dir: Path = _WANDB_CACHE_DIR
) -> Path:
    """Download (or reuse cached) ONNX policy from a wandb run.

    ``run_path`` is the standard ``entity/project/run_id`` form (e.g.
    ``Apptronik/arm-farm-lift-rgb/227y12j9``). ``onnx_name`` is optional;
    if omitted, picks the only ``.onnx`` file in the run. Cached under
    ``~/.cache/arm_farm/wandb_onnx/<run_id>/<filename>`` so subsequent
    rollouts reuse without re-downloading.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    run_id = run_path.rsplit("/", 1)[-1]
    cache_subdir = cache_dir / run_id
    cache_subdir.mkdir(parents=True, exist_ok=True)

    onnx_files = [f for f in run.files() if f.name.endswith(".onnx")]
    if not onnx_files:
        raise RuntimeError(
            f"No .onnx files in wandb run {run_path!r}. mjlab uploads ONNX only "
            "when --agent.upload-model True (the default) AND a save_interval has "
            "fired at least once."
        )
    if onnx_name is None:
        if len(onnx_files) > 1:
            names = sorted(f.name for f in onnx_files)
            raise RuntimeError(
                f"Run {run_path!r} has multiple .onnx files: {names}. Pass --wandb-onnx-name to disambiguate."
            )
        chosen = onnx_files[0]
    else:
        candidates = [f for f in onnx_files if f.name.endswith(onnx_name) or Path(f.name).name == onnx_name]
        if not candidates:
            available = sorted(f.name for f in onnx_files)
            raise RuntimeError(f"ONNX {onnx_name!r} not in run {run_path!r}. Available: {available}")
        chosen = candidates[0]

    local_path = cache_subdir / Path(chosen.name).name
    if not local_path.exists():
        logger.info("Downloading ONNX %s from wandb run %s -> %s", chosen.name, run_path, local_path)
        chosen.download(root=str(cache_subdir), replace=True)
        # wandb may preserve subdir structure inside chosen.name; normalize.
        downloaded = cache_subdir / chosen.name
        if downloaded != local_path and downloaded.exists():
            downloaded.rename(local_path)
    else:
        logger.info("Using cached wandb ONNX at %s", local_path)
    return local_path


def main(
    task: str = "Cube",
    action: ActionSource = "zero",
    sine_period_s: float = 4.0,
    checkpoint: Path | None = None,
    wandb_run_path: str | None = None,
    wandb_onnx_name: str | None = None,
    seed: int = 0,
) -> None:
    """Open a native-mujoco rollout of a registered task in the viser viewer.

    Args:
        task: registered mjlab task ID (``Cube``, ``Cube-Rgb``, ``Cube-Depth``,
            ``Play``, or any of the upstream mjlab tasks). Defaults to ``Cube``.
        action: scripted action source — ``zero`` (default) | ``random`` |
            ``sine``. Ignored when ``--checkpoint`` or ``--wandb-run-path`` is set.
        sine_period_s: full-sweep period for the ``sine`` action source.
        checkpoint: path to a mjlab-exported ONNX policy. When provided, the
            policy drives the rollout instead of the scripted action source.
        wandb_run_path: ``entity/project/run_id`` of a mjlab training run. Mutually
            exclusive with ``--checkpoint``; downloads the ONNX from wandb (cached
            under ``~/.cache/arm_farm/wandb_onnx/``) and uses it as the policy.
        wandb_onnx_name: optional ONNX filename inside the wandb run when
            multiple are uploaded. Defaults to the only .onnx in the run.
        seed: RNG seed for command resampling and random-action mode.
    """
    if checkpoint is not None and wandb_run_path is not None:
        raise SystemExit("Pass either --checkpoint or --wandb-run-path, not both.")
    if wandb_run_path is not None:
        checkpoint = _resolve_wandb_onnx(wandb_run_path, wandb_onnx_name)

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1

    scene = Scene(env_cfg.scene, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    _reset_to_init_state(model, data)

    if checkpoint is not None:
        ctx = _build_policy_context(checkpoint, model, seed, env_cfg)
        # Sample a goal + cube pose immediately so frame 0 already shows the
        # scene the policy will be asked to solve.
        _resample_command(model, data, ctx)
        ctx.next_resample_time = data.time + COMMAND_RESAMPLE_PERIOD_S
        step_fn = _make_policy_step_fn(ctx)
        cam_desc = (
            f"camera={ctx.vision.data_type} {ctx.vision.height}x{ctx.vision.width}"
            if ctx.vision is not None
            else "camera=none"
        )
        logger.info(
            "Native preview: task=%s, checkpoint=%s, obs_names=%s, %s",
            task,
            checkpoint,
            ctx.obs_names,
            cam_desc,
        )
    else:
        np.random.seed(seed)
        step_fn = _make_scripted_step_fn(action=action, sine_period_s=sine_period_s)
        logger.info(
            "Native preview: task=%s, action=%s, nq=%d, nu=%d",
            task,
            action,
            model.nq,
            model.nu,
        )

    # mjviser is the visualizer for every action source and platform; see
    # mjviser.Viewer for pause/resume and speed controls.
    Viewer(model, data, step_fn=step_fn, reset_fn=_reset_to_init_state).run()


def cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(main)


if __name__ == "__main__":
    cli()
