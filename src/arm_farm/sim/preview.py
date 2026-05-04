"""Native MuJoCo + mjviser preview of a registered task.

Reuses mjlab's ``Scene`` for spec composition but drives the rollout
under native ``mujoco.mj_step`` — mjwarp and the manager-based env
never load, so Mac CPU playback runs ~10x faster. Obs and the
``LiftingCommand`` goal sampler are reimplemented term-for-term against
``mj_data``.

Action sources: ``zero``/``random``/``sine`` or an ONNX policy exported
by ``ArmFarmManipulationRunner``. The viser GUI adds a wandb checkpoint
dropdown, camera panels, and live reward plots.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
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

# 10 s at ~28.57 Hz policy cadence (200 Hz physics, decimation=7).
REWARD_PLOT_HISTORY = 286


def _make_scripted_step_fn(
    action: ActionSource,
    sine_period_s: float,
    on_tick: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    tick_period: int = 4,
) -> StepFn:
    """Scripted-action ``step_fn``. ``on_tick`` fires once per ``tick_period``
    mj_steps so callers can hook reward/preview pumps at policy cadence."""
    counter = [0]

    def maybe_tick(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if on_tick is None:
            return
        counter[0] = (counter[0] + 1) % max(tick_period, 1)
        if counter[0] == 0:
            on_tick(model, data)

    if action == "zero":
        if on_tick is None:
            return mujoco.mj_step

        def step_zero(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            mujoco.mj_step(model, data)
            maybe_tick(model, data)

        return step_zero

    if action == "random":

        def step_random(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            data.ctrl[:] = np.random.uniform(lo, hi)
            mujoco.mj_step(model, data)
            maybe_tick(model, data)

        return step_random

    if action == "sine":
        omega = 2.0 * math.pi / sine_period_s

        def step_sine(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            phase = data.time * omega
            mid = 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])
            half = 0.5 * (model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0])
            data.ctrl[:] = mid + half * math.sin(phase)
            mujoco.mj_step(model, data)
            maybe_tick(model, data)

        return step_sine

    raise ValueError(f"Unknown action source: {action!r}")


def _parse_csv_metadata(value: str, *, dtype: type = float) -> list:
    """Decode mjlab's ``list_to_csv_str`` metadata."""
    if not value:
        return []
    return [dtype(x) for x in value.split(",")]


def _resolve_name(model: mujoco.MjModel, obj: int, candidates: list[str]) -> int:
    for name in candidates:
        idx = mujoco.mj_name2id(model, obj, name)
        if idx >= 0:
            return idx
    raise KeyError(f"None of {candidates!r} found in model (obj_type={obj})")


def _find_cube_freejoint(model: mujoco.MjModel) -> int:
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name and "cube" in name:
                return jid
    raise RuntimeError("No cube freejoint found in compiled model")


def _find_cube_body(model: mujoco.MjModel) -> int:
    cube_jid = _find_cube_freejoint(model)
    return int(model.jnt_bodyid[cube_jid])


CameraDataType = Literal["rgb", "depth"]


def _scene_option_for_groups(geom_groups: tuple[int, ...]) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in geom_groups:
        if 0 <= g < len(opt.geomgroup):
            opt.geomgroup[g] = 1
    return opt


@dataclass
class _CameraTerm:
    """One policy-camera obs term. ``display_enabled`` gates the panel only;
    the render always runs because the policy consumes it."""

    obs_term_name: str
    sensor_name: str
    camera_name_in_model: str
    data_type: CameraDataType
    height: int
    width: int
    cutoff_distance: float | None  # depth-clamp distance; None for RGB
    enabled_geom_groups: tuple[int, ...]
    renderer: mujoco.Renderer
    scene_option: mujoco.MjvOption
    display_enabled: bool = True


@dataclass
class _VisionCfg:
    obs_input_name: str
    concatenate_dim: int  # axis mjlab concatenates terms along
    cameras: list[_CameraTerm]


# Viewer-only cameras rendered each step; defaults cover both views so the
# panel works regardless of which one the policy was trained on.
DEFAULT_PREVIEW_CAMERAS: tuple[str, ...] = ("robot/front", "robot/wrist")
# Fallback render resolution when no matching CameraSensorCfg exists.
DEFAULT_PREVIEW_CAMERA_HW: tuple[int, int] = (32, 32)
# Depth far cutoff (m) for the depth preview row; matches the Cube-Depth task default.
DEFAULT_PREVIEW_DEPTH_CUTOFF_M: float = 0.5
# Nearest-neighbor upscale target for panel display. 32 -> 96 is a clean 3x.
DISPLAY_LONG_SIDE: int = 96

# mjviser auto-frames at ``3 * model.stat.extent``. The scene terrain pushes
# extent to ~2 m (camera 6 m away) but the SO-101 only spans ~0.4 m, so
# override for a tighter default that frames the robot + cube workspace.
INITIAL_CAMERA_EXTENT_M: float = 0.5


@dataclass
class _PreviewCamera:
    """Viewer-only camera. ``enabled`` gates render + panel together; a
    hidden camera costs nothing.

    ``renderer`` is built lazily on the panel-worker thread because
    ``mujoco.Renderer`` binds a GL context to its constructing thread —
    rendering must happen on the same thread that built it.

    ``mode="rgb"`` renders the colour image; ``mode="depth"`` enables
    ``Renderer.enable_depth_rendering`` and normalises by ``cutoff_m``."""

    label: str
    camera_name_in_model: str
    height: int
    width: int
    enabled_geom_groups: tuple[int, ...]
    scene_option: mujoco.MjvOption
    mode: CameraDataType = "rgb"
    cutoff_m: float = DEFAULT_PREVIEW_DEPTH_CUTOFF_M
    renderer: mujoco.Renderer | None = None
    enabled: bool = True


def _list_named_cameras(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for cid in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        if name:
            names.append(name)
    return names


def _build_preview_cameras(
    model: mujoco.MjModel,
    requested: tuple[str, ...],
    skip_camera_names: set[str],
    env_cfg: ManagerBasedRlEnvCfg | None = None,
    mode: CameraDataType = "rgb",
) -> list[_PreviewCamera]:
    """Build renderers for the requested viewer-only cameras. Skips any
    name in ``skip_camera_names`` (already covered by policy obs) or
    missing from the model. Render resolution mirrors a matching
    ``CameraSensorCfg`` when one exists. ``mode`` selects RGB vs depth
    output; the same MJCF cameras are reused for both."""
    available = set(_list_named_cameras(model))
    sensor_hw: dict[str, tuple[int, int]] = {}
    if env_cfg is not None:
        for s in env_cfg.scene.sensors or ():
            if isinstance(s, CameraSensorCfg) and s.camera_name:
                sensor_hw[s.camera_name] = (int(s.height), int(s.width))
    enabled_geom_groups: tuple[int, ...] = (0, 2, 3)
    cameras: list[_PreviewCamera] = []
    for cam_name in requested:
        if cam_name in skip_camera_names:
            continue
        if cam_name not in available:
            logger.info("Preview camera %r not found in model; skipping.", cam_name)
            continue
        height, width = sensor_hw.get(cam_name, DEFAULT_PREVIEW_CAMERA_HW)
        cameras.append(
            _PreviewCamera(
                label=cam_name.rsplit("/", 1)[-1],
                camera_name_in_model=cam_name,
                height=height,
                width=width,
                enabled_geom_groups=enabled_geom_groups,
                scene_option=_scene_option_for_groups(enabled_geom_groups),
                mode=mode,
            )
        )
    return cameras


def _ensure_preview_renderer(cam: _PreviewCamera, model: mujoco.MjModel) -> mujoco.Renderer:
    """Build ``cam.renderer`` on the calling thread the first time it's
    needed. Must be called from the panel-worker thread so the GL context
    is created there."""
    if cam.renderer is None:
        renderer = mujoco.Renderer(model, height=cam.height, width=cam.width)
        if cam.mode == "depth":
            renderer.enable_depth_rendering()
        cam.renderer = renderer
    return cam.renderer


def _resolve_obs_term_func_name(func: Any) -> str:
    return getattr(func, "__name__", repr(func))


def _build_camera_term(
    model: mujoco.MjModel,
    obs_term_name: str,
    term: Any,
    sensors: tuple[CameraSensorCfg, ...] | tuple,
) -> _CameraTerm:
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

    cam_sensor: CameraSensorCfg | None = next(
        (s for s in sensors if isinstance(s, CameraSensorCfg) and s.name == sensor_name),
        None,
    )
    if cam_sensor is None:
        available = [getattr(s, "name", repr(s)) for s in sensors]
        raise RuntimeError(
            f"CameraSensorCfg for sensor_name={sensor_name!r} not found in scene.sensors; "
            f"available: {available}."
        )

    height = int(cam_sensor.height)
    width = int(cam_sensor.width)
    enabled_geom_groups = tuple(cam_sensor.enabled_geom_groups)
    renderer = mujoco.Renderer(model, height=height, width=width)
    if data_type == "depth":
        renderer.enable_depth_rendering()

    return _CameraTerm(
        obs_term_name=obs_term_name,
        sensor_name=sensor_name,
        camera_name_in_model=cam_sensor.camera_name or sensor_name,
        data_type=data_type,
        height=height,
        width=width,
        cutoff_distance=cutoff,
        enabled_geom_groups=enabled_geom_groups,
        renderer=renderer,
        scene_option=_scene_option_for_groups(enabled_geom_groups),
    )


def _vision_cfg_from_env_cfg(
    env_cfg: ManagerBasedRlEnvCfg,
    model: mujoco.MjModel,
    ort_input_names: list[str],
) -> _VisionCfg | None:
    """Per-camera renderers + metadata for vision tasks. Returns ``None``
    for blind tasks (no ``camera`` obs group)."""
    if "camera" not in env_cfg.observations:
        return None
    if "camera" not in ort_input_names:
        if len(ort_input_names) < 2:
            raise RuntimeError(
                f"Task has a 'camera' obs group but the ONNX has only {len(ort_input_names)} "
                "input(s); pass an ONNX exported from the matching vision task."
            )
        raise RuntimeError(
            f"ONNX inputs {ort_input_names!r} do not include 'camera'; expected mjlab >= 1.3 vision export."
        )

    camera_group = env_cfg.observations["camera"]
    sensors = env_cfg.scene.sensors or ()
    cameras = [
        _build_camera_term(model, obs_term_name, term, sensors)
        for obs_term_name, term in camera_group.terms.items()
    ]
    if not cameras:
        return None
    # Channel-axis concatenation requires a uniform data_type.
    if len({c.data_type for c in cameras}) > 1:
        raise NotImplementedError(
            "Native vision path requires all camera terms to share a data_type; "
            f"got {[c.data_type for c in cameras]}."
        )
    return _VisionCfg(
        obs_input_name="camera",
        concatenate_dim=int(getattr(camera_group, "concatenate_dim", -1)),
        cameras=cameras,
    )


def _render_camera_obs(cam: _CameraTerm, data: mujoco.MjData) -> np.ndarray:
    """Render one camera with mjlab's normalization. RGB → ``(3,H,W)`` float
    in ``[0,1]``; depth → ``(1,H,W)`` float clamped to ``[0,1]`` via
    ``cutoff_distance``."""
    cam.renderer.update_scene(data, camera=cam.camera_name_in_model, scene_option=cam.scene_option)
    frame = cam.renderer.render()
    if cam.data_type == "rgb":
        return frame.transpose(2, 0, 1).astype(np.float32) / 255.0
    cutoff = cam.cutoff_distance or 1.0
    return np.clip(frame / cutoff, 0.0, 1.0).astype(np.float32)[None, :, :]


def _stack_camera_obs(vision: _VisionCfg, frames: list[np.ndarray]) -> np.ndarray:
    """Concatenate ``(C,H,W)`` frames along the obs-group's concat dim.
    Negative ``concatenate_dim`` matches mjlab's manager (post-batch axis)."""
    if len(frames) == 1:
        return frames[0]
    dim = vision.concatenate_dim
    if dim < 0:
        dim = frames[0].ndim + dim
    return np.concatenate(frames, axis=dim)


def _depth_to_rgb_for_display(frame_chw: np.ndarray) -> np.ndarray:
    g = np.clip(frame_chw[0], 0.0, 1.0)
    return (np.repeat(g[:, :, None], 3, axis=2) * 255.0).astype(np.uint8)


def _depth_render_to_display_rgb(depth_hw: np.ndarray, cutoff_m: float) -> np.ndarray:
    """``mujoco.Renderer.render()`` after ``enable_depth_rendering()`` returns
    an ``(H, W)`` float depth in meters; normalise by ``cutoff_m`` and tile
    to greyscale RGB for the panel."""
    g = np.clip(depth_hw / max(cutoff_m, 1e-6), 0.0, 1.0)
    return (np.repeat(g[:, :, None], 3, axis=2) * 255.0).astype(np.uint8)


def _rgb_obs_to_display(frame_chw: np.ndarray) -> np.ndarray:
    return (np.clip(frame_chw.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)


def _upscale_for_panel(frame_hwc: np.ndarray, long_side: int = DISPLAY_LONG_SIDE) -> np.ndarray:
    """Nearest-neighbor upscale to the largest integer factor that keeps the
    longest side ``<= long_side``. Cheaper than rendering at panel resolution
    and pixel-exact for clean ratios."""
    h, w = frame_hwc.shape[:2]
    factor = max(1, long_side // max(h, w))
    if factor == 1:
        return frame_hwc
    return np.repeat(np.repeat(frame_hwc, factor, axis=0), factor, axis=1)


@dataclass
class _RewardCtx:
    """Native reward callables, mirroring ``env_cfg.rewards`` so the viewer
    can plot without instantiating the manager-based env."""

    names: list[str]
    weights: list[float]
    fns: list[Callable[[mujoco.MjModel, mujoco.MjData, _PolicyCtx], float]]
    history_t: deque[float] = field(default_factory=lambda: deque(maxlen=REWARD_PLOT_HISTORY))
    history_per_term: list[deque[float]] = field(default_factory=list)
    history_total: deque[float] = field(default_factory=lambda: deque(maxlen=REWARD_PLOT_HISTORY))
    # Running [min, max] since last reset, per term and for the total. All
    # reward plots key their y-axis off these instead of uplot autoscale.
    per_term_min: list[float] = field(default_factory=list)
    per_term_max: list[float] = field(default_factory=list)
    total_min: float = field(default_factory=lambda: float("inf"))
    total_max: float = field(default_factory=lambda: float("-inf"))
    # First-call diagnostic: prints the raw + weighted value of each term once
    # so it's easy to confirm a "flat zero" plot really is from a zero signal.
    diag_logged: bool = False


def _build_reward_ctx(env_cfg: ManagerBasedRlEnvCfg) -> _RewardCtx:
    names: list[str] = []
    weights: list[float] = []
    fns: list[Callable[..., float]] = []

    for name, term in env_cfg.rewards.items():
        func_name = _resolve_obs_term_func_name(term.func)
        weight = float(term.weight)
        if func_name == "staged_position_reward":
            reaching_std = float(term.params["reaching_std"])
            bringing_std = float(term.params["bringing_std"])
            fn = _make_staged_position_fn(reaching_std, bringing_std)
        elif func_name == "bring_object_reward":
            std = float(term.params["std"])
            fn = _make_bring_object_fn(std)
        elif func_name == "action_rate_l2":
            fn = _action_rate_l2
        elif func_name == "joint_pos_limits":
            fn = _joint_pos_limits
        elif func_name == "joint_velocity_hinge_penalty":
            max_vel = float(term.params["max_vel"])
            fn = _make_joint_vel_hinge_fn(max_vel)
        elif func_name == "object_is_lifted":
            minimal_height = float(term.params["minimal_height"])
            fn = _make_object_is_lifted_fn(minimal_height)
        else:
            logger.warning(
                "Reward term %r (func=%s) not implemented natively; plotted as 0.",
                name,
                func_name,
            )
            fn = _zero_reward
        names.append(name)
        weights.append(weight)
        fns.append(fn)

    history = [deque(maxlen=REWARD_PLOT_HISTORY) for _ in names]
    per_term_min = [float("inf")] * len(names)
    per_term_max = [float("-inf")] * len(names)
    return _RewardCtx(
        names=names,
        weights=weights,
        fns=fns,
        history_per_term=history,
        per_term_min=per_term_min,
        per_term_max=per_term_max,
    )


def _make_staged_position_fn(
    reaching_std: float, bringing_std: float
) -> Callable[[mujoco.MjModel, mujoco.MjData, _PolicyCtx], float]:
    rs2 = reaching_std**2
    bs2 = bringing_std**2

    def fn(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> float:
        ee_pos = np.asarray(data.site_xpos[ctx.ee_site_id], dtype=np.float64)
        cube_pos = np.asarray(data.xpos[ctx.cube_body_id], dtype=np.float64)
        reach = float(np.exp(-np.square(ee_pos - cube_pos).sum() / rs2))
        bring = float(np.exp(-np.square(ctx.target_pos - cube_pos).sum() / bs2))
        return reach * (1.0 + bring)

    return fn


def _make_bring_object_fn(
    std: float,
) -> Callable[[mujoco.MjModel, mujoco.MjData, _PolicyCtx], float]:
    s2 = std**2

    def fn(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> float:
        cube_pos = np.asarray(data.xpos[ctx.cube_body_id], dtype=np.float64)
        return float(np.exp(-np.square(ctx.target_pos - cube_pos).sum() / s2))

    return fn


def _action_rate_l2(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> float:
    return float(np.square(ctx.last_action - ctx.prev_action).sum())


def _joint_pos_limits(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> float:
    qpos = data.qpos[ctx.joint_qpos_adrs]
    lo = ctx.joint_soft_lo
    hi = ctx.joint_soft_hi
    over_lo = np.clip(-(qpos - lo), a_min=0.0, a_max=None)
    over_hi = np.clip(qpos - hi, a_min=0.0, a_max=None)
    return float((over_lo + over_hi).sum())


def _make_joint_vel_hinge_fn(
    max_vel: float,
) -> Callable[[mujoco.MjModel, mujoco.MjData, _PolicyCtx], float]:
    def fn(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> float:
        qvel = data.qvel[ctx.joint_qvel_adrs]
        excess = np.clip(np.abs(qvel) - max_vel, a_min=0.0, a_max=None)
        return float(np.square(excess).sum())

    return fn


def _make_object_is_lifted_fn(
    minimal_height: float,
) -> Callable[[mujoco.MjModel, mujoco.MjData, _PolicyCtx], float]:
    def fn(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> float:
        return 1.0 if float(data.xpos[ctx.cube_body_id, 2]) > minimal_height else 0.0

    return fn


def _zero_reward(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> float:
    return 0.0


@dataclass
class _PolicyCtx:
    """Indices and state for running an ONNX Cube-task policy. The
    ``_CubeSampler`` is shared with the reset path so cube/goal stay
    coherent across scripted and policy modes."""

    session: ort.InferenceSession | None
    obs_names: list[str]
    state_input_name: str
    action_scale: np.ndarray  # (6,)
    default_joint_pos: np.ndarray  # (6,)
    joint_qpos_adrs: np.ndarray  # (6,)
    joint_qvel_adrs: np.ndarray  # (6,)
    joint_soft_lo: np.ndarray  # (6,)
    joint_soft_hi: np.ndarray  # (6,)
    actuator_ids: np.ndarray  # (6,)
    ee_site_id: int
    decimation: int
    sampler: _CubeSampler
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    vision: _VisionCfg | None = None
    decimation_counter: int = 0
    cached_ctrl: np.ndarray | None = None

    @property
    def cube_body_id(self) -> int:
        return self.sampler.cube_body_id

    @property
    def cube_qpos_adr(self) -> int:
        return self.sampler.cube_qpos_adr

    @property
    def cube_qvel_adr(self) -> int:
        return self.sampler.cube_qvel_adr

    @property
    def target_pos(self) -> np.ndarray:
        return self.sampler.target_pos


def _command_cfg(env_cfg: ManagerBasedRlEnvCfg) -> Any:
    if env_cfg.commands is None or "lift_height" not in env_cfg.commands:
        raise RuntimeError("env_cfg has no 'lift_height' command; vision/blind cube task expected.")
    return env_cfg.commands["lift_height"]


@dataclass
class _CubeSampler:
    """Native ``LiftingCommand`` clone. Lives outside ``_PolicyCtx`` so the
    cube + goal still resample under scripted action modes; without it the
    cube spawns at the keyframe origin, inside the SO-101 base footprint."""

    cube_body_id: int
    cube_qpos_adr: int  # qpos slice: x, y, z, qw, qx, qy, qz
    cube_qvel_adr: int  # qvel slice: 6 fields
    object_range_lo: np.ndarray  # (3,) world XYZ
    object_range_hi: np.ndarray
    object_yaw_lo: float
    object_yaw_hi: float
    target_range_lo: np.ndarray  # (3,) world XYZ
    target_range_hi: np.ndarray
    command_resample_period_s: float
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    next_resample_time: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


def _build_cube_sampler(env_cfg: ManagerBasedRlEnvCfg, model: mujoco.MjModel, seed: int) -> _CubeSampler:
    cmd = _command_cfg(env_cfg)
    obj_r = cmd.object_pose_range
    if obj_r is None:
        raise RuntimeError(
            "lift_height command has object_pose_range=None; native preview cannot resample without bounds."
        )
    object_range_lo = np.array([obj_r.x[0], obj_r.y[0], obj_r.z[0]], dtype=np.float64)
    object_range_hi = np.array([obj_r.x[1], obj_r.y[1], obj_r.z[1]], dtype=np.float64)
    yaw_lo, yaw_hi = float(obj_r.yaw[0]), float(obj_r.yaw[1])

    if getattr(cmd, "difficulty", "dynamic") == "fixed":
        target_range_lo = np.array([0.4, 0.0, 0.3], dtype=np.float64)
        target_range_hi = target_range_lo.copy()
    else:
        tr = cmd.target_position_range
        target_range_lo = np.array([tr.x[0], tr.y[0], tr.z[0]], dtype=np.float64)
        target_range_hi = np.array([tr.x[1], tr.y[1], tr.z[1]], dtype=np.float64)

    resample_lo, resample_hi = cmd.resampling_time_range
    command_resample_period_s = float(0.5 * (resample_lo + resample_hi))

    cube_jid = _find_cube_freejoint(model)
    return _CubeSampler(
        cube_body_id=_find_cube_body(model),
        cube_qpos_adr=int(model.jnt_qposadr[cube_jid]),
        cube_qvel_adr=int(model.jnt_dofadr[cube_jid]),
        object_range_lo=object_range_lo,
        object_range_hi=object_range_hi,
        object_yaw_lo=yaw_lo,
        object_yaw_hi=yaw_hi,
        target_range_lo=target_range_lo,
        target_range_hi=target_range_hi,
        command_resample_period_s=command_resample_period_s,
        rng=np.random.default_rng(seed),
    )


def _resample_cube(model: mujoco.MjModel, data: mujoco.MjData, sampler: _CubeSampler) -> None:
    """Mirror ``LiftingCommand._resample_command`` against native ``mj_data``:
    writes the cube freejoint pose into ``data.qpos`` and the world-frame
    goal into ``sampler.target_pos``."""
    if np.array_equal(sampler.target_range_lo, sampler.target_range_hi):
        sampler.target_pos[:] = sampler.target_range_lo
    else:
        sampler.target_pos[:] = sampler.rng.uniform(sampler.target_range_lo, sampler.target_range_hi)

    cube_xy_z = sampler.rng.uniform(sampler.object_range_lo, sampler.object_range_hi)
    cube_yaw = sampler.rng.uniform(sampler.object_yaw_lo, sampler.object_yaw_hi)
    half = 0.5 * cube_yaw
    quat_wxyz = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
    qpos_slice = slice(sampler.cube_qpos_adr, sampler.cube_qpos_adr + 7)
    qvel_slice = slice(sampler.cube_qvel_adr, sampler.cube_qvel_adr + 6)
    data.qpos[qpos_slice] = np.concatenate([cube_xy_z, quat_wxyz])
    data.qvel[qvel_slice] = 0.0
    mujoco.mj_forward(model, data)


@dataclass
class _ArmIndices:
    joint_qpos_adrs: np.ndarray
    joint_qvel_adrs: np.ndarray
    actuator_ids: np.ndarray
    joint_soft_lo: np.ndarray
    joint_soft_hi: np.ndarray
    ee_site_id: int


_SOFT_JOINT_POS_LIMIT_FACTOR = 0.95


def _resolve_arm_indices(model: mujoco.MjModel, joint_names: list[str]) -> _ArmIndices:
    """Resolve joint/actuator/ee_site indices for the named joints. Soft
    joint-pos limits are computed as mjlab does on the Entity side: shrink
    each ``jnt_range`` toward its midpoint by ``_SOFT_JOINT_POS_LIMIT_FACTOR``.
    Without this, the actuator ctrl saturates at the joint limit and the
    penalty stays identically zero."""
    qpos_adrs, qvel_adrs, act_ids, soft_lo, soft_hi = [], [], [], [], []
    for j in joint_names:
        jid = _resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, [f"robot/{j}", j])
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        qvel_adrs.append(int(model.jnt_dofadr[jid]))
        act_ids.append(_resolve_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"robot/{j}", j]))
        rng_lo, rng_hi = model.jnt_range[jid]
        if bool(model.jnt_limited[jid]):
            mid = 0.5 * (float(rng_lo) + float(rng_hi))
            half = 0.5 * (float(rng_hi) - float(rng_lo)) * _SOFT_JOINT_POS_LIMIT_FACTOR
            soft_lo.append(mid - half)
            soft_hi.append(mid + half)
        else:
            soft_lo.append(-math.inf)
            soft_hi.append(math.inf)
    return _ArmIndices(
        joint_qpos_adrs=np.array(qpos_adrs, dtype=np.int32),
        joint_qvel_adrs=np.array(qvel_adrs, dtype=np.int32),
        actuator_ids=np.array(act_ids, dtype=np.int32),
        joint_soft_lo=np.array(soft_lo, dtype=np.float64),
        joint_soft_hi=np.array(soft_hi, dtype=np.float64),
        ee_site_id=_resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, ["robot/ee_site", "ee_site"]),
    )


def _build_reward_only_ctx(
    env_cfg: ManagerBasedRlEnvCfg, model: mujoco.MjModel, sampler: _CubeSampler
) -> _PolicyCtx:
    """``_PolicyCtx`` with no ONNX session, just enough for reward plotting in
    scripted mode. Joint names come from the SO-ARM101 constants since no
    ONNX metadata is available."""
    from arm_farm.sim.assets.so101 import ARM_JOINTS

    idx = _resolve_arm_indices(model, list(ARM_JOINTS))
    return _PolicyCtx(
        session=None,
        obs_names=[],
        state_input_name="",
        action_scale=np.zeros(len(ARM_JOINTS), dtype=np.float32),
        default_joint_pos=np.zeros(len(ARM_JOINTS), dtype=np.float64),
        joint_qpos_adrs=idx.joint_qpos_adrs,
        joint_qvel_adrs=idx.joint_qvel_adrs,
        joint_soft_lo=idx.joint_soft_lo,
        joint_soft_hi=idx.joint_soft_hi,
        actuator_ids=idx.actuator_ids,
        ee_site_id=idx.ee_site_id,
        decimation=int(env_cfg.decimation),
        sampler=sampler,
    )


def _build_policy_context(
    onnx_path: Path,
    model: mujoco.MjModel,
    sampler: _CubeSampler,
    env_cfg: ManagerBasedRlEnvCfg,
) -> _PolicyCtx:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

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

    idx = _resolve_arm_indices(model, joint_names)
    ort_input_names = [inp.name for inp in session.get_inputs()]
    vision = _vision_cfg_from_env_cfg(env_cfg, model, ort_input_names)
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
        joint_qpos_adrs=idx.joint_qpos_adrs,
        joint_qvel_adrs=idx.joint_qvel_adrs,
        joint_soft_lo=idx.joint_soft_lo,
        joint_soft_hi=idx.joint_soft_hi,
        actuator_ids=idx.actuator_ids,
        ee_site_id=idx.ee_site_id,
        decimation=int(env_cfg.decimation),
        sampler=sampler,
        vision=vision,
    )


def _ee_frame(data: mujoco.MjData, ee_site_id: int) -> tuple[np.ndarray, np.ndarray]:
    ee_pos = np.asarray(data.site_xpos[ee_site_id], dtype=np.float64)
    ee_xmat = np.asarray(data.site_xmat[ee_site_id], dtype=np.float64).reshape(3, 3)
    return ee_pos, ee_xmat


def _compute_actor_obs(model: mujoco.MjModel, data: mujoco.MjData, ctx: _PolicyCtx) -> np.ndarray:
    """Native equivalent of mjlab's Cube-task obs manager."""
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


def _query_policy_and_apply(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: _PolicyCtx,
    on_camera_obs: Callable[[list[np.ndarray]], None] | None,
    on_reward: Callable[[float], None] | None,
) -> None:
    """One policy inference, writing ``data.ctrl``. The caller drives ``mj_step``."""
    sampler = ctx.sampler
    if data.time >= sampler.next_resample_time:
        _resample_cube(model, data, sampler)
        sampler.next_resample_time = data.time + sampler.command_resample_period_s

    inputs: dict[str, np.ndarray] = {
        ctx.state_input_name: _compute_actor_obs(model, data, ctx)[None, :],
    }
    rendered_frames: list[np.ndarray] | None = None
    if ctx.vision is not None:
        rendered_frames = [_render_camera_obs(c, data) for c in ctx.vision.cameras]
        cam = _stack_camera_obs(ctx.vision, rendered_frames)
        inputs[ctx.vision.obs_input_name] = cam[None, :]

    action = ctx.session.run(None, inputs)[0][0]
    ctx.prev_action[:] = ctx.last_action
    ctx.last_action[:] = action

    target = action.astype(np.float64) * ctx.action_scale + ctx.default_joint_pos
    data.ctrl[ctx.actuator_ids] = target
    ctx.cached_ctrl = data.ctrl.copy()

    if rendered_frames is not None and on_camera_obs is not None:
        on_camera_obs(rendered_frames)
    if on_reward is not None:
        on_reward(data.time)


def _make_policy_step_fn(
    ctx: _PolicyCtx,
    on_camera_obs: Callable[[list[np.ndarray]], None] | None = None,
    on_policy_step: Callable[[float], None] | None = None,
    on_tick: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
) -> StepFn:
    """ONNX inference + JointPositionAction wrapped as a ``step_fn``. Honours
    ``env_cfg.decimation`` by caching ctrl across the intermediate
    ``mj_step`` calls. ``on_tick`` fires once per policy decision."""

    def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        is_policy_tick = ctx.decimation_counter == 0
        if is_policy_tick:
            _query_policy_and_apply(model, data, ctx, on_camera_obs, on_policy_step)
        elif ctx.cached_ctrl is not None:
            data.ctrl[:] = ctx.cached_ctrl
        ctx.decimation_counter = (ctx.decimation_counter + 1) % max(ctx.decimation, 1)
        mujoco.mj_step(model, data)
        if is_policy_tick and on_tick is not None:
            on_tick(model, data)

    return step


def _make_reset_fn(
    sampler: _CubeSampler | None = None,
    ctx_holder: _PolicyCtxHolder | None = None,
) -> ResetFn:
    """Reset to the ``init_state`` keyframe; resample cube + goal when
    ``sampler`` is set; clear per-episode policy state when ``ctx_holder.ctx``
    is set. The sampler-less form is for tests."""

    def reset_fn(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, INIT_STATE_KEYFRAME)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        if sampler is not None:
            _resample_cube(model, data, sampler)
            sampler.next_resample_time = data.time + sampler.command_resample_period_s
        if ctx_holder is not None and ctx_holder.ctx is not None:
            ctx = ctx_holder.ctx
            ctx.decimation_counter = 0
            ctx.cached_ctrl = None
            ctx.last_action[:] = 0.0
            ctx.prev_action[:] = 0.0

    return reset_fn


@dataclass
class _PolicyCtxHolder:
    """Mutable holder so callbacks can swap in a new policy at runtime."""

    ctx: _PolicyCtx | None = None


_WANDB_CACHE_DIR = Path.home() / ".cache" / "arm_farm" / "wandb_onnx"


def _wandb_lookup_help(api, run_path: str) -> str:
    parts = run_path.split("/")
    if len(parts) != 3:
        return f"Expected entity/project/run_id form; got {run_path!r}."
    entity, project, _ = parts
    try:
        runs = list(api.runs(f"{entity}/{project}", per_page=50))
    except Exception:
        runs = None
    if runs:
        lines = [f"Available runs in {entity}/{project} (id  name  state):"]
        lines.extend(f"  {r.id}  {r.name!r}  {r.state}" for r in runs[:50])
        if len(runs) > 50:
            lines.append(f"  ... ({len(runs) - 50} more)")
        return "\n".join(lines)
    try:
        projects = [p.name for p in api.projects(entity)]
    except Exception:
        projects = None
    if projects is not None:
        return (
            f"Project {project!r} is empty or not visible to this account. "
            f"Available projects in {entity}: {projects}"
        )
    return (
        f"Could not list anything under entity {entity!r}. Verify your wandb "
        "login (e.g. `wandb login --verify`) and that this account is a member "
        f"of {entity!r}."
    )


def _list_wandb_checkpoint_files(run_path: str) -> list[str]:
    """Sorted ``.onnx``/``.pt`` filenames in a wandb run. ``.pt`` is included
    so the dropdown can flag it as needing offline conversion."""
    import wandb
    from wandb.errors import CommError

    api = wandb.Api()
    try:
        run = api.run(run_path)
    except (CommError, ValueError) as e:
        raise RuntimeError(f"wandb run {run_path!r} not found.\n{_wandb_lookup_help(api, run_path)}") from e
    files = [f.name for f in run.files() if f.name.endswith((".onnx", ".pt"))]
    if not files:
        raise RuntimeError(
            f"No .onnx or .pt files in wandb run {run_path!r}. mjlab uploads "
            "checkpoints only when --agent.upload-model True (the default) AND "
            "a save_interval has fired at least once."
        )
    return sorted(files)


def _download_wandb_file(run_path: str, name: str, cache_dir: Path = _WANDB_CACHE_DIR) -> Path:
    import wandb
    from wandb.errors import CommError

    api = wandb.Api()
    try:
        run = api.run(run_path)
    except (CommError, ValueError) as e:
        raise RuntimeError(f"wandb run {run_path!r} not found.\n{_wandb_lookup_help(api, run_path)}") from e
    run_id = run_path.rsplit("/", 1)[-1]
    cache_subdir = cache_dir / run_id
    cache_subdir.mkdir(parents=True, exist_ok=True)

    matches = [f for f in run.files() if f.name == name or Path(f.name).name == name]
    if not matches:
        all_files = sorted(f.name for f in run.files())
        raise RuntimeError(f"File {name!r} not in run {run_path!r}. Available: {all_files}")
    chosen = matches[0]
    local_path = cache_subdir / Path(chosen.name).name
    if not local_path.exists():
        logger.info("Downloading %s from wandb run %s -> %s", chosen.name, run_path, local_path)
        chosen.download(root=str(cache_subdir), replace=True)
        downloaded = cache_subdir / chosen.name
        if downloaded != local_path and downloaded.exists():
            downloaded.rename(local_path)
    else:
        logger.info("Using cached wandb file at %s", local_path)
    return local_path


def _resolve_wandb_onnx(
    run_path: str, onnx_name: str | None = None, cache_dir: Path = _WANDB_CACHE_DIR
) -> Path:
    """Download (or reuse cached) ONNX policy from a wandb run. Auto-picks
    when the run contains exactly one. ``.pt``-only runs need ``convert2onnx``
    first."""
    if onnx_name is None:
        files = _list_wandb_checkpoint_files(run_path)
        onnx_files = [f for f in files if f.endswith(".onnx")]
        if not onnx_files:
            raise RuntimeError(
                f"Run {run_path!r} has no .onnx files. Use "
                "`convert2onnx <run-path>` to convert "
                "the .pt checkpoints once."
            )
        if len(onnx_files) > 1:
            raise RuntimeError(
                f"Run {run_path!r} has multiple .onnx files: {onnx_files}. "
                "Pass --wandb-onnx-name to disambiguate."
            )
        onnx_name = onnx_files[0]
    return _download_wandb_file(run_path, onnx_name, cache_dir=cache_dir)


# Viser's GUI panel is a vertical stack — there's no row container — so
# camera rows are emitted as a single ``add_html`` element wrapping a
# flexbox of <img> tags with inline base64 PNGs. The full HTML rebuilds
# every tick; encoding two 96x96 frames is well under the 50 Hz budget.


def _png_data_uri(frame_hwc: np.ndarray) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame_hwc).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _row_html(items: list[tuple[str, np.ndarray]]) -> str:
    """Render ``[(label, frame_hwc), ...]`` as a flex row of stacked tiles."""
    if not items:
        return ""
    tiles = []
    for label, frame in items:
        uri = _png_data_uri(frame)
        tiles.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'gap:0.25em;font-size:0.8em;">'
            f'<img src="{uri}" style="image-rendering:pixelated;max-width:100%;" />'
            f"<span>{label}</span></div>"
        )
    return (
        '<div style="display:flex;flex-direction:row;gap:0.5em;'
        'justify-content:space-around;flex-wrap:wrap;width:100%;">' + "".join(tiles) + "</div>"
    )


def _add_camera_panels(server, vision: _VisionCfg) -> dict[str, Any]:
    """Camera-obs row: one HTML element with all enabled cameras side by side.

    Per-camera checkboxes toggle ``cam.display_enabled`` to gate the panel
    push; the render itself always runs because the policy consumes it.
    Cameras start hidden so the panel doesn't appear until opted into.
    """
    state: dict[str, Any] = {}
    with server.gui.add_folder("Camera obs", expand_by_default=False):
        for cam in vision.cameras:
            cam.display_enabled = False
            cb = server.gui.add_checkbox(f"Show {cam.obs_term_name}", initial_value=False)

            def _on_toggle(_, _cam=cam, _cb=cb) -> None:
                _cam.display_enabled = bool(_cb.value)

            cb.on_update(_on_toggle)
        state["_html"] = server.gui.add_html(_row_html([]))
    state["_cameras"] = vision.cameras
    return state


def _push_camera_frames(handles: dict[str, Any], vision: _VisionCfg, frames: list[np.ndarray]) -> None:
    if "_html" not in handles:
        return
    items: list[tuple[str, np.ndarray]] = []
    for cam, frame in zip(vision.cameras, frames, strict=True):
        if not cam.display_enabled:
            continue
        display = _rgb_obs_to_display(frame) if cam.data_type == "rgb" else _depth_to_rgb_for_display(frame)
        items.append((cam.obs_term_name, _upscale_for_panel(display)))
    handles["_html"].content = _row_html(items)


def _add_preview_camera_panels(
    server, preview_cameras: list[_PreviewCamera], folder_label: str = "Camera preview"
) -> dict[str, Any]:
    """Preview-camera row: same flex layout as ``_add_camera_panels``.

    Unlike obs cameras, ``cam.enabled`` here gates both the render and the
    panel push, so disabled previews cost nothing.
    """
    state: dict[str, Any] = {}
    if not preview_cameras:
        return state
    with server.gui.add_folder(folder_label, expand_by_default=False):
        for cam in preview_cameras:
            cam.enabled = False
            cb = server.gui.add_checkbox(f"Show {cam.label}", initial_value=False)

            def _on_toggle(_, _cam=cam, _cb=cb) -> None:
                _cam.enabled = bool(_cb.value)

            cb.on_update(_on_toggle)
        state["_html"] = server.gui.add_html(_row_html([]))
    return state


_PLACEHOLDER_ONNX_OPTION = "(no checkpoints)"


def _add_checkpoint_dropdown(
    server,
    run_path: str,
    initial_onnx_name: str | None,
    initial_loaded_name: str | None,
    on_select: Callable[[str], None],
) -> None:
    """Wandb checkpoint dropdown: selection auto-loads, Refresh re-queries
    wandb, and a failed fetch still renders the panel with the error inline."""
    with server.gui.add_folder("Checkpoint", expand_by_default=False):
        server.gui.add_markdown(f"**Run:** `{run_path}`")

        try:
            files = _list_wandb_checkpoint_files(run_path)
            initial = initial_onnx_name if initial_onnx_name in files else files[0]
            initial_status = (
                f"Loaded `{initial_loaded_name}`."
                if initial_loaded_name in files
                else f"Available: {len(files)}. Pick one to load."
            )
            dropdown_options: list[str] = list(files)
            dropdown_initial = initial
        except Exception as e:
            logger.warning("Could not fetch wandb checkpoint list: %s", e)
            initial_status = f"**Could not fetch checkpoint list:** {e}"
            dropdown_options = [_PLACEHOLDER_ONNX_OPTION]
            dropdown_initial = _PLACEHOLDER_ONNX_OPTION

        dropdown = server.gui.add_dropdown(
            "ONNX file", options=dropdown_options, initial_value=dropdown_initial
        )
        status = server.gui.add_markdown(initial_status)
        refresh_btn = server.gui.add_button("Refresh list")

        # Mutating ``dropdown.options`` fires on_update; track the last
        # loaded name so refresh-driven fires don't redownload.
        last_loaded = [initial_loaded_name or dropdown.value]

        def _do_load(name: str) -> None:
            if name == _PLACEHOLDER_ONNX_OPTION:
                return
            status.content = f"Loading `{name}`..."
            try:
                on_select(name)
            except Exception as e:
                logger.exception("Checkpoint load failed: %s", e)
                status.content = f"**Error:** {e}"
            else:
                last_loaded[0] = name
                status.content = f"Loaded `{name}`."

        @dropdown.on_update
        def _(_) -> None:
            if dropdown.value != last_loaded[0]:
                _do_load(dropdown.value)

        @refresh_btn.on_click
        def _(_) -> None:
            try:
                new_files = _list_wandb_checkpoint_files(run_path)
            except Exception as e:
                logger.exception("Checkpoint refresh failed: %s", e)
                status.content = f"**Error refreshing:** {e}"
                return
            current = dropdown.value
            dropdown.options = tuple(new_files)
            if current in new_files:
                dropdown.value = current
            else:
                dropdown.value = new_files[0]
            status.content = f"Refreshed: {len(new_files)} checkpoint(s) available."


_REWARD_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _add_reward_plot(server, reward_ctx: _RewardCtx, snapshot_lock: threading.Lock) -> Callable[[], None]:
    """Total weighted reward at the top, then one uplot per term in its own
    collapsible sub-folder. Returns a ``refresh()`` that snapshots history
    under ``snapshot_lock`` so the sim-thread writer can't extend the deques
    mid-read."""
    with server.gui.add_folder("Rewards", expand_by_default=False):
        total_plot = _add_one_reward_plot(server, "total (weighted)", 0)
        plots: list[Any] = []
        for i, name in enumerate(reward_ctx.names):
            with server.gui.add_folder(name, expand_by_default=False):
                # +1 so the total plot keeps colour 0 to itself.
                plots.append(_add_one_reward_plot(server, name, i + 1))

    def refresh() -> None:
        with snapshot_lock:
            if not reward_ctx.history_t:
                return
            t = np.asarray(reward_ctx.history_t, dtype=np.float64)
            total = np.asarray(reward_ctx.history_total, dtype=np.float64)
            ys = [np.asarray(h, dtype=np.float64) for h in reward_ctx.history_per_term]
            total_min = reward_ctx.total_min
            total_max = reward_ctx.total_max
            term_mins = list(reward_ctx.per_term_min)
            term_maxs = list(reward_ctx.per_term_max)
        total_plot.data = (t, total)
        scales, axes = _manual_y_axis(total_min, total_max)
        total_plot.scales = scales
        total_plot.axes = axes
        for plot, y, ymin, ymax in zip(plots, ys, term_mins, term_maxs, strict=True):
            plot.data = (t, y)
            scales, axes = _manual_y_axis(ymin, ymax)
            plot.scales = scales
            plot.axes = axes

    return refresh


def _format_y_tick(v: float) -> str:
    """Compact tick label: scientific for tiny / huge magnitudes so the axis
    column doesn't blow up in the GUI; fixed-point otherwise."""
    if v == 0.0:
        return "0"
    mag = abs(v)
    if mag < 1e-3 or mag >= 1e4:
        return f"{v:.2e}"
    if mag < 1.0:
        return f"{v:.3f}"
    if mag < 100.0:
        return f"{v:.2f}"
    return f"{v:.0f}"


def _manual_y_axis(ymin: float, ymax: float) -> tuple[dict[str, dict[str, Any]], tuple[Any, ...]]:
    """Build uplot ``scales`` + ``axes`` config that pins the y-axis to
    ``[ymin, ymax]`` padded by ~10% of the span and labels three ticks at
    ``ymin``, ``midpoint``, and ``ymax``. Falls back to ``[-1e-6, 1e-6]``
    while no samples have arrived (running min/max still at ±inf) and to
    ``v ± pad`` when the window is exactly constant — uplot collapses
    degenerate ranges so the line vanishes at the panel edge otherwise."""
    import viser.uplot as uplot

    if not (math.isfinite(ymin) and math.isfinite(ymax)):
        ymin, ymax = -1e-6, 1e-6
    span = ymax - ymin
    pad = max(abs(ymax) * 0.1, 1e-6) if span <= 0.0 else 0.1 * span
    lo, hi = ymin - pad, ymax + pad
    mid = 0.5 * (ymin + ymax)
    splits = (ymin, mid, ymax)
    values = tuple(_format_y_tick(v) for v in splits)
    scales: dict[str, dict[str, Any]] = {
        "x": {"time": False, "auto": True},
        "y": {"auto": False, "range": (lo, hi)},
    }
    axes = (
        uplot.Axis(scale="x", show=False),
        uplot.Axis(scale="y", splits=splits, values=values),
    )
    return scales, axes


def _add_one_reward_plot(server, name: str, idx: int) -> Any:
    import viser.uplot as uplot

    return server.gui.add_uplot(
        data=(np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)),
        series=(
            uplot.Series(label="t", scale="x", show=False),
            uplot.Series(
                label=name,
                scale="y",
                stroke=_REWARD_PALETTE[idx % len(_REWARD_PALETTE)],
                width=1.5,
            ),
        ),
        # Explicit auto on both scales — uplot otherwise inherits a "y" scale
        # without an auto flag from the first .data assignment, which leaves
        # the y range frozen to the seed point.
        scales={"x": {"time": False, "auto": True}, "y": {"auto": True}},
        legend={"show": False},
        aspect=2.5,
    )


def main(
    task: str = "Cube",
    action: ActionSource = "zero",
    sine_period_s: float = 4.0,
    checkpoint: Path | None = None,
    wandb_run_path: str | None = None,
    wandb_onnx_name: str | None = None,
    seed: int = 0,
    preview_cameras: tuple[str, ...] = DEFAULT_PREVIEW_CAMERAS,
) -> None:
    """Open a native-mujoco rollout of a registered task in the viser viewer.

    Args:
        task: mjlab task ID (``Cube``/``Cube-Rgb``/``Cube-Depth``/``Play``).
        action: scripted source; ignored when a checkpoint is provided.
        sine_period_s: full-sweep period for ``action="sine"``.
        checkpoint: ONNX policy path. Drives the rollout if set.
        wandb_run_path: ``entity/project/run_id``. Exposes a dropdown of every
            ``.onnx`` in the run; ``--checkpoint`` wins on the first frame.
        wandb_onnx_name: dropdown's preselected entry.
        seed: RNG seed for command resampling and random-action mode.
        preview_cameras: viewer-only MJCF camera names; missing or
            obs-shadowed names are skipped.
    """
    if checkpoint is not None and wandb_run_path is not None:
        raise SystemExit("Pass either --checkpoint or --wandb-run-path, not both.")

    # Spin up the viser server before scene compile so the URL banner appears
    # ~1.2 s sooner; the user can navigate to it while the model finishes
    # building. Reused below via ``Viewer(server=server)``.
    import viser

    server = viser.ViserServer()

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1

    scene = Scene(env_cfg.scene, device="cpu")
    model = scene.compile()
    model.stat.extent = INITIAL_CAMERA_EXTENT_M
    data = mujoco.MjData(model)

    holder = _PolicyCtxHolder()
    sim_lock = threading.RLock()  # guards model+data+holder vs GUI mutations
    # Separate lock for reward-history deques so the plot-pump thread can
    # snapshot without interleaving with sim-thread appends.
    reward_lock = threading.Lock()

    sampler = _build_cube_sampler(env_cfg, model, seed=seed)
    inner_reset_fn = _make_reset_fn(sampler=sampler, ctx_holder=holder)
    reward_ctx = _build_reward_ctx(env_cfg)

    def reset_fn(_model: mujoco.MjModel, _data: mujoco.MjData) -> None:
        # Reward history clears on every reset so plots don't carry data
        # across the time discontinuity.
        inner_reset_fn(_model, _data)
        with reward_lock:
            reward_ctx.history_t.clear()
            reward_ctx.history_total.clear()
            for hist in reward_ctx.history_per_term:
                hist.clear()
            reward_ctx.total_min = float("inf")
            reward_ctx.total_max = float("-inf")
            for i in range(len(reward_ctx.per_term_min)):
                reward_ctx.per_term_min[i] = float("inf")
                reward_ctx.per_term_max[i] = float("-inf")

    reset_fn(model, data)

    # Resolve the initial checkpoint (explicit > wandb auto-pick > none).
    # .pt-only wandb runs need a one-time convert2onnx first.
    initial_checkpoint: Path | None = None
    initial_wandb_name: str | None = None
    if checkpoint is not None:
        initial_checkpoint = checkpoint
    elif wandb_run_path is not None:
        try:
            onnx_files = [f for f in _list_wandb_checkpoint_files(wandb_run_path) if f.endswith(".onnx")]
            if not onnx_files:
                raise RuntimeError(
                    f"Wandb run {wandb_run_path!r} has no .onnx files. Run "
                    "`convert2onnx <run-path>` to convert the .pt checkpoints once, then relaunch."
                )
            initial_wandb_name = wandb_onnx_name if wandb_onnx_name in onnx_files else onnx_files[-1]
            initial_checkpoint = _download_wandb_file(wandb_run_path, initial_wandb_name)
        except Exception as e:
            logger.warning("Could not auto-load wandb checkpoint up front: %s", e)

    if initial_checkpoint is not None:
        holder.ctx = _build_policy_context(initial_checkpoint, model, sampler, env_cfg)
        reset_fn(model, data)
    else:
        np.random.seed(seed)

    # Reward plot reads through this holder so swaps (install_policy) keep
    # the same plot updating with a fresh ctx.
    reward_eval_ctx: _PolicyCtx = (
        holder.ctx if holder.ctx is not None else _build_reward_only_ctx(env_cfg, model, sampler)
    )

    def _record_rewards(t: float) -> None:
        # Compute outside the lock; only the deque appends need protection.
        raws = [fn(model, data, reward_eval_ctx) for fn in reward_ctx.fns]
        values = [w * r for r, w in zip(raws, reward_ctx.weights, strict=True)]
        total = float(sum(values))
        with reward_lock:
            reward_ctx.history_t.append(float(t))
            reward_ctx.history_total.append(total)
            for i, (hist, v) in enumerate(zip(reward_ctx.history_per_term, values, strict=True)):
                hist.append(v)
                if v < reward_ctx.per_term_min[i]:
                    reward_ctx.per_term_min[i] = v
                if v > reward_ctx.per_term_max[i]:
                    reward_ctx.per_term_max[i] = v
            reward_ctx.total_min = min(reward_ctx.total_min, total)
            reward_ctx.total_max = max(reward_ctx.total_max, total)
            log_now = not reward_ctx.diag_logged
            reward_ctx.diag_logged = True
        if log_now:
            for name, raw, w, v in zip(reward_ctx.names, raws, reward_ctx.weights, values, strict=True):
                logger.info("reward[%s]: raw=%+.6g  weight=%+.4g  weighted=%+.6g", name, raw, w, v)

    cam_panels: dict[str, Any] = {}
    preview_panels: dict[str, Any] = {}
    preview_cams: list[_PreviewCamera] = []

    # Panel-rendering hand-off: the viewer thread stashes the latest obs
    # frames into ``latest_obs[0]`` and the panel-worker thread renders
    # preview cams + encodes obs frames + pushes HTML off the hot path.
    # Without this, the per-tick render+encode cost CAPs the 60 Hz loop
    # (~28 ms) and drops realtime below 1x.
    latest_obs: list[tuple[_VisionCfg, list[np.ndarray]] | None] = [None]
    panel_lock = threading.Lock()

    def _build_step_fn(ctx: _PolicyCtx | None) -> StepFn:
        if ctx is not None:

            def on_camera_obs(frames: list[np.ndarray], _ctx=ctx) -> None:
                if _ctx.vision is None:
                    return
                with panel_lock:
                    latest_obs[0] = (_ctx.vision, frames)

            return _make_policy_step_fn(
                ctx,
                on_camera_obs=on_camera_obs,
                on_policy_step=_record_rewards,
            )

        # Scripted mode: tick at policy cadence so rewards refresh at a
        # comparable rate to the policy path. ctrl is mirrored into the
        # action shadows so ``action_rate_l2`` reflects ctrl variation.
        def scripted_tick(_model: mujoco.MjModel, _data: mujoco.MjData) -> None:
            ctrl_now = _data.ctrl[reward_eval_ctx.actuator_ids].astype(np.float32)
            reward_eval_ctx.prev_action[:] = reward_eval_ctx.last_action
            reward_eval_ctx.last_action[:] = ctrl_now
            _record_rewards(_data.time)

        return _make_scripted_step_fn(
            action=action,
            sine_period_s=sine_period_s,
            on_tick=scripted_tick,
            tick_period=int(env_cfg.decimation),
        )

    def install_policy(onnx_path: Path) -> None:
        """Build a fresh _PolicyCtx and swap step_fn under the sim lock.
        ``reward_eval_ctx`` is rebound so the existing reward plot keeps
        updating across swaps."""
        nonlocal reward_eval_ctx
        new_ctx = _build_policy_context(onnx_path, model, sampler, env_cfg)
        with sim_lock:
            holder.ctx = new_ctx
            reward_eval_ctx = new_ctx
            reset_fn(model, data)
            viewer._step_fn = _build_step_fn(new_ctx)
        cam_desc = (
            "camera="
            + ",".join(
                f"{c.obs_term_name}({c.data_type} {c.height}x{c.width})" for c in new_ctx.vision.cameras
            )
            if new_ctx.vision is not None
            else "camera=none"
        )
        logger.info(
            "Loaded policy: onnx=%s, obs_names=%s, decimation=%d, %s",
            onnx_path,
            new_ctx.obs_names,
            new_ctx.decimation,
            cam_desc,
        )

    viewer = Viewer(model, data, step_fn=_build_step_fn(holder.ctx), reset_fn=reset_fn, server=server)

    obs_camera_mjcf_names: set[str] = set()
    if holder.ctx is not None and holder.ctx.vision is not None:
        cam_panels = _add_camera_panels(server, holder.ctx.vision)
        obs_camera_mjcf_names = {c.camera_name_in_model for c in holder.ctx.vision.cameras}

    preview_cams = _build_preview_cameras(model, preview_cameras, obs_camera_mjcf_names, env_cfg, mode="rgb")
    preview_panels = _add_preview_camera_panels(server, preview_cams, folder_label="Camera preview (RGB)")
    # Sibling depth row: same MJCF cameras, same toggles, separate renderers
    # (depth-rendering must be enabled on construction, so renderers can't
    # be shared with the RGB row).
    depth_preview_cams = _build_preview_cameras(
        model, preview_cameras, obs_camera_mjcf_names, env_cfg, mode="depth"
    )
    depth_preview_panels = _add_preview_camera_panels(
        server, depth_preview_cams, folder_label="Camera preview (depth)"
    )
    # No seed render: the panel worker thread fills the panel within its
    # first iteration. Seeding here would create the GL context on the
    # wrong thread and force preview rendering back onto the viewer hot
    # path on macOS.

    refresh_reward_plot = _add_reward_plot(server, reward_ctx, reward_lock)

    # Build the dropdown even on fetch failure so the user can see the error
    # and hit Refresh once auth is sorted.
    if wandb_run_path is not None:

        def _on_select(checkpoint_name: str) -> None:
            if not checkpoint_name.endswith(".onnx"):
                raise RuntimeError(
                    f"{checkpoint_name!r} is a .pt file. Convert it once with "
                    "`convert2onnx <run-path>`, then click Refresh in the dropdown."
                )
            install_policy(_download_wandb_file(wandb_run_path, checkpoint_name))

        _add_checkpoint_dropdown(
            server,
            run_path=wandb_run_path,
            initial_onnx_name=wandb_onnx_name,
            initial_loaded_name=initial_wandb_name,
            on_select=_on_select,
        )

    # uplot pushes are expensive; cap reward-plot refresh to ~2 Hz.
    stop_refresh = threading.Event()

    def _plot_pump() -> None:
        import time as _time

        while not stop_refresh.is_set():
            try:
                refresh_reward_plot()
            except Exception:
                logger.exception("reward plot refresh failed")
            _time.sleep(0.5)

    pump_thread = threading.Thread(target=_plot_pump, name="reward-plot-pump", daemon=True)
    pump_thread.start()

    # Panel-worker thread: renders preview cams + encodes obs frames + pushes
    # HTML off the viewer thread so the 60 Hz render loop doesn't get CAPPED
    # when cameras are enabled. ``mujoco.Renderer`` is GL-context-bound to
    # its constructing thread, so preview renderers are built lazily inside
    # this loop on first use.
    preview_panel_dirty = [False]  # True while RGB panel HTML still shows a stale frame
    depth_preview_panel_dirty = [False]

    def _pump_preview_row(cams: list[_PreviewCamera], panels: dict[str, Any], dirty: list[bool]) -> None:
        if not panels:
            return
        if any(c.enabled for c in cams):
            # Hold sim_lock only across the cheap state copy (update_scene);
            # the GL render runs lock-free.
            staged: list[tuple[_PreviewCamera, mujoco.Renderer]] = []
            with sim_lock:
                for cam in cams:
                    if not cam.enabled:
                        continue
                    renderer = _ensure_preview_renderer(cam, model)
                    renderer.update_scene(
                        data, camera=cam.camera_name_in_model, scene_option=cam.scene_option
                    )
                    staged.append((cam, renderer))
            rendered: list[tuple[str, np.ndarray]] = []
            for cam, r in staged:
                frame = r.render()
                display = _depth_render_to_display_rgb(frame, cam.cutoff_m) if cam.mode == "depth" else frame
                rendered.append((cam.label, _upscale_for_panel(display)))
            panels["_html"].content = _row_html(rendered)
            dirty[0] = True
        elif dirty[0]:
            panels["_html"].content = _row_html([])
            dirty[0] = False

    def _panel_pump() -> None:
        import time as _time

        while not stop_refresh.is_set():
            try:
                _pump_preview_row(preview_cams, preview_panels, preview_panel_dirty)
                _pump_preview_row(depth_preview_cams, depth_preview_panels, depth_preview_panel_dirty)

                if cam_panels:
                    with panel_lock:
                        snap = latest_obs[0]
                        latest_obs[0] = None
                    if snap is not None:
                        _push_camera_frames(cam_panels, snap[0], snap[1])
            except Exception:
                logger.exception("panel pump iteration failed")
            _time.sleep(0.005)

    panel_thread = threading.Thread(target=_panel_pump, name="panel-pump", daemon=True)
    panel_thread.start()

    try:
        viewer.run()
    finally:
        stop_refresh.set()
        # Hard-exit skips ~700 ms of interpreter shutdown spent finalizing
        # torch/mujoco_warp/wandb. ``viewer.run()`` has already called viser's
        # ``stop()``, there are no open wandb runs / fork children / temp
        # files, and the daemon plot-pump thread dies with the process.
        import os as _os

        _os._exit(0)


def cli() -> None:
    from arm_farm.sim._env import load_env

    load_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(main)


if __name__ == "__main__":
    cli()
