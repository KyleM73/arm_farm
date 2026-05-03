"""Native preview path: locks contracts that aren't obvious from the code."""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("mjviser")


def _compile(task: str) -> tuple:
    import mujoco
    from mjlab.scene import Scene
    from mjlab.tasks.registry import load_env_cfg

    import arm_farm.sim  # noqa: F401

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    scene = Scene(env_cfg.scene, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    return env_cfg, model, data


def _make_minimal_policy_ctx(env_cfg, model):
    """Build a ``_PolicyCtx`` with no ONNX session — enough for obs/reward tests."""
    import math

    import mujoco
    import numpy as np

    from arm_farm.sim.assets.so101 import ARM_JOINTS
    from arm_farm.sim.preview import _build_cube_sampler, _PolicyCtx, _resolve_name

    qpos_adrs, qvel_adrs, actuator_ids, soft_lo, soft_hi = [], [], [], [], []
    for j in ARM_JOINTS:
        jid = _resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, [f"robot/{j}", j])
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        qvel_adrs.append(int(model.jnt_dofadr[jid]))
        actuator_ids.append(_resolve_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"robot/{j}", j]))
        rng_lo, rng_hi = model.jnt_range[jid]
        if bool(model.jnt_limited[jid]):
            soft_lo.append(float(rng_lo))
            soft_hi.append(float(rng_hi))
        else:
            soft_lo.append(-math.inf)
            soft_hi.append(math.inf)

    return _PolicyCtx(
        session=None,
        obs_names=["joint_pos", "joint_vel", "ee_to_cube", "cube_to_goal", "actions"],
        state_input_name="obs",
        action_scale=np.zeros(6, dtype=np.float32),
        default_joint_pos=np.zeros(6, dtype=np.float64),
        joint_qpos_adrs=np.array(qpos_adrs, dtype=np.int32),
        joint_qvel_adrs=np.array(qvel_adrs, dtype=np.int32),
        joint_soft_lo=np.array(soft_lo, dtype=np.float64),
        joint_soft_hi=np.array(soft_hi, dtype=np.float64),
        actuator_ids=np.array(actuator_ids, dtype=np.int32),
        ee_site_id=_resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, ["robot/ee_site", "ee_site"]),
        decimation=int(env_cfg.decimation),
        sampler=_build_cube_sampler(env_cfg, model, seed=0),
    )


@pytest.mark.parametrize("action", ["zero", "random", "sine"])
def test_scripted_step_fn_advances_without_nan(action: str) -> None:
    """Each scripted ctrl mode must keep qpos finite over a few steps."""
    import numpy as np

    from arm_farm.sim.preview import _make_reset_fn, _make_scripted_step_fn

    _env_cfg, model, data = _compile("Cube")
    _make_reset_fn()(model, data)
    step_fn = _make_scripted_step_fn(action=action, sine_period_s=4.0)
    t0 = data.time
    for _ in range(5):
        step_fn(model, data)
    assert data.time > t0
    assert not np.isnan(data.qpos).any()


def test_compute_actor_obs_matches_cube_task_layout() -> None:
    """Locks the blind-task obs shape: ``(joint_pos+vel + ee_to_cube + cube_to_goal + actions)``."""
    import mujoco
    import numpy as np

    from arm_farm.sim.preview import _compute_actor_obs

    env_cfg, model, data = _compile("Cube")
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state"))
    mujoco.mj_forward(model, data)

    obs = _compute_actor_obs(model, data, _make_minimal_policy_ctx(env_cfg, model))
    assert obs.shape == (6 + 6 + 3 + 3 + 6,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_compute_actor_obs_rotates_goal_into_ee_frame() -> None:
    """Vision tasks swap ``ee_to_cube``/``cube_to_goal`` for an EE-frame
    ``goal_position``. Catches sign flips in the rotation."""
    import mujoco
    import numpy as np

    from arm_farm.sim.preview import _compute_actor_obs

    env_cfg, model, data = _compile("Cube-Rgb")
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state"))
    mujoco.mj_forward(model, data)

    ctx = _make_minimal_policy_ctx(env_cfg, model)
    ctx.obs_names = ["joint_pos", "joint_vel", "goal_position", "actions"]
    ctx.target_pos[:] = np.asarray(data.site_xpos[ctx.ee_site_id]) + np.array([0.1, 0.0, 0.0])

    obs = _compute_actor_obs(model, data, ctx)
    assert obs.shape == (6 + 6 + 3 + 6,)
    # 0.1 m offset in world frame should have unit length in the EE frame too.
    assert abs(float(np.linalg.norm(obs[12:15])) - 0.1) < 1e-5


def test_resample_cube_respects_env_cfg_ranges() -> None:
    """Sampler must pull ranges off the loaded env_cfg, not hard-coded constants —
    keeps the native preview honest if mjlab tweaks the LiftingCommand defaults."""
    import numpy as np

    from arm_farm.sim.preview import _build_cube_sampler, _resample_cube

    env_cfg, model, data = _compile("Cube")
    sampler = _build_cube_sampler(env_cfg, model, seed=0)
    _resample_cube(model, data, sampler)
    cube_xyz = data.qpos[sampler.cube_qpos_adr : sampler.cube_qpos_adr + 3]
    cube_quat = data.qpos[sampler.cube_qpos_adr + 3 : sampler.cube_qpos_adr + 7]

    assert (cube_xyz >= sampler.object_range_lo).all()
    assert (cube_xyz <= sampler.object_range_hi).all()
    assert abs(np.linalg.norm(cube_quat) - 1.0) < 1e-6


def test_reset_fn_moves_cube_off_keyframe_origin() -> None:
    """Without sampler-on-reset the cube spawns inside the SO-101 base. Must
    work in both scripted and policy modes; policy state must zero out."""
    import numpy as np

    from arm_farm.sim.preview import _build_cube_sampler, _make_reset_fn, _PolicyCtxHolder

    env_cfg, model, data = _compile("Cube")
    sampler = _build_cube_sampler(env_cfg, model, seed=0)

    _make_reset_fn(sampler=sampler)(model, data)
    assert not np.allclose(data.qpos[sampler.cube_qpos_adr : sampler.cube_qpos_adr + 3], 0.0)

    ctx = _make_minimal_policy_ctx(env_cfg, model)
    ctx.last_action[:] = 0.5
    ctx.decimation_counter = 3
    _make_reset_fn(sampler=sampler, ctx_holder=_PolicyCtxHolder(ctx=ctx))(model, data)
    assert ctx.decimation_counter == 0
    assert np.allclose(ctx.last_action, 0.0)


@pytest.mark.parametrize(
    ("task", "expected_data_type", "expected_camera"),
    [
        ("Cube-Rgb", "rgb", "robot/front"),
        ("Cube-Depth", "depth", "robot/wrist"),
    ],
)
def test_vision_cfg_extracts_policy_camera(task: str, expected_data_type: str, expected_camera: str) -> None:
    """Cube-Rgb → front only (wrist is preview-only); Cube-Depth → wrist."""
    from arm_farm.sim.preview import _vision_cfg_from_env_cfg

    env_cfg, model, _data = _compile(task)
    vcfg = _vision_cfg_from_env_cfg(env_cfg, model, ort_input_names=["obs", "camera"])
    assert vcfg is not None
    assert len(vcfg.cameras) == 1
    cam = vcfg.cameras[0]
    assert cam.data_type == expected_data_type
    assert cam.camera_name_in_model == expected_camera
    assert cam.height == 32 and cam.width == 32


def test_vision_cfg_handles_blind_and_mismatched_inputs() -> None:
    """Blind task → None; vision task with a 1-input (blind) ONNX → clear error."""
    from arm_farm.sim.preview import _vision_cfg_from_env_cfg

    env_cfg, model, _data = _compile("Cube")
    assert _vision_cfg_from_env_cfg(env_cfg, model, ort_input_names=["obs"]) is None

    env_cfg, model, _data = _compile("Cube-Rgb")
    with pytest.raises(RuntimeError, match="camera"):
        _vision_cfg_from_env_cfg(env_cfg, model, ort_input_names=["obs"])


@pytest.mark.parametrize(
    ("task", "expected_shape"),
    [
        ("Cube-Rgb", (3, 32, 32)),
        ("Cube-Depth", (1, 32, 32)),
    ],
)
def test_render_camera_obs_matches_mjlab_normalisation(task: str, expected_shape: tuple) -> None:
    """``(3,H,W) f32 in [0,1]`` for RGB; ``(1,H,W) f32`` clamped via cutoff for depth."""
    import numpy as np

    from arm_farm.sim.preview import _render_camera_obs, _vision_cfg_from_env_cfg

    env_cfg, model, data = _compile(task)
    vcfg = _vision_cfg_from_env_cfg(env_cfg, model, ort_input_names=["obs", "camera"])
    assert vcfg is not None
    obs = _render_camera_obs(vcfg.cameras[0], data)
    assert obs.shape == expected_shape
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_reward_ctx_covers_every_env_cfg_term() -> None:
    """Adding a reward term in env_cfg without wiring it into ``_build_reward_ctx``
    silently zeros it on the plot; this asserts every term has a callable."""
    import math

    from arm_farm.sim.preview import _build_reward_ctx

    env_cfg, model, data = _compile("Cube")
    ctx = _make_minimal_policy_ctx(env_cfg, model)
    rctx = _build_reward_ctx(env_cfg)
    assert set(rctx.names) == set(env_cfg.rewards.keys())
    for fn, name in zip(rctx.fns, rctx.names, strict=True):
        val = fn(model, data, ctx)
        assert math.isfinite(val), f"reward {name} returned {val}"
