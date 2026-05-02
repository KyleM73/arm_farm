"""Smoke-test the native MuJoCo preview path: scene composes, model steps."""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("mjviser")


def test_scene_composes_to_native_mjmodel() -> None:
    import mujoco
    from mjlab.scene import Scene
    from mjlab.tasks.registry import load_env_cfg

    import arm_farm.sim  # noqa: F401  (registers tasks)

    env_cfg = load_env_cfg("Cube", play=True)
    env_cfg.scene.num_envs = 1

    scene = Scene(env_cfg.scene, device="cpu")
    model = scene.compile()
    assert isinstance(model, mujoco.MjModel)
    # Robot has 6 actuated joints + cube has 1 freejoint = 7 joint entries.
    assert model.njnt >= 7
    # Mjlab.Scene merges per-entity init_state keyframes into a scene keyframe.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state") >= 0


@pytest.mark.parametrize("action", ["zero", "random", "sine"])
def test_step_fn_advances_native_sim(action: str) -> None:
    import mujoco
    import numpy as np
    from mjlab.scene import Scene
    from mjlab.tasks.registry import load_env_cfg

    import arm_farm.sim  # noqa: F401
    from arm_farm.sim.play_native import _make_scripted_step_fn, _reset_to_init_state

    env_cfg = load_env_cfg("Cube", play=True)
    env_cfg.scene.num_envs = 1
    scene = Scene(env_cfg.scene, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    _reset_to_init_state(model, data)

    step_fn = _make_scripted_step_fn(action=action, sine_period_s=4.0)  # ty: ignore[invalid-argument-type]
    t0 = data.time
    for _ in range(5):
        step_fn(model, data)
    assert data.time > t0
    assert not np.isnan(data.qpos).any()


def _build_compiled_cube_model() -> tuple:
    """Helper for the policy-path tests: compile the Cube task scene."""
    import mujoco
    from mjlab.scene import Scene
    from mjlab.tasks.registry import load_env_cfg

    import arm_farm.sim  # noqa: F401

    env_cfg = load_env_cfg("Cube", play=True)
    env_cfg.scene.num_envs = 1
    scene = Scene(env_cfg.scene, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    return model, data


def test_resolve_indices_against_compiled_cube_model() -> None:
    """Confirm the helpers find the joints, ee_site, and cube freejoint."""
    import mujoco
    import numpy as np

    from arm_farm.sim.assets.so101 import ARM_JOINTS
    from arm_farm.sim.play_native import (
        _find_cube_body,
        _find_cube_freejoint,
        _resolve_name,
    )

    model, _data = _build_compiled_cube_model()
    for j in ARM_JOINTS:
        assert _resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, [f"robot/{j}", j]) >= 0
        assert _resolve_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"robot/{j}", j]) >= 0
    assert _resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, ["robot/ee_site", "ee_site"]) >= 0

    cube_jid = _find_cube_freejoint(model)
    assert model.jnt_type[cube_jid] == mujoco.mjtJoint.mjJNT_FREE
    cube_bid = _find_cube_body(model)
    assert int(np.asarray(model.jnt_bodyid)[cube_jid]) == cube_bid


def test_compute_actor_obs_matches_cube_task_layout() -> None:
    """``_compute_actor_obs`` produces a (24,) f32 vector for the Cube task."""
    import mujoco
    import numpy as np

    from arm_farm.sim.assets.so101 import ARM_JOINTS
    from arm_farm.sim.play_native import (
        _compute_actor_obs,
        _find_cube_body,
        _find_cube_freejoint,
        _PolicyCtx,
        _resolve_name,
    )

    model, data = _build_compiled_cube_model()
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state"))
    mujoco.mj_forward(model, data)

    qpos_adrs = []
    qvel_adrs = []
    actuator_ids = []
    for j in ARM_JOINTS:
        jid = _resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, [f"robot/{j}", j])
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        qvel_adrs.append(int(model.jnt_dofadr[jid]))
        actuator_ids.append(_resolve_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"robot/{j}", j]))
    cube_jid = _find_cube_freejoint(model)

    ctx = _PolicyCtx(
        session=None,  # ty: ignore[invalid-argument-type]
        obs_names=["joint_pos", "joint_vel", "ee_to_cube", "cube_to_goal", "actions"],
        state_input_name="obs",
        action_scale=np.zeros(6, dtype=np.float32),
        default_joint_pos=np.zeros(6, dtype=np.float64),
        joint_qpos_adrs=np.array(qpos_adrs, dtype=np.int32),
        joint_qvel_adrs=np.array(qvel_adrs, dtype=np.int32),
        actuator_ids=np.array(actuator_ids, dtype=np.int32),
        ee_site_id=_resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, ["robot/ee_site", "ee_site"]),
        cube_body_id=_find_cube_body(model),
        cube_qpos_adr=int(model.jnt_qposadr[cube_jid]),
        cube_qvel_adr=int(model.jnt_dofadr[cube_jid]),
    )

    obs = _compute_actor_obs(model, data, ctx)
    assert obs.shape == (6 + 6 + 3 + 3 + 6,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_resample_command_writes_cube_qpos() -> None:
    """``_resample_command`` repositions the cube + sets a new target."""
    import numpy as np

    from arm_farm.sim.play_native import (
        OBJECT_RANGE_HI,
        OBJECT_RANGE_LO,
        TARGET_RANGE_HI,
        TARGET_RANGE_LO,
        _find_cube_body,
        _find_cube_freejoint,
        _PolicyCtx,
        _resample_command,
    )

    model, data = _build_compiled_cube_model()
    cube_jid = _find_cube_freejoint(model)
    cube_qpos_adr = int(model.jnt_qposadr[cube_jid])
    cube_qvel_adr = int(model.jnt_dofadr[cube_jid])

    ctx = _PolicyCtx(
        session=None,  # ty: ignore[invalid-argument-type]
        obs_names=[],
        state_input_name="obs",
        action_scale=np.zeros(6, dtype=np.float32),
        default_joint_pos=np.zeros(6, dtype=np.float64),
        joint_qpos_adrs=np.zeros(6, dtype=np.int32),
        joint_qvel_adrs=np.zeros(6, dtype=np.int32),
        actuator_ids=np.zeros(6, dtype=np.int32),
        ee_site_id=0,
        cube_body_id=_find_cube_body(model),
        cube_qpos_adr=cube_qpos_adr,
        cube_qvel_adr=cube_qvel_adr,
    )

    _resample_command(model, data, ctx)
    cube_xyz = data.qpos[cube_qpos_adr : cube_qpos_adr + 3]
    cube_quat = data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7]

    assert (cube_xyz >= OBJECT_RANGE_LO).all() and (cube_xyz <= OBJECT_RANGE_HI).all()
    assert abs(np.linalg.norm(cube_quat) - 1.0) < 1e-6  # Unit quaternion.
    assert (ctx.target_pos >= TARGET_RANGE_LO).all()
    assert (ctx.target_pos <= TARGET_RANGE_HI).all()


def _build_compiled_env_cfg(task: str) -> tuple:
    """Helper: load play env_cfg, build native model + data, return env_cfg too."""
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


def test_vision_cfg_from_cube_rgb_env_cfg() -> None:
    """``_vision_cfg_from_env_cfg`` extracts RGB camera config from Cube-Rgb."""
    from arm_farm.sim.play_native import _vision_cfg_from_env_cfg

    env_cfg, _, _ = _build_compiled_env_cfg("Cube-Rgb")
    vcfg = _vision_cfg_from_env_cfg(env_cfg, ort_input_names=["obs", "camera"])
    assert vcfg is not None
    assert vcfg.data_type == "rgb"
    assert vcfg.height == 64 and vcfg.width == 64
    assert vcfg.camera_name_in_model == "robot/front"
    assert vcfg.cutoff_distance is None


def test_vision_cfg_from_cube_depth_env_cfg() -> None:
    """``_vision_cfg_from_env_cfg`` extracts depth camera config from Cube-Depth."""
    from arm_farm.sim.play_native import _vision_cfg_from_env_cfg

    env_cfg, _, _ = _build_compiled_env_cfg("Cube-Depth")
    vcfg = _vision_cfg_from_env_cfg(env_cfg, ort_input_names=["obs", "camera"])
    assert vcfg is not None
    assert vcfg.data_type == "depth"
    assert vcfg.height == 32 and vcfg.width == 32
    assert vcfg.camera_name_in_model == "robot/wrist"
    assert vcfg.cutoff_distance is not None and vcfg.cutoff_distance > 0


def test_vision_cfg_returns_none_for_blind_task() -> None:
    """Blind ``Cube`` task has no camera obs group; helper returns None."""
    from arm_farm.sim.play_native import _vision_cfg_from_env_cfg

    env_cfg, _, _ = _build_compiled_env_cfg("Cube")
    assert _vision_cfg_from_env_cfg(env_cfg, ort_input_names=["obs"]) is None


def test_vision_cfg_rejects_blind_onnx_for_vision_task() -> None:
    """Passing a 1-input ONNX with a vision task surfaces a clear error."""
    import pytest

    from arm_farm.sim.play_native import _vision_cfg_from_env_cfg

    env_cfg, _, _ = _build_compiled_env_cfg("Cube-Rgb")
    with pytest.raises(RuntimeError, match="camera"):
        _vision_cfg_from_env_cfg(env_cfg, ort_input_names=["obs"])


def test_render_camera_obs_rgb_shape_and_range() -> None:
    """RGB render returns (3, H, W) f32 in [0, 1]."""
    import numpy as np

    from arm_farm.sim.play_native import (
        _build_render_scene_option,
        _make_renderer,
        _render_camera_obs,
        _vision_cfg_from_env_cfg,
    )

    env_cfg, model, data = _build_compiled_env_cfg("Cube-Rgb")
    vcfg = _vision_cfg_from_env_cfg(env_cfg, ort_input_names=["obs", "camera"])
    assert vcfg is not None

    renderer = _make_renderer(model, vcfg)
    scene_option = _build_render_scene_option(vcfg.enabled_geom_groups)

    obs = _render_camera_obs(renderer, data, vcfg, scene_option)
    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_render_camera_obs_depth_shape_and_clamp() -> None:
    """Depth render returns (1, H, W) f32 clamped to [0, 1] by cutoff."""
    import numpy as np

    from arm_farm.sim.play_native import (
        _build_render_scene_option,
        _make_renderer,
        _render_camera_obs,
        _vision_cfg_from_env_cfg,
    )

    env_cfg, model, data = _build_compiled_env_cfg("Cube-Depth")
    vcfg = _vision_cfg_from_env_cfg(env_cfg, ort_input_names=["obs", "camera"])
    assert vcfg is not None

    renderer = _make_renderer(model, vcfg)
    scene_option = _build_render_scene_option(vcfg.enabled_geom_groups)

    obs = _render_camera_obs(renderer, data, vcfg, scene_option)
    assert obs.shape == (1, 32, 32)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_compute_actor_obs_includes_goal_position_for_vision() -> None:
    """Vision tasks swap ee_to_cube/cube_to_goal for a goal_position term."""
    import mujoco
    import numpy as np

    from arm_farm.sim.assets.so101 import ARM_JOINTS
    from arm_farm.sim.play_native import (
        _compute_actor_obs,
        _find_cube_body,
        _find_cube_freejoint,
        _PolicyCtx,
        _resolve_name,
    )

    _env_cfg, model, data = _build_compiled_env_cfg("Cube-Rgb")
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state"))
    mujoco.mj_forward(model, data)

    qpos_adrs, qvel_adrs, actuator_ids = [], [], []
    for j in ARM_JOINTS:
        jid = _resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, [f"robot/{j}", j])
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        qvel_adrs.append(int(model.jnt_dofadr[jid]))
        actuator_ids.append(_resolve_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"robot/{j}", j]))
    cube_jid = _find_cube_freejoint(model)

    ctx = _PolicyCtx(
        session=None,  # ty: ignore[invalid-argument-type]
        obs_names=["joint_pos", "joint_vel", "goal_position", "actions"],
        state_input_name="obs",
        action_scale=np.zeros(6, dtype=np.float32),
        default_joint_pos=np.zeros(6, dtype=np.float64),
        joint_qpos_adrs=np.array(qpos_adrs, dtype=np.int32),
        joint_qvel_adrs=np.array(qvel_adrs, dtype=np.int32),
        actuator_ids=np.array(actuator_ids, dtype=np.int32),
        ee_site_id=_resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, ["robot/ee_site", "ee_site"]),
        cube_body_id=_find_cube_body(model),
        cube_qpos_adr=int(model.jnt_qposadr[cube_jid]),
        cube_qvel_adr=int(model.jnt_dofadr[cube_jid]),
    )
    # Pretend the goal sits 0.1 m forward of the EE in the world frame.
    ee_pos = np.asarray(data.site_xpos[ctx.ee_site_id], dtype=np.float64)
    ctx.target_pos[:] = ee_pos + np.array([0.1, 0.0, 0.0])

    obs = _compute_actor_obs(model, data, ctx)
    assert obs.shape == (6 + 6 + 3 + 6,)  # joint_pos + joint_vel + goal_position + actions
    assert obs.dtype == np.float32
    # The goal_position chunk (after joint_pos[6] + joint_vel[6]) should be the
    # target rotated into the EE frame; magnitude is preserved by rotation.
    goal_in_ee = obs[12:15]
    assert abs(float(np.linalg.norm(goal_in_ee)) - 0.1) < 1e-5
