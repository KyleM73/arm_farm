"""Shared SO-ARM101 lift-cube env factory.

``make_so101_lift_cube_env_cfg`` wraps mjlab's ``make_lift_cube_env_cfg`` and
fills in the SO-ARM101-specific fields: robot/cube entities, action scale,
ee-frame site, viewer body, collision sensor target.

``add_camera_obs`` layers a camera sensor on top and swaps the actor's
privileged state observations for a goal-position scalar — the
``Cube-Rgb`` and ``Cube-Depth`` variants both use it.
"""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg

from arm_farm.sim.assets.so101 import (
    EE_BODY,
    EE_SITE,
    SO101_ACTION_SCALE,
    get_cube_spec,
    get_so101_cfg,
)

# How many parallel envs the play-mode configs spin up by default. mjlab's
# viser viewer arranges them in a grid (with ``scene.env_spacing`` between
# centres), giving a quick visual read on policy variance from a single
# rollout.
PLAY_NUM_ENVS: int = 16


def make_so101_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_lift_cube_env_cfg()

    cfg.scene.entities = {
        "robot": get_so101_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
    }

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SO101_ACTION_SCALE

    # Bind ee-frame observations and reward to the SO-ARM101 grasp site.
    cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (EE_SITE,)
    cfg.observations["critic"].terms["ee_to_cube"].params["asset_cfg"].site_names = (EE_SITE,)
    cfg.rewards["lift"].params["asset_cfg"].site_names = (EE_SITE,)

    # SO-ARM101 collision geoms aren't individually named in the MJCF, so the
    # fingertip-friction DR events don't have anything to target. Drop them
    # rather than silently masking the assertion in mjlab.
    for k in ("fingertip_friction_slide", "fingertip_friction_spin", "fingertip_friction_roll"):
        cfg.events.pop(k, None)

    # Point the end-effector / ground contact sensor at the gripper subtree.
    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = EE_BODY

    cfg.viewer.body_name = EE_BODY
    cfg.viewer.distance = 1.0
    cfg.viewer.elevation = -20.0
    cfg.viewer.azimuth = 135.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        assert cfg.commands is not None
        cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)
        # 4x4 grid in viser by default. Override at the CLI for headless or
        # CPU-bound runs: ``play Cube --env.scene.num-envs N``.
        cfg.scene.num_envs = PLAY_NUM_ENVS

    return cfg


def add_camera_obs(
    cfg: ManagerBasedRlEnvCfg,
    camera_cfg: CameraSensorCfg,
    camera_obs_term: ObservationTermCfg,
) -> None:
    """Attach a camera sensor + observation group, swap actor obs to vision.

    Used by the visual cube tasks (Cube-Rgb, Cube-Depth). The actor's
    privileged state (``ee_to_cube``, ``cube_to_goal``) is replaced with a
    plain ``goal_position`` term so the policy has to learn from the camera
    instead of cheating on the hand-cube relative pose.
    """
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera_cfg,)

    cfg.observations["camera"] = ObservationGroupCfg(
        terms={camera_cfg.name: camera_obs_term},
        enable_corruption=False,
        concatenate_terms=True,
    )

    actor_obs = cfg.observations["actor"]
    actor_obs.terms.pop("ee_to_cube", None)
    actor_obs.terms.pop("cube_to_goal", None)
    actor_obs.terms["goal_position"] = ObservationTermCfg(
        func=manipulation_mdp.target_position,
        params={
            "command_name": "lift_height",
            "asset_cfg": SceneEntityCfg("robot", site_names=(EE_SITE,)),
        },
    )
