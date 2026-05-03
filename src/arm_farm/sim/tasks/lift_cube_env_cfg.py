"""SO-ARM101 lift-cube env factory and vision-variant builder."""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

from arm_farm.sim.assets.so101 import (
    EE_BODY,
    EE_SITE,
    SO101_ACTION_SCALE,
    get_cube_spec,
    get_so101_cfg,
)

# Play-mode default: viser arranges the envs in a grid for variance read-out.
PLAY_NUM_ENVS: int = 4

# Train-mode default; override per-run with ``--env.scene.num-envs N``.
# Vision tasks at 32x32 RGB fit 4096 envs on a 32 GiB GPU.
TRAIN_NUM_ENVS: int = 4096


def make_so101_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_lift_cube_env_cfg()

    # 200 Hz physics; decimation=7 → ~28.6 Hz policy/camera obs, approximately
    # matching the 30 Hz lerobot dataset rate.
    cfg.decimation = 7

    cfg.scene.entities = {
        "robot": get_so101_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
    }

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SO101_ACTION_SCALE

    # Re-anchor ee-frame obs/rewards onto the SO-ARM101 grasp site.
    cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (EE_SITE,)
    cfg.observations["critic"].terms["ee_to_cube"].params["asset_cfg"].site_names = (EE_SITE,)
    cfg.rewards["lift"].params["asset_cfg"].site_names = (EE_SITE,)

    # SO-ARM101 fingertip geoms are unnamed in the MJCF; drop the DR events
    # that target them rather than silently masking mjlab's assertion.
    for k in ("fingertip_friction_slide", "fingertip_friction_spin", "fingertip_friction_roll"):
        cfg.events.pop(k, None)

    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = EE_BODY

    lift_cmd = cfg.commands["lift_height"]
    assert isinstance(lift_cmd, LiftingCommandCfg)
    lift_cmd.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
        x=(0.18, 0.28),
        y=(-0.15, 0.15),
        z=(0.02, 0.05),
    )
    lift_cmd.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
        x=(0.18, 0.28),
        y=(-0.15, 0.15),
        z=(0.15, 0.25),
    )

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
        cfg.scene.num_envs = PLAY_NUM_ENVS
    else:
        cfg.scene.num_envs = TRAIN_NUM_ENVS

    return cfg


def add_camera_obs(
    cfg: ManagerBasedRlEnvCfg,
    camera_cfg: CameraSensorCfg,
    camera_obs_term: ObservationTermCfg,
) -> None:
    """Add the camera obs and replace the actor's privileged ``ee_to_cube`` /
    ``cube_to_goal`` with a plain ``goal_position`` so the policy must learn
    from vision."""
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
