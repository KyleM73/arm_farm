from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from arm_farm.sim.tasks.lift_cube_env_cfg import add_camera_obs, make_so101_lift_cube_env_cfg


def _rgb_cam(name: str, mjcf_name: str) -> CameraSensorCfg:
    # 32x32 matches mjlab YAM-Rgb (yam_lift_cube_vision_env_cfg). Bumping to 64
    # roughly quarters the max parallel-env count on a 32 GiB GPU.
    return CameraSensorCfg(
        name=name,
        camera_name=mjcf_name,
        height=32,
        width=32,
        data_types=("rgb",),
        enabled_geom_groups=(0, 2, 3),
        use_shadows=False,
        use_textures=True,
    )


def _rgb_obs(sensor_name: str) -> ObservationTermCfg:
    return ObservationTermCfg(
        func=manipulation_mdp.camera_rgb,
        params={"sensor_name": sensor_name},
    )


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_so101_lift_cube_env_cfg(play=play)
    # First camera (front) goes through the helper, which sets up the camera
    # obs group and swaps the actor's privileged state for ``goal_position``.
    add_camera_obs(
        cfg,
        camera_cfg=_rgb_cam("front", "robot/front"),
        camera_obs_term=_rgb_obs("front"),
    )
    # Optional second camera (wrist). The wrist MJCF camera (``robot/wrist``)
    # is attached to the robot entity via ``so101_constants.WRIST_CAMERA``, so
    # uncommenting these two lines is the only change needed to feed the
    # policy a (3, 32, 64) front+wrist concatenation along the width axis.
    # cfg.scene.sensors = (cfg.scene.sensors or ()) + (_rgb_cam("wrist", "robot/wrist"),)
    # cfg.observations["camera"].terms["wrist"] = _rgb_obs("wrist")
    cfg.events["cube_color"] = EventTermCfg(
        func=dr.geom_rgba,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
            "operation": "abs",
            "distribution": "uniform",
            "axes": [0, 1, 2],
            "ranges": (0.0, 1.0),
        },
    )
    return cfg
