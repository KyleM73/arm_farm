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
    # 32x32 matches mjlab YAM-Rgb; bumping to 64 ~quarters max envs on 32 GiB.
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
    add_camera_obs(
        cfg,
        camera_cfg=_rgb_cam("front", "robot/front"),
        camera_obs_term=_rgb_obs("front"),
    )
    # Wrist camera (``so101_constants.WRIST_CAMERA``) is available for
    # rendering/overlays. To feed it to the policy, add it to
    # ``cfg.scene.sensors`` + ``cfg.observations["camera"].terms`` like
    # ``front`` (yields a ``(3, 32, 64)`` front+wrist concat).
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
