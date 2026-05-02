from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from arm_farm.sim.tasks.lift_cube_env_cfg import add_camera_obs, make_so101_lift_cube_env_cfg


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_so101_lift_cube_env_cfg(play=play)
    add_camera_obs(
        cfg,
        camera_cfg=CameraSensorCfg(
            name="front",
            camera_name="robot/front",
            height=64,
            width=64,
            data_types=("rgb",),
            enabled_geom_groups=(0, 2, 3),
            use_shadows=False,
            use_textures=True,
        ),
        camera_obs_term=ObservationTermCfg(
            func=manipulation_mdp.camera_rgb,
            params={"sensor_name": "front"},
        ),
    )
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
