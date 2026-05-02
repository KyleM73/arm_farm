from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from arm_farm.sim.tasks.lift_cube_env_cfg import add_camera_obs, make_so101_lift_cube_env_cfg

DEPTH_CUTOFF_M = 0.5


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_so101_lift_cube_env_cfg(play=play)
    add_camera_obs(
        cfg,
        camera_cfg=CameraSensorCfg(
            name="wrist",
            camera_name="robot/wrist",
            height=32,
            width=32,
            data_types=("depth",),
            enabled_geom_groups=(0, 2, 3),
            use_shadows=False,
            use_textures=True,
        ),
        camera_obs_term=ObservationTermCfg(
            func=manipulation_mdp.camera_depth,
            params={"sensor_name": "wrist", "cutoff_distance": DEPTH_CUTOFF_M},
        ),
    )
    return cfg
