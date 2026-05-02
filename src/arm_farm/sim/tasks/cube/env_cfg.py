from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from arm_farm.sim.tasks.lift_cube_env_cfg import make_so101_lift_cube_env_cfg


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return make_so101_lift_cube_env_cfg(play=play)
