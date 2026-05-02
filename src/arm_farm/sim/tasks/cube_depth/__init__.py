"""Depth visual lift-cube task — registered as ``Cube-Depth``.

Wrist-mounted depth camera (32x32, normalised to [0, 1] with a 0.5 m cutoff)
plus goal-position scalar. No cube color DR (depth is colour-invariant).
"""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from arm_farm.sim.tasks.cube_depth.env_cfg import make_env_cfg
from arm_farm.sim.tasks.cube_depth.rl_cfg import make_rl_cfg

register_mjlab_task(
    task_id="Cube-Depth",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)
