"""RGB visual lift-cube task — registered as ``Cube-Rgb``.

Adds the front-mount RGB camera + cube color DR. Privileged state info
(``ee_to_cube``, ``cube_to_goal``) is removed from the actor's obs and
replaced by a goal-position term, mirroring the YAM-Rgb config.
"""

from mjlab.tasks.registry import register_mjlab_task

from arm_farm.sim.runner import ArmFarmManipulationRunner
from arm_farm.sim.tasks.cube_rgb.env_cfg import make_env_cfg
from arm_farm.sim.tasks.cube_rgb.rl_cfg import make_rl_cfg

register_mjlab_task(
    task_id="Cube-Rgb",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=ArmFarmManipulationRunner,
)
