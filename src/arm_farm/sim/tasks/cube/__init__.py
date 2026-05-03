"""State-only (blind) SO-ARM101 lift-cube task — registered as ``Cube``."""

from mjlab.tasks.registry import register_mjlab_task

from arm_farm.sim.runner import ArmFarmManipulationRunner
from arm_farm.sim.tasks.cube.env_cfg import make_env_cfg
from arm_farm.sim.tasks.cube.rl_cfg import make_rl_cfg

register_mjlab_task(
    task_id="Cube",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=ArmFarmManipulationRunner,
)
