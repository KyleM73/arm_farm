"""Idle preview task — registered as ``Play``. Robot + cube + plane; no
rewards, infinite episode. Smoke-test target on Mac."""

from mjlab.tasks.registry import register_mjlab_task

from arm_farm.sim.tasks.play.env_cfg import make_env_cfg
from arm_farm.sim.tasks.play.rl_cfg import make_rl_cfg

register_mjlab_task(
    task_id="Play",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
)
