"""Idle preview task — registered as ``Play``.

Robot + cube + plane, no rewards, no terminations beyond timeout, infinite
episode length. Useful as a smoke check on Mac (`play.sh Play --agent zero`)
before any RL plumbing is wired up.
"""

from mjlab.tasks.registry import register_mjlab_task

from arm_farm.sim.tasks.play.env_cfg import make_env_cfg
from arm_farm.sim.tasks.play.rl_cfg import make_rl_cfg

register_mjlab_task(
    task_id="Play",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
)
