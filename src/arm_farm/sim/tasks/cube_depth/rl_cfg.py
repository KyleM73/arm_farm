"""PPO config for ``Cube-Depth`` — same SpatialSoftmax CNN backbone as Cube-Rgb."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from arm_farm.sim.tasks.cube_rgb.rl_cfg import make_rl_cfg as _make_vision_rl_cfg


def make_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    return _make_vision_rl_cfg(experiment_name="so101_cube_depth")
