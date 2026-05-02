"""SO-ARM101 entity for mjlab sims.

The MJCF (`xmls/so101.xml`) is vendored from
https://github.com/TheRobotStudio/SO-ARM100 (Apache-2.0). The local scene wrapper
adds an end-effector site and front camera; the wrist camera is attached via
`EntityCfg.cameras`.
"""

from arm_farm.sim.assets.so101.so101_constants import (
    ARM_JOINTS,
    EE_BODY,
    EE_SITE,
    GRIPPER_JOINT,
    SO101_ACTION_SCALE,
    SO101_DAMPING,
    SO101_EFFORT_LIMIT,
    SO101_STIFFNESS,
    get_cube_spec,
    get_so101_cfg,
)

__all__ = (
    "ARM_JOINTS",
    "EE_BODY",
    "EE_SITE",
    "GRIPPER_JOINT",
    "SO101_ACTION_SCALE",
    "SO101_DAMPING",
    "SO101_EFFORT_LIMIT",
    "SO101_STIFFNESS",
    "get_cube_spec",
    "get_so101_cfg",
)
