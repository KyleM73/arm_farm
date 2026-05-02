"""lerobot ``Robot`` wrapper around an mjlab SO-ARM101 sim env.

Importing this module registers ``MujocoSO101RobotConfig`` as a draccus subclass
of ``RobotConfig``, so ``--robot.type=mujoco_so101`` resolves on the lerobot CLI:

    uv run lerobot-replay --robot.type=mujoco_so101 --dataset.repo_id=...

Action and observation joint values are in normalised LeRobot units
(arm joints in [-100, 100], gripper in [0, 100]) so recorded SO-ARM101
datasets line up with the sim wrapper without remapping.
"""

from arm_farm.sim.lerobot_robot.config_mujoco_so101 import MujocoSO101RobotConfig
from arm_farm.sim.lerobot_robot.mujoco_so101 import MujocoSO101

__all__ = ("MujocoSO101", "MujocoSO101RobotConfig")
