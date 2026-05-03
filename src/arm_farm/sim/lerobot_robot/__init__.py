"""Registers ``MujocoSO101RobotConfig`` as ``--robot.type=mujoco_so101``.

Joint values use lerobot units (arm in ``[-100, 100]``, gripper in
``[0, 100]``) so recorded SO-ARM101 datasets feed the sim wrapper
without remapping."""

from arm_farm.sim.lerobot_robot.config_mujoco_so101 import MujocoSO101RobotConfig
from arm_farm.sim.lerobot_robot.mujoco_so101 import MujocoSO101

__all__ = ("MujocoSO101", "MujocoSO101RobotConfig")
