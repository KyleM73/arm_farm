"""Registers Cube/Cube-Rgb/Cube-Depth/Play tasks and the mujoco_so101 Robot.

Loaded by mjlab via the ``mjlab.tasks`` entry point in ``pyproject.toml``."""

from arm_farm.sim import lerobot_robot as _lerobot_robot  # noqa: F401  (registers Robot)
from arm_farm.sim import tasks as _tasks  # noqa: F401  (registers tasks)

__all__: tuple[str, ...] = ()
