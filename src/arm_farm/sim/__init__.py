"""arm_farm sim package.

Importing this module registers the SO-ARM101 mjlab tasks (`Cube`, `Cube-Rgb`,
`Cube-Depth`, `Play`) via ``mjlab.tasks.registry.register_mjlab_task``. mjlab
discovers this package through the ``mjlab.tasks`` entry point declared in
``pyproject.toml`` and imports it during ``mjlab`` startup.
"""

from arm_farm.sim import lerobot_robot as _lerobot_robot  # noqa: F401  (registers Robot)
from arm_farm.sim import tasks as _tasks  # noqa: F401  (registers tasks)

__all__: tuple[str, ...] = ()
