"""SO-ARM101 mjlab tasks.

Importing each subpackage triggers ``register_mjlab_task`` for one task ID.
"""

from arm_farm.sim.tasks import cube as _cube  # noqa: F401
from arm_farm.sim.tasks import cube_depth as _cube_depth  # noqa: F401
from arm_farm.sim.tasks import cube_rgb as _cube_rgb  # noqa: F401
from arm_farm.sim.tasks import play as _play  # noqa: F401

__all__: tuple[str, ...] = ()
