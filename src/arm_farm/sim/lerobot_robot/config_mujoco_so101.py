"""Config for ``MujocoSO101``; registered as ``--robot.type=mujoco_so101``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from lerobot.robots.config import RobotConfig

# Annotated as plain str on the dataclass because draccus can't decode
# ``Literal``; this alias is for type hints + the __post_init__ check.
ViewerBackend = Literal["native", "viser", "none"]
_VIEWER_CHOICES: tuple[str, ...] = get_args(ViewerBackend)


@RobotConfig.register_subclass("mujoco_so101")
@dataclass(kw_only=True)
class MujocoSO101RobotConfig(RobotConfig):
    """SO-ARM101 mjlab sim Robot config.

    Attributes:
        seed: env reset seed.
        viewer: ``"viser"`` (web) | ``"native"`` (OpenGL; needs ``mjpython``
            on macOS) | ``"none"`` (headless).
        decimation: physics steps per ``send_action``. 7 → ~28.6 Hz at the
            default 0.005 s timestep, approximately matching the 30 Hz
            lerobot dataset frame rate.
    """

    seed: int = 0
    viewer: str = "viser"
    decimation: int = 7

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.viewer not in _VIEWER_CHOICES:
            raise ValueError(f"viewer must be one of {_VIEWER_CHOICES}, got {self.viewer!r}")
