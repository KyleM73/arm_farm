"""Config dataclass for ``MujocoSO101``.

Registered as a ``RobotConfig`` choice subclass, so ``draccus`` (and lerobot's
CLI) recognise ``--robot.type=mujoco_so101``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from lerobot.robots.config import RobotConfig

# `viewer` is annotated as a plain ``str`` on the dataclass below because
# draccus has no decoder for ``Literal`` types; the alias lives here for
# type hints in helper signatures and gives ``__post_init__`` the
# canonical choice list.
ViewerBackend = Literal["native", "viser", "none"]
_VIEWER_CHOICES: tuple[str, ...] = get_args(ViewerBackend)


@RobotConfig.register_subclass("mujoco_so101")
@dataclass(kw_only=True)
class MujocoSO101RobotConfig(RobotConfig):
    """Configuration for the SO-ARM101 mjlab sim Robot.

    Attributes:
        seed: deterministic env reset seed.
        viewer: which viewer backend to attach during the rollout. One of:
            - ``"viser"`` (default): web viewer via
              ``mjviser.ViserMujocoScene``. The URL is logged at
              connect; no local display required and works under
              regular ``uv run`` without ``mjpython``.
            - ``"native"``: MuJoCo's OpenGL window via
              ``mujoco.viewer.launch_passive``. Needs a display, plus
              ``mjpython`` on macOS — ``scripts/sim/replay.sh`` handles
              the wrapping when ``ARM_FARM_SIM_VIEWER=native``.
            - ``"none"``: skip the viewer (CI, headless replay-to-disk).
        decimation: physics steps per ``send_action`` call. Default 4 with the
            Play env's 0.005 s timestep gives a ~50 Hz control rate, matching
            mjlab's manipulation defaults; bump to 7 for ~30 Hz to align with
            lerobot dataset framerates.
    """

    seed: int = 0
    viewer: str = "viser"
    decimation: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.viewer not in _VIEWER_CHOICES:
            raise ValueError(f"viewer must be one of {_VIEWER_CHOICES}, got {self.viewer!r}")
