"""SO-ARM101 lerobot config presets, populated from env vars; mirrors what
the shell scripts pass on the CLI."""

from __future__ import annotations

import os

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable `{name}` is not set. "
            "Copy `.env.example` to `.env`, fill in your ports, and `source .env`."
        )
    return value


def follower_cameras() -> dict[str, CameraConfig]:
    return {
        "wrist": OpenCVCameraConfig(
            index_or_path=int(_env("ARM_FARM_WRIST_CAM")),
            fps=30,
            width=640,
            height=480,
        ),
        "front": OpenCVCameraConfig(
            index_or_path=int(_env("ARM_FARM_FRONT_CAM")),
            fps=30,
            width=640,
            height=480,
        ),
    }


def follower_config(*, with_cameras: bool = True) -> SOFollowerRobotConfig:
    return SOFollowerRobotConfig(
        port=_env("ARM_FARM_FOLLOWER_PORT"),
        id=os.environ.get("ARM_FARM_FOLLOWER_ID", "follower_01"),
        cameras=follower_cameras() if with_cameras else {},
    )


def leader_config() -> SOLeaderTeleopConfig:
    return SOLeaderTeleopConfig(
        port=_env("ARM_FARM_LEADER_PORT"),
        id=os.environ.get("ARM_FARM_LEADER_ID", "leader_01"),
    )
