"""Tasks + Robot subclass register on package import (the entry-point contract)."""

from __future__ import annotations

import pytest

mjlab = pytest.importorskip("mjlab")


def test_arm_farm_tasks_are_registered() -> None:
    from mjlab.tasks.registry import list_tasks

    import arm_farm.sim  # noqa: F401

    missing = {"Cube", "Cube-Rgb", "Cube-Depth", "Play"} - set(list_tasks())
    assert not missing, f"missing tasks: {missing}"


def test_robot_subclass_is_registered() -> None:
    from lerobot.robots.config import RobotConfig

    import arm_farm.sim  # noqa: F401

    assert "mujoco_so101" in RobotConfig.get_known_choices()
