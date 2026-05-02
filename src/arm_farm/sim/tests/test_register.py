"""Verify the four arm_farm sim tasks land in mjlab's task registry on import."""

from __future__ import annotations

import pytest

# Skip the whole file if mjlab isn't installed (the `sim` extra wasn't synced).
mjlab = pytest.importorskip("mjlab")


def test_arm_farm_tasks_are_registered() -> None:
    from mjlab.tasks.registry import list_tasks

    import arm_farm.sim  # noqa: F401  (import triggers register_mjlab_task calls)

    expected = {"Cube", "Cube-Rgb", "Cube-Depth", "Play"}
    registered = set(list_tasks())
    missing = expected - registered
    assert not missing, f"Missing arm_farm sim tasks in mjlab registry: {missing}"


def test_robot_subclass_is_registered() -> None:
    from lerobot.robots.config import RobotConfig

    import arm_farm.sim  # noqa: F401

    # draccus.ChoiceRegistry exposes registered choice names via _name_to_subclass.
    names = set(RobotConfig.get_known_choices().keys())
    assert "mujoco_so101" in names, (
        f"mujoco_so101 not registered as a RobotConfig choice. Found: {sorted(names)}"
    )
