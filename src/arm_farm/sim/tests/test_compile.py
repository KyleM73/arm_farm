"""Each registered task compiles end-to-end. Catches MJCF / mjlab regressions."""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")


def test_so101_entity_compiles_with_arm_joints_and_ee_site() -> None:
    """SO-ARM101 MJCF compiles; programmatic ``ee_site`` survives spec injection."""
    from mjlab.entity.entity import Entity

    from arm_farm.sim.assets.so101 import ARM_JOINTS, get_so101_cfg

    model = Entity(get_so101_cfg()).spec.compile()
    joint_names = {model.joint(i).name for i in range(model.njnt)}
    assert set(ARM_JOINTS) <= joint_names, f"missing arm joints: {joint_names}"
    site_names = {model.site(i).name for i in range(model.nsite)}
    assert "ee_site" in site_names


@pytest.mark.parametrize("task_id", ["Cube", "Cube-Rgb", "Cube-Depth", "Play"])
def test_task_envcfg_loads(task_id: str) -> None:
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import arm_farm.sim  # noqa: F401  (registers tasks)

    env_cfg = load_env_cfg(task_id)
    assert "robot" in env_cfg.scene.entities
    assert load_rl_cfg(task_id) is not None
