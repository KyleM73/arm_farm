"""Confirm each task's env_cfg compiles to a valid MuJoCo model with 6 actuated joints.

Cheap end-to-end check that the SO-ARM101 MJCF + mjlab actuator injection +
scene includes are wired up correctly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")


def test_so101_entity_compiles() -> None:
    from mjlab.entity.entity import Entity

    from arm_farm.sim.assets.so101 import ARM_JOINTS, get_so101_cfg

    robot = Entity(get_so101_cfg())
    model = robot.spec.compile()
    # Robot entity alone — the cube is registered separately, so the only joints
    # here are the 6 actuated arm joints.
    joint_names = {model.joint(i).name for i in range(model.njnt)}
    for j in ARM_JOINTS:
        assert j in joint_names, f"Joint {j!r} missing from compiled model: {joint_names}"
    # ee_site is added programmatically in get_spec(); confirm it's present.
    site_names = {model.site(i).name for i in range(model.nsite)}
    assert "ee_site" in site_names, f"ee_site missing: {site_names}"


@pytest.mark.parametrize("task_id", ["Cube", "Cube-Rgb", "Cube-Depth", "Play"])
def test_task_envcfg_loads(task_id: str) -> None:
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import arm_farm.sim  # noqa: F401  (registers tasks)

    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)
    assert env_cfg.scene.entities, f"{task_id}: empty scene.entities"
    assert "robot" in env_cfg.scene.entities, f"{task_id}: missing robot entity"
    assert rl_cfg is not None
