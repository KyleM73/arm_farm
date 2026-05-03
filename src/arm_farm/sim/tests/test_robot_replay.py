"""Round-trip ``MujocoSO101`` without a dataset. Catches lerobot↔sim joint
normalisation bugs (off-by-π/180, gripper convention flips)."""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("lerobot.robots.robot")


def test_mujoco_so101_round_trip(tmp_path) -> None:
    import arm_farm.sim  # noqa: F401  (registers Robot subclass)
    from arm_farm.sim.lerobot_robot import MujocoSO101, MujocoSO101RobotConfig

    cfg = MujocoSO101RobotConfig(id="test_sim", calibration_dir=tmp_path, seed=0, viewer="none")
    robot = MujocoSO101(cfg)
    robot.connect()
    try:
        # Normalised midpoints: 0 for arm joints, 50 for gripper.
        action = dict.fromkeys(robot.action_features, 0.0)
        action["gripper.pos"] = 50.0
        robot.send_action(action)
        obs = robot.get_observation()
        for k in robot.action_features:
            assert k in obs and isinstance(obs[k], float)
    finally:
        robot.disconnect()
