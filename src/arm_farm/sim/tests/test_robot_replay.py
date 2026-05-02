"""Round-trip the MujocoSO101 wrapper without a dataset.

Builds the wrapper, connects (headless), sends one action, reads one
observation, disconnects. Confirms the joint-position normalisation + env
plumbing line up.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("lerobot.robots.robot")


def test_mujoco_so101_round_trip(tmp_path) -> None:
    import arm_farm.sim  # noqa: F401  (registers Robot subclass)
    from arm_farm.sim.lerobot_robot import MujocoSO101, MujocoSO101RobotConfig

    cfg = MujocoSO101RobotConfig(
        id="test_sim",
        calibration_dir=tmp_path,
        seed=0,
        viewer="none",
    )
    robot = MujocoSO101(cfg)
    robot.connect()
    try:
        # Action: hold all motors at their normalised midpoint (0 for arm, 50 for gripper).
        # action_features is a dict whose keys are already `{joint}.pos`.
        action = dict.fromkeys(robot.action_features, 0.0)
        action["gripper.pos"] = 50.0
        robot.send_action(action)
        obs = robot.get_observation()
        for k in robot.action_features:
            assert k in obs, f"missing observation key {k!r}"
            assert isinstance(obs[k], float)
    finally:
        robot.disconnect()
