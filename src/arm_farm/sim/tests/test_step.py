"""End-to-end Cube rollout via mjlab's ManagerBasedRlEnv. Catches NaN rewards
or obs from arm_farm-defined reward terms."""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")


def test_cube_zero_action_rollout() -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import arm_farm.sim  # noqa: F401

    cfg = load_env_cfg("Cube", play=True)
    cfg.scene.num_envs = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        obs, _info = env.reset(seed=0)
        action = torch.zeros(1, env.action_space.shape[-1], device=env.device)
        for _ in range(5):
            obs, reward, *_ = env.step(action)
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["actor"]).all()
    finally:
        env.close()
