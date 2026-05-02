"""Tiny rollout against the registered ``Cube`` env to catch obvious wiring bugs.

Single env, 5 zero-action steps, just confirms reset() and step() both run
without raising and the obs dict has the expected keys.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")


def test_cube_zero_action_rollout() -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import arm_farm.sim  # noqa: F401  (registers tasks)

    cfg = load_env_cfg("Cube", play=True)
    cfg.scene.num_envs = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        obs, _info = env.reset(seed=0)
        assert "actor" in obs, f"actor obs group missing: {list(obs)}"
        action = torch.zeros(1, env.action_space.shape[-1], device=env.device)
        for _ in range(5):
            obs, reward, _terminated, _truncated, _info = env.step(action)
            assert torch.isfinite(reward).all(), f"non-finite reward: {reward}"
            actor_obs = obs["actor"]
            assert isinstance(actor_obs, torch.Tensor)
            assert torch.isfinite(actor_obs).all(), "non-finite actor obs"
    finally:
        env.close()
