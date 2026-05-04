"""Custom reward terms for the SO-ARM101 lift-cube tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def object_is_lifted(
    env: ManagerBasedRlEnv,
    object_name: str,
    minimal_height: float,
) -> torch.Tensor:
    """1.0 when the object's root z exceeds ``minimal_height``, else 0.0.

    Mirrors IsaacLab's ``manipulation.lift.mdp.rewards.object_is_lifted``: a
    discrete "off-the-table" indicator that breaks the rolling local optimum
    in the staged position reward.
    """
    obj: Entity = env.scene[object_name]
    return (obj.data.root_link_pos_w[:, 2] > minimal_height).float()
