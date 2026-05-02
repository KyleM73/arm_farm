"""SO-ARM101 mjlab sim, wrapped as a lerobot ``Robot``.

The wrapper builds a tiny mjlab env on ``connect()`` (reuses the ``Play``
task's env config but switches the joint position action term to absolute
mode) and exposes joint positions in lerobot's normalised motor units.
``lerobot-replay --robot.type=mujoco_so101`` then drives it with recorded
SO-ARM101 datasets without any remapping.

Two viewer backends are supported (selected via ``--robot.viewer=...``):
``native`` for MuJoCo's OpenGL window, ``viser`` for a web viewer the user
opens in a browser. ``none`` skips the viewer entirely.

Cameras are not surfaced yet: ``observation_features`` is joint positions
only. Add ``CameraSensorCfg``s to the env and extend ``get_observation`` once
visual sim2real is in scope.
"""

from __future__ import annotations

import contextlib
import logging
from functools import cached_property
from typing import Any

import mujoco.viewer
import torch
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from arm_farm.sim.assets.so101 import ARM_JOINTS, GRIPPER_JOINT
from arm_farm.sim.lerobot_robot.config_mujoco_so101 import MujocoSO101RobotConfig
from arm_farm.sim.tasks.play.env_cfg import make_env_cfg as _make_play_env_cfg

logger = logging.getLogger(__name__)


def _norm_to_rad(joint: str, normalized: float, lo: float, hi: float) -> float:
    if joint == GRIPPER_JOINT:
        return lo + (normalized / 100.0) * (hi - lo)
    return lo + ((normalized + 100.0) / 200.0) * (hi - lo)


def _rad_to_norm(joint: str, rad: float, lo: float, hi: float) -> float:
    if joint == GRIPPER_JOINT:
        return ((rad - lo) / (hi - lo)) * 100.0
    return ((rad - lo) / (hi - lo)) * 200.0 - 100.0


def _build_replay_env_cfg(decimation: int) -> ManagerBasedRlEnvCfg:
    """Reuse the ``Play`` task's scene + sim, but swap to absolute joint targets.

    With ``scale=1.0`` and ``use_default_offset=False`` each incoming action is
    interpreted as a joint configuration in radians, matching how
    ``lerobot-replay`` feeds back recorded targets.
    """
    cfg = _make_play_env_cfg(play=True)
    # Replay drives one env from a single recorded trajectory; clamp back
    # to 1 even though the Play task defaults to a 16-env viser grid.
    cfg.scene.num_envs = 1
    cfg.decimation = decimation
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = 1.0
    action.use_default_offset = False
    return cfg


class MujocoSO101(Robot):
    """SO-ARM101 simulated via mjlab.

    Action and observation values are in lerobot's normalised motor units
    (arm joints: [-100, 100]; gripper: [0, 100]). The wrapper converts
    to/from sim radians using the joint ranges read from the compiled
    ``mujoco.MjModel`` so any MJCF edit is picked up automatically.
    """

    config_class = MujocoSO101RobotConfig
    name = "mujoco_so101"

    def __init__(self, config: MujocoSO101RobotConfig):
        super().__init__(config)
        self.config = config
        self._env: ManagerBasedRlEnv | None = None
        self._native_viewer: Any = None  # mujoco.viewer.Handle
        self._viser_scene: Any = None  # mjviser.ViserMujocoScene
        self._viser_server: Any = None  # viser.ViserServer
        self._joint_idx: dict[str, int] = {}
        self._joint_ranges: dict[str, tuple[float, float]] = {}
        self._action_tensor: torch.Tensor | None = None

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        return {f"{j}.pos": float for j in ARM_JOINTS}

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {f"{j}.pos": float for j in ARM_JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._env is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # interface signature; unused by the sim wrapper
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env_cfg = _build_replay_env_cfg(decimation=self.config.decimation)
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        env.reset(seed=self.config.seed)

        # Cache joint name → index and joint ranges from the entity so
        # send_action / get_observation don't hit O(n) lookups per step.
        robot = env.scene["robot"]
        self._joint_idx = {name: i for i, name in enumerate(robot.joint_names)}
        limits = robot.data.joint_pos_limits[0]  # (n_joints, 2)
        self._joint_ranges = {
            name: (float(limits[self._joint_idx[name], 0]), float(limits[self._joint_idx[name], 1]))
            for name in ARM_JOINTS
        }
        self._action_tensor = torch.zeros(1, len(ARM_JOINTS), device=env.device)

        self._env = env
        self._attach_viewer(env)

    def _attach_viewer(self, env: ManagerBasedRlEnv) -> None:
        backend = self.config.viewer
        if backend == "none":
            return
        if backend == "native":
            # mjwarp's `sim.step` syncs GPU state back into mj_data, so a
            # passive Handle bound to (mj_model, mj_data) reflects the
            # current rollout without any custom mjlab viewer machinery.
            self._native_viewer = mujoco.viewer.launch_passive(env.sim.mj_model, env.sim.mj_data)
            return
        if backend == "viser":
            import viser
            from mjviser import ViserMujocoScene

            self._viser_server = viser.ViserServer()
            self._viser_scene = ViserMujocoScene(self._viser_server, env.sim.mj_model, num_envs=1)
            self._viser_scene.update_from_mjdata(env.sim.mj_data)
            logger.info("MujocoSO101 viser server ready: %s", self._viser_server)
            return
        raise ValueError(f"Unknown viewer backend: {backend!r}")

    def disconnect(self) -> None:
        if self._native_viewer is not None:
            with contextlib.suppress(Exception):
                self._native_viewer.close()
        if self._viser_server is not None:
            with contextlib.suppress(Exception):
                self._viser_server.stop()
        if self._env is not None:
            with contextlib.suppress(Exception):
                self._env.close()
        self._env = None
        self._native_viewer = None
        self._viser_scene = None
        self._viser_server = None
        self._joint_idx = {}
        self._joint_ranges = {}
        self._action_tensor = None

    def get_observation(self) -> RobotObservation:
        if self._env is None:
            raise RuntimeError("MujocoSO101 is not connected; call connect() first.")
        qpos = self._env.scene["robot"].data.joint_pos[0]
        return {
            f"{j}.pos": _rad_to_norm(j, float(qpos[self._joint_idx[j]]), *self._joint_ranges[j])
            for j in ARM_JOINTS
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._env is None or self._action_tensor is None:
            raise RuntimeError("MujocoSO101 is not connected; call connect() first.")
        for i, j in enumerate(ARM_JOINTS):
            self._action_tensor[0, i] = _norm_to_rad(j, float(action[f"{j}.pos"]), *self._joint_ranges[j])
        self._env.step(self._action_tensor)
        if self._native_viewer is not None:
            with contextlib.suppress(Exception):
                self._native_viewer.sync()
        if self._viser_scene is not None:
            with contextlib.suppress(Exception):
                self._viser_scene.update_from_mjdata(self._env.sim.mj_data)
        return action
