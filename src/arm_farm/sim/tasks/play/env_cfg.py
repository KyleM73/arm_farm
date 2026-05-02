from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from arm_farm.sim.assets.so101 import (
    EE_BODY,
    SO101_ACTION_SCALE,
    get_cube_spec,
    get_so101_cfg,
)
from arm_farm.sim.tasks.lift_cube_env_cfg import PLAY_NUM_ENVS


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    del play  # Same cfg for train and play; this task isn't trained.

    actor_terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel),
        "actions": ObservationTermCfg(func=last_action),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=SO101_ACTION_SCALE,
            use_default_offset=True,
        )
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=PLAY_NUM_ENVS,
            env_spacing=1.0,
            entities={
                "robot": get_so101_cfg(),
                "cube": EntityCfg(spec_fn=get_cube_spec),
            },
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        },
        actions=actions,
        commands={},
        events={},
        rewards={},
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=EE_BODY,
            distance=1.0,
            elevation=-20.0,
            azimuth=135.0,
        ),
        sim=SimulationCfg(
            nconmax=55,
            njmax=600,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=int(1e9),
        is_finite_horizon=False,
    )
