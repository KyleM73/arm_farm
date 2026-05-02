"""SO-ARM101 constants for mjlab.

PD gains and effort limits use the leisaac-derived defaults (stiffness=17.8,
damping=0.60, effort=2.94 N·m). The MJCF's `sts3215` joint defaults
(armature=0.028, frictionloss=0.052) are kept by leaving the actuator-cfg
overrides as None.

Action scale follows the YAM recipe — ``0.25 * effort_limit / stiffness`` —
which keeps a policy output of ±1 inside roughly ±2.4° of relative joint motion
per step.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CameraCfg

SO101_DIR: Path = Path(__file__).parent
SCENE_XML: Path = SO101_DIR / "xmls" / "so101_scene.xml"

# 6 actuated joints. Order matches the SOFollower motor IDs (1..6) on the real arm
# so recorded LeRobot datasets can be replayed without remapping.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_JOINT: str = "gripper"

# Site + body the lift-cube task and the lerobot Robot wrapper key off of.
# `EE_BODY` is the body where the wrist camera + ee_site live; `EE_SITE` is
# the grasp-point site added programmatically in `get_spec()` below.
EE_SITE: str = "ee_site"
EE_BODY: str = "gripper"

# STS3215 servo, conservative position-controlled defaults.
SO101_STIFFNESS: float = 17.8  # P gain (N·m / rad)
SO101_DAMPING: float = 0.60  # D gain (N·m·s / rad)
SO101_EFFORT_LIMIT: float = 2.94  # N·m, matches MJCF forcerange


def _add_ee_site(spec: mujoco.MjSpec) -> None:
    # Grasp point sits ~10 cm below the gripper body origin, between the static
    # jaw and the moving jaw. Refine in the viewer alongside the real
    # RealSense / D405 mount once one is attached.
    #
    # ``group=5`` keeps the site out of the default-visible group set so the
    # camera sensors (which render geom groups 0/2/3) do not feed an ee_site
    # blob into the policy. The native / viser debug viewers can re-enable
    # group 5 to make it visible during inspection.
    gripper = spec.body(EE_BODY)
    gripper.add_site(
        name=EE_SITE,
        pos=(0.0, 0.0, -0.10),
        quat=(1.0, 0.0, 0.0, 0.0),
        group=5,
    )


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    _add_ee_site(spec)
    return spec


def get_cube_spec(
    cube_size: float = 0.0125,
    mass: float = 0.05,
    rgba: tuple[float, float, float, float] = (0.85, 0.15, 0.15, 1.0),
) -> mujoco.MjSpec:
    """Standalone cube entity matching mjlab YAM's get_cube_spec layout."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(cube_size, cube_size, cube_size),
        mass=mass,
        rgba=rgba,
    )
    return spec


# Wrist camera attached to the gripper body. Mount placement is intended to
# mirror the wowrobo SO-ARM USB camera (lateral mount on the gripper, lens
# pointing along the gripper toward the jaws / grasp site).
#
# Frame convention (gripper-local): -z points toward the grasp site at gripper
# offset (0, 0, -0.10) — i.e. "forward" along the gripper. ``pos`` is in
# gripper-local coordinates; ``quat`` is (w, x, y, z) and rotates the camera
# frame relative to the gripper. Tune both alongside the real mount in the
# native viewer (``arm-farm-play-native --task=Cube-Rgb``); use
# ``scripts/sim/_render_cameras.py`` to dump still PNGs of the camera POV.
WRIST_CAMERA = CameraCfg(
    name="wrist",
    body=EE_BODY,
    fovy=58.0,
    pos=(0.0, 0.06, -0.02),
    quat=(0.6830127, -0.1830127, -0.1830127, -0.6830127),
)


ACTUATORS: tuple[BuiltinPositionActuatorCfg, ...] = tuple(
    BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        stiffness=SO101_STIFFNESS,
        damping=SO101_DAMPING,
        effort_limit=SO101_EFFORT_LIMIT,
    )
    for name in ARM_JOINTS
)


ARTICULATION = EntityArticulationInfoCfg(
    actuators=ACTUATORS,
    soft_joint_pos_limit_factor=0.95,
)


# Folded-back rest pose. Shoulder-lift down + elbow-flex bent keeps the
# gripper above the table on reset so the arm doesn't tip into the floor
# under gravity before the policy takes its first step. Refine in the
# viewer alongside the real arm's calibrated rest pose.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "shoulder_pan": 0.0,
        "shoulder_lift": -1.0,
        "elbow_flex": 1.4,
        "wrist_flex": 0.5,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_so101_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        init_state=HOME_KEYFRAME,
        articulation=ARTICULATION,
        cameras=(WRIST_CAMERA,),
    )


SO101_ACTION_SCALE: dict[str, float] = dict.fromkeys(ARM_JOINTS, 0.25 * SO101_EFFORT_LIMIT / SO101_STIFFNESS)


if __name__ == "__main__":
    # `uv run python -m arm_farm.sim.assets.so101.so101_constants` opens the
    # MJCF in MuJoCo's native viewer. Useful for tuning home pose and ee_site.
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_so101_cfg())
    viewer.launch(robot.spec.compile())
