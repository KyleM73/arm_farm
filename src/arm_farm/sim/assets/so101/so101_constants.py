"""SO-ARM101 constants for mjlab.

PD gains + effort limit are leisaac defaults; MJCF ``sts3215`` armature /
frictionloss are kept by leaving actuator-cfg overrides at None. Action
scale is ``0.5 * effort / stiffness`` ≈ ±4.7° per step at ±1 — twice the
YAM default to shorten the credit-assignment horizon during early
exploration, while still leaving ~50% actuator-torque headroom.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CameraCfg

SO101_DIR: Path = Path(__file__).parent
SCENE_XML: Path = SO101_DIR / "xmls" / "so101_scene.xml"

# Order matches SOFollower motor IDs 1..6 so recorded LeRobot datasets
# replay without remapping.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_JOINT: str = "gripper"

# `EE_BODY` hosts the wrist camera + ee_site; `EE_SITE` is the grasp point
# added programmatically in `get_spec()`.
EE_SITE: str = "ee_site"
EE_BODY: str = "gripper"

SO101_STIFFNESS: float = 17.8  # P gain (N·m / rad)
SO101_DAMPING: float = 0.60  # D gain (N·m·s / rad)
SO101_EFFORT_LIMIT: float = 2.94  # N·m, matches MJCF forcerange


def _add_ee_site(spec: mujoco.MjSpec) -> None:
    # Grasp point sits ~10 cm below the gripper origin, between the jaws.
    # ``group=5`` keeps it out of geom-group 0/2/3 so camera sensors don't
    # feed an ee_site blob into the policy; debug viewers can re-enable it.
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
    """Cube entity matching mjlab YAM's layout."""
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


# Wrist camera mirrors the wowrobo SO-ARM USB mount: lateral on the gripper,
# lens pointing along -z toward the grasp site at gripper offset (0,0,-0.10).
# Tune alongside the real mount via ``preview --task=Cube-Rgb``;
# ``scripts/sim/_render_cameras.py`` dumps still POV frames.
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


# Folded-back rest pose; keeps the gripper off the table at reset.
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


SO101_ACTION_SCALE: dict[str, float] = dict.fromkeys(ARM_JOINTS, 0.5 * SO101_EFFORT_LIMIT / SO101_STIFFNESS)


if __name__ == "__main__":
    # Tune home pose / ee_site:
    # ``uv run python -m arm_farm.sim.assets.so101.so101_constants``
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_so101_cfg())
    viewer.launch(robot.spec.compile())
