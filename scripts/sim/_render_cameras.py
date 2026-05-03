"""Render front + wrist cameras for the Cube-Rgb task at 32x32 (policy obs)
and 256x256 (eyeball). Output: ``outputs/sim/camera_check/``."""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

import arm_farm.sim  # noqa: F401  (registers tasks)

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "sim" / "camera_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def main() -> None:
    env_cfg = load_env_cfg("Cube-Rgb", play=True)
    env_cfg.scene.num_envs = 1
    scene = Scene(env_cfg.scene, device="cpu")
    model = scene.compile()
    # MuJoCo default 0.3 leaves the overview frustums too small to read.
    model.vis.scale.camera = 0.6
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    # Cube centred in the OBJECT_RANGE xy, z=half_size so the bottom rests on the floor.
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name and "cube" in name:
                qpos_adr = int(model.jnt_qposadr[jid])
                qvel_adr = int(model.jnt_dofadr[jid])
                cube_xy_z = np.array([0.325, 0.0, 0.0125], dtype=np.float64)
                quat_wxyz = _yaw_quat_wxyz(0.0)
                data.qpos[qpos_adr : qpos_adr + 7] = np.concatenate([cube_xy_z, quat_wxyz])
                data.qvel[qvel_adr : qvel_adr + 6] = 0.0
                break

    mujoco.mj_forward(model, data)

    # Match the training renderer: geom groups 0/2/3, no sites (ee_site
    # lives at group 5 to keep it out of the policy POV).
    enabled_groups = (0, 2, 3)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    for g in enabled_groups:
        scene_option.geomgroup[g] = 1
    scene_option.sitegroup[:] = 0

    for camera, mjcf_name in [("front", "robot/front"), ("wrist", "robot/wrist")]:
        for label, hw in [("32", 32), ("256", 256)]:
            renderer = mujoco.Renderer(model, height=hw, width=hw)
            renderer.update_scene(data, camera=mjcf_name, scene_option=scene_option)
            frame = renderer.render()  # (H, W, 3) uint8
            out = OUT_DIR / f"{camera}_{label}x{label}.png"
            imageio.imwrite(out, frame)
            print(f"wrote {out}")
            renderer.close()

    # Free-camera overviews with mjVIS_CAMERA on so front/wrist appear as
    # frustum gizmos. Sites stay off so the inspection isn't cluttered.
    overview_opt = mujoco.MjvOption()
    overview_opt.geomgroup[:] = 0
    for g in enabled_groups:
        overview_opt.geomgroup[g] = 1
    overview_opt.sitegroup[:] = 0
    overview_opt.flags[int(mujoco.mjtVisFlag.mjVIS_CAMERA)] = 1

    # (azimuth_deg, elevation_deg, lookat, distance). MjvCamera azimuth: 0=+x,
    # 90=+y, 180=-x, 270=-y; elevation is pitch (negative looks down).
    # All non-top views are at -y so the wrist mount (on +y) is silhouetted.
    views = {
        "iso": (225.0, -25.0, [0.30, 0.0, 0.20], 1.10),
        "wide_iso": (300.0, -25.0, [0.35, 0.0, 0.20], 1.20),
        "top": (90.0, -85.0, [0.35, 0.0, 0.10], 1.20),
        "side": (270.0, -10.0, [0.35, 0.0, 0.20], 0.90),
        "gripper_zoom": (270.0, -5.0, [0.16, 0.0, 0.11], 0.18),
    }
    for label, (azimuth, elevation, lookat, distance) in views.items():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = np.array(lookat, dtype=np.float64)
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        renderer = mujoco.Renderer(model, height=768, width=768)
        renderer.update_scene(data, camera=cam, scene_option=overview_opt)
        frame = renderer.render()
        out = OUT_DIR / f"overview_{label}.png"
        imageio.imwrite(out, frame)
        print(f"wrote {out}")
        renderer.close()

    stitched = np.concatenate(
        [
            imageio.imread(OUT_DIR / "front_256x256.png"),
            imageio.imread(OUT_DIR / "wrist_256x256.png"),
        ],
        axis=1,
    )
    imageio.imwrite(OUT_DIR / "front+wrist_side_by_side.png", stitched)
    print(f"wrote {OUT_DIR / 'front+wrist_side_by_side.png'}")


if __name__ == "__main__":
    main()
