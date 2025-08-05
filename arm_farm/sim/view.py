import mujoco
import mujoco.viewer
from importlib import resources
import os

def main():
    # Use importlib.resources to find the path to the 'robots' package directory.
    # This makes the script runnable from anywhere.
    robots_package_path = str(resources.files('arm_farm.robots'))
    xml = os.path.join(robots_package_path, 'scene.xml')

    try:
        model = mujoco.MjModel.from_xml_path(xml)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == '__main__':
    main()