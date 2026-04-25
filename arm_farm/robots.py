"""Heavily inspired from Huggingface:lerobot"""

import os
from pathlib import Path
from dataclasses import dataclass, field

from typing import Any, Literal, override

from arm_farm import DIR_PATH, CalibrationCfg, CameraCfg

@dataclass
class ArmCfg:
    name: str
    id: str
    port: str
    soft_joint_limit: float | list[float] | None
    disable_torque_on_disconnect: bool = True
    cameras: dict[str, CameraCfg] = field(default_factory=dict)
    joint_mode: Literal["deg", "rad"] = "rad"

class Arm:
    name: str
    def __init__(self, config: ArmCfg) -> None:
        self.cfg: ArmCfg = config
        self.name = config.name
        self.id: str = config.id
        self.calibration_dir: Path = Path(os.path.join(DIR_PATH, "workspace"))
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath: Path = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, CalibrationCfg] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()
    
    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}:{self.id}"
    
    @property
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        pass

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the robot.

        Args:
            calibrate (bool): If True, automatically calibrate the robot after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        pass

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        pass

    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to load calibration data from the specified file.

        Args:
            fpath (Path | None): Optional path to the calibration file. Defaults to `self.calibration_fpath`.
        """
        fpath: Path = self.calibration_fpath if fpath is None else fpath
        from omegaconf import OmegaConf
        self.calibration = OmegaConf.load(file_=fpath)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to save calibration data to the specified file.

        Args:
            fpath (Path | None): Optional path to save the calibration file. Defaults to `self.calibration_fpath`.
        """
        fpath: Path = self.calibration_fpath if fpath is None else fpath
        from omegaconf import OmegaConf
        OmegaConf.save(config=self.calibration, f=fpath)

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state. Its structure
                should match :pymeth:`observation_features`.
        """

        pass

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action. Its structure should match
                :pymeth:`action_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        pass

    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        pass

if __name__ == "__main__":
    cfg: ArmCfg = ArmCfg(name="test", id="test", port="/dev/null", soft_joint_limit=None)
    robot: Arm = Arm(config=cfg)
    print(robot)