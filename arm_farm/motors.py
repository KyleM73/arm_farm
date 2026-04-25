from dataclasses import dataclass
from typing import Any, Literal, override
from contextlib import contextmanager
from functools import cached_property
from copy import deepcopy
from pprint import pformat
import serial
import logging


@dataclass
class CalibrationCfg:
    id: str
    mode: int
    offset: int
    range_min: int
    range_max: int


@dataclass
class MotorCfg:
    """Motor dataclass."""
    id: int
    model: str
    joint_mode: Literal["deg", "rad"]
    calibration: CalibrationCfg


@dataclass
class Motor:
    """Motor instance with id and model."""
    id: int
    model: str


@dataclass
class MotorCalibration:
    """Motor calibration data."""
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = 4095


# Constants for Feetech motors
SCAN_BAUDRATES = [9600, 57600, 115200, 1000000]
DEFAULT_BAUDRATE = 1000000
DEFAULT_TIMEOUT_MS = 20
MODEL_BAUDRATE_TABLE = {
    "sts3215": {9600: 0, 57600: 1, 115200: 2, 1000000: 3},
}
MODEL_CONTROL_TABLE = {
    "sts3215": {
        "ID": (5, 1),
        "Baud_Rate": (6, 1),
        "Torque_Enable": (40, 1),
        "Lock": (48, 1),
        "Present_Position": (56, 2),
        "Goal_Position": (42, 2),
    },
}
MODEL_ENCODING_TABLE = {}
MODEL_NUMBER_TABLE = {"sts3215": 15}
MODEL_RESOLUTION = {"sts3215": 4096}
MODEL_PROTOCOL = {"sts3215": 1}
NORMALIZED_DATA = ["Present_Position", "Goal_Position"]


def get_address(
    ctrl_table: dict[str, dict[str, tuple[int, int]]],
    model: str,
    register: str,
) -> tuple[int, int]:
    """Get address and length for a register from control table."""
    if model not in ctrl_table:
        raise ValueError(f"Model {model} not found in control table")
    if register not in ctrl_table[model]:
        raise ValueError(f"Register {register} not found for model {model}")
    return ctrl_table[model][register]


def patch_setPacketTimeout(self: Any, timeout_ms: int) -> None:
    """Patch function for setting packet timeout."""
    _ = self
    _ = timeout_ms


logger: logging.Logger = logging.getLogger(name=__name__)


class FeetechMotorsBus:
    """
    The FeetechMotorsBus class allows to efficiently read and write to
    the attached motors. It represents several motors daisy-chained
    together and connected through a serial port.

    A FeetechMotorsBus instance requires a port (e.g.
    `FeetechMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    To find the port, you can run the utility script
    `arm_farm/utils/find_port.py`

    Example of usage for 1 Feetech sts3215 motor connected to the bus:
    ```python
    bus = FeetechMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={"my_motor": Motor(1, "sts3215")},
    )
    bus.connect()

    position = bus.read("Present_Position", "my_motor", normalize=False)

    # Move from a few motor steps as an example
    few_steps = 30
    bus.write(
        "Goal_Position", "my_motor", position + few_steps, normalize=False
    )

    # When done, properly disconnect the port using
    bus.disconnect()
    ```
    """

    # Class attributes from FeetechMotorsBus
    apply_drive_mode: bool = True
    available_baudrates: list[int] = deepcopy(SCAN_BAUDRATES)
    default_baudrate: int = DEFAULT_BAUDRATE
    default_timeout: int = DEFAULT_TIMEOUT_MS
    model_baudrate_table: dict[str, dict[int, int]] = deepcopy(
        MODEL_BAUDRATE_TABLE
    )
    model_ctrl_table: dict[str, dict[str, tuple[int, int]]] = deepcopy(
        MODEL_CONTROL_TABLE
    )
    model_encoding_table: dict[str, Any] = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table: dict[str, int] = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table: dict[str, int] = deepcopy(MODEL_RESOLUTION)
    normalized_data: list[str] = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = 1,  # DEFAULT_PROTOCOL_VERSION
    ):
        self.port: str = port
        self.motors: dict[str, Motor] = motors
        self.calibration: dict[str, MotorCalibration] = (
            calibration if calibration else {}
        )
        self.protocol_version: int = protocol_version

        self.port_handler = None
        self.packet_handler = None
        self.sync_reader = None
        self.sync_writer = None
        self._comm_success = None
        self._no_error = 0x00

        self._id_to_model_dict: dict[int, str] = {
            m.id: m.model for m in self.motors.values()
        }
        self._id_to_name_dict: dict[int, str] = {
            m.id: motor for motor, m in self.motors.items()
        }
        self._model_nb_to_model_dict: dict[int, str] = {
            v: k for k, v in self.model_number_table.items()
        }

        self._validate_motors()
        self._assert_same_protocol()

        # Initialize Feetech SDK components
        try:
            import scservo_sdk as scs
            self.port_handler = scs.PortHandler(self.port)
            # HACK: monkeypatch
            if hasattr(scs, 'patch_setPacketTimeout'):
                self.port_handler.setPacketTimeout = (
                    patch_setPacketTimeout.__get__(
                        self.port_handler, scs.PortHandler
                    )
                )
            self.packet_handler = scs.PacketHandler(protocol_version)
            self.sync_reader = scs.GroupSyncRead(
                self.port_handler, self.packet_handler, 0, 0
            )
            self.sync_writer = scs.GroupSyncWrite(
                self.port_handler, self.packet_handler, 0, 0
            )
            self._comm_success = scs.COMM_SUCCESS
        except ImportError:
            logger.warning("scservo_sdk not available")

        protocol_check = any(
            MODEL_PROTOCOL.get(model, 1) != self.protocol_version
            for model in self.models
        )
        if protocol_check:
            raise ValueError("All motors must use the same protocol version")

    def __len__(self):
        return len(self.motors)

    @override
    def __repr__(self):
        motors_str = pformat(self.motors, indent=8, sort_dicts=False)
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{motors_str},\n"
            ")',\n"
        )

    @cached_property
    def _has_different_ctrl_tables(self) -> bool:
        unique_tables = set()
        for model in self.models:
            table_id = id(self.model_ctrl_table.get(model, {}))
            unique_tables.add(table_id)
        return len(unique_tables) > 1

    @cached_property
    def models(self) -> list[str]:
        return list(set(m.model for m in self.motors.values()))

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _model_nb_to_model(self, motor_nb: int) -> str:
        return self._model_nb_to_model_dict[motor_nb]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    def _get_motor_id(self, motor: str | int) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        else:
            raise ValueError(f"Invalid motor type: {type(motor)}")

    def _get_motor_model(self, motor: str | int) -> str:
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model(motor_id=motor)
        else:
            raise ValueError(f"Invalid motor type: {type(motor)}")

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors.keys())
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors
        else:
            raise ValueError(f"Invalid motors type: {type(motors)}")

    def _get_ids_values_dict(
        self, values: int | float | dict[str, int | float]
    ) -> dict[int, int | float]:
        if isinstance(values, (int, float)):
            return {m.id: values for m in self.motors.values()}
        elif isinstance(values, dict):
            return {
                self.motors[motor].id: value
                for motor, value in values.items()
            }
        else:
            raise ValueError(f"Invalid values type: {type(values)}")

    def _validate_motors(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate motor IDs found")

        # Ensure ctrl table available for all models
        for model in self.models:
            if model not in self.model_ctrl_table:
                raise ValueError(
                    f"Control table not available for model: {model}"
                )

    def _is_comm_success(self, comm: int) -> bool:
        return comm == self._comm_success

    def _is_error(self, error: int) -> bool:
        return error != self._no_error

    def _assert_motors_exist(self) -> None:
        expected_models = {
            m.id: self.model_number_table[m.model]
            for m in self.motors.values()
        }

        found_models = {}
        for id_ in self.ids:
            model_number = self.ping(id_, raise_on_error=False)
            if model_number is not None:
                found_models[id_] = model_number

        missing_ids = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found_models[id_])
            for id_ in found_models
            if expected_models.get(id_) != found_models[id_]
        }

        if missing_ids or wrong_models:
            error_msg = "Motor validation failed:"
            if missing_ids:
                error_msg += f" Missing motors: {missing_ids}"
            if wrong_models:
                error_msg += f" Wrong models: {wrong_models}"
            raise RuntimeError(error_msg)

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        if instruction_name == "sync_read" and self.protocol_version == 1:
            raise ValueError(
                "sync_read is not compatible with protocol version 1"
            )
        if instruction_name == "broadcast_ping" and self.protocol_version == 1:
            raise ValueError(
                "broadcast_ping is not compatible with protocol version 1"
            )

    def _assert_same_protocol(self) -> None:
        protocol_check = any(
            MODEL_PROTOCOL.get(model, 1) != self.protocol_version
            for model in self.models
        )
        if protocol_check:
            raise ValueError("All motors must use the same protocol version")

    def _assert_same_firmware(self) -> None:
        firmware_versions = self._read_firmware_version(
            self.ids, raise_on_error=True
        )
        if len(set(firmware_versions.values())) != 1:
            raise ValueError(
                "All motors must have the same firmware version"
            )

    @property
    def is_connected(self) -> bool:
        if self.port_handler is None:
            return False
        if hasattr(self.port_handler, 'is_open'):
            return self.port_handler.is_open
        return False

    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and initialise communication.

        Args:
            handshake (bool, optional): Pings every expected motor and
                performs additional integrity checks specific to the
                implementation. Defaults to `True`.

        Raises:
            DeviceAlreadyConnectedError: The port is already open.
            ConnectionError: The underlying SDK failed to open the port or
                the handshake did not succeed.
        """
        if self.is_connected:
            raise RuntimeError("Device already connected")

        self._connect(handshake)
        self.set_timeout()
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        try:
            if self.port_handler is None:
                raise ConnectionError("Port handler not initialized")
                
            if not self.port_handler.openPort():
                raise ConnectionError(f"Failed to open port {self.port}")

            if not self.port_handler.setBaudRate(self.default_baudrate):
                raise ConnectionError(
                    f"Failed to set baudrate {self.default_baudrate}"
                )

            if handshake:
                self._handshake()
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            raise ConnectionError(f"Failed to connect: {e}")

    def _handshake(self) -> None:
        self._assert_motors_exist()
        self._assert_same_firmware()

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port (optionally disabling torque first).

        Args:
            disable_torque (bool, optional): If `True` (default) torque is
                disabled on every motor before closing the port. This can
                prevent damaging motors if they are left applying resisting
                torque after disconnect.
        """
        if not self.is_connected:
            return

        if disable_torque:
            self.disable_torque()

        if self.port_handler is not None:
            self.port_handler.closePort()
        logger.debug(f"{self.__class__.__name__} disconnected.")

    @classmethod
    def scan_port(
        cls, port: str, *args: Any, **kwargs: Any
    ) -> dict[int, list[int]]:
        # Implementation would go here
        _ = port
        _ = args
        _ = kwargs
        return {}

    def setup_motor(
        self,
        motor: str,
        initial_baudrate: int | None = None,
        initial_id: int | None = None,
    ) -> None:
        """Assign the correct ID and baud-rate to a single motor."""
        if not self.is_connected:
            raise RuntimeError("Bus not connected")

        if initial_baudrate is None:
            initial_baudrate, initial_id = self._find_single_motor(
                motor, initial_baudrate
            )

        if initial_id is None:
            _, initial_id = self._find_single_motor(motor, initial_baudrate)

        model = self.motors[motor].model
        target_id = self.motors[motor].id
        self.set_baudrate(initial_baudrate)
        self._disable_torque(initial_id, model)

        # Set ID
        addr, length = get_address(self.model_ctrl_table, model, "ID")
        _ = self._write(addr, length, initial_id, target_id)

        # Set Baudrate
        addr, length = get_address(
            self.model_ctrl_table, model, "Baud_Rate"
        )
        baudrate_value = self.model_baudrate_table[model][
            self.default_baudrate
        ]
        _ = self._write(addr, length, target_id, baudrate_value)

        self.set_baudrate(self.default_baudrate)

    def _find_single_motor(
        self, motor: str, initial_baudrate: int | None = None
    ) -> tuple[int, int]:
        if self.protocol_version == 0:
            return self._find_single_motor_p0(motor, initial_baudrate)
        else:
            return self._find_single_motor_p1(motor, initial_baudrate)

    def _find_single_motor_p0(
        self, motor: str, initial_baudrate: int | None = None
    ) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None
            else self.model_baudrate_table[model]
        )

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            # Implementation would scan for the motor
            pass

        raise RuntimeError(
            f"Motor '{motor}' (model '{model}') was not found. "
            + "Make sure it is connected."
        )

    def _find_single_motor_p1(
        self, motor: str, initial_baudrate: int | None = None
    ) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None
            else self.model_baudrate_table[model]
        )

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            # Implementation would scan for the motor
            pass

        raise RuntimeError(
            f"Motor '{motor}' (model '{model}') was not found. "
            + "Make sure it is connected."
        )

    def configure_motors(
        self,
        return_delay_time: int = 0,
        maximum_acceleration: int = 254,
        acceleration: int = 254,
    ) -> None:
        for motor in self.motors:
            # Configure each motor with the specified parameters
            _ = motor

    @property
    def is_calibrated(self) -> bool:
        return len(self.calibration) == len(self.motors)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        # Implementation would read calibration data from motors
        calibration = {}
        for motor, m in self.motors.items():
            # Read homing offset, min/max positions
            calibration[motor] = MotorCalibration()
        return calibration

    def write_calibration(
        self,
        calibration_dict: dict[str, MotorCalibration],
        cache: bool = True,
    ) -> None:
        for motor, calibration in calibration_dict.items():
            # Write calibration data to motor
            _ = motor
            _ = calibration

        if cache:
            self.calibration.update(calibration_dict)

    def _get_half_turn_homings(
        self, positions: dict[str | int, int | float]
    ) -> dict[str | int, int | float]:
        """
        On Feetech Motors:
        Present_Position = Actual_Position - Homing_Offset
        """
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            resolution = self.model_resolution_table.get(model, 4096)
            half_turn_homings[motor] = pos - (resolution // 2)
        return half_turn_homings

    def disable_torque(
        self,
        motors: str | list[str] | None = None,
        num_retry: int = 0,
    ) -> None:
        for motor in self._get_motors_list(motors):
            motor_id = self.motors[motor].id
            model = self.motors[motor].model
            self._disable_torque(motor_id, model, num_retry)

    def _disable_torque(
        self, motor_id: int, model: str, num_retry: int = 0
    ) -> None:
        addr, length = get_address(
            self.model_ctrl_table, model, "Torque_Enable"
        )
        # TorqueMode.DISABLED.value
        _ = self._write(addr, length, motor_id, 0, num_retry=num_retry)
        addr, length = get_address(self.model_ctrl_table, model, "Lock")
        _ = self._write(addr, length, motor_id, 0, num_retry=num_retry)

    def enable_torque(
        self,
        motors: str | list[str] | None = None,
        num_retry: int = 0,
    ) -> None:
        for motor in self._get_motors_list(motors):
            motor_id = self.motors[motor].id
            model = self.motors[motor].model
            addr, length = get_address(
                self.model_ctrl_table, model, "Torque_Enable"
            )
            # TorqueMode.ENABLED.value
            _ = self._write(addr, length, motor_id, 1, num_retry=num_retry)

    @contextmanager
    def torque_disabled(
        self, motors: str | list[str] | None = None
    ):
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_timeout(self, timeout_ms: int | None = None):
        """Change the packet timeout used by the SDK."""
        timeout_ms = (
            timeout_ms if timeout_ms is not None else self.default_timeout
        )
        if (self.port_handler is not None and
                hasattr(self.port_handler, 'setPacketTimeoutMillis')):
            self.port_handler.setPacketTimeoutMillis(timeout_ms)

    def get_baudrate(self) -> int:
        """Return the current baud-rate configured on the port."""
        if self.port_handler is not None:
            return self.port_handler.getBaudRate()
        return 0

    def set_baudrate(self, baudrate: int) -> None:
        """Set a new UART baud-rate on the port."""
        if self.port_handler is None:
            raise RuntimeError("Port handler not initialized")

        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            if not self.port_handler.setBaudRate(baudrate):
                raise RuntimeError(f"Failed to set baudrate to {baudrate}")

    def reset_calibration(
        self,
        motors: str | int | list[str | int] | None = None,
    ) -> None:
        """Restore factory calibration for the selected motors."""
        if motors is None:
            motor_list = list(self.motors.keys())
        elif isinstance(motors, (str, int)):
            motor_list = [motors]
        elif isinstance(motors, list):
            motor_list = motors
        else:
            raise ValueError(f"Invalid motors type: {type(motors)}")

        for motor in motor_list:
            # Reset calibration logic here
            _ = motor

        self.calibration = {}

    def set_half_turn_homings(
        self,
        motors: str | int | list[str | int] | None = None,
    ) -> dict[str | int, int | float]:
        """Centre each motor range around its current position."""
        if motors is None:
            motor_list = list(self.motors.keys())
        elif isinstance(motors, (str, int)):
            motor_list = [motors]
        elif isinstance(motors, list):
            motor_list = motors
        else:
            raise ValueError(f"Invalid motors type: {type(motors)}")

        self.reset_calibration(motor_list)
        actual_positions = self.sync_read(
            "Present_Position", motor_list, normalize=False
        )
        homing_offsets = self._get_half_turn_homings(actual_positions)
        for motor, offset in homing_offsets.items():
            # Write homing offset
            _ = motor
            _ = offset

        return homing_offsets

    def record_ranges_of_motion(
        self,
        motors: str | int | list[str | int] | None = None,
        display_values: bool = True,
    ) -> tuple[
        dict[str | int, int | float],
        dict[str | int, int | float],
    ]:
        """Interactively record the min/max encoder values of each motor."""
        if motors is None:
            motor_list = list(self.motors.keys())
        elif isinstance(motors, (str, int)):
            motor_list = [motors]
        elif isinstance(motors, list):
            motor_list = motors
        else:
            raise ValueError(f"Invalid motors type: {type(motors)}")

        start_positions = self.sync_read(
            "Present_Position", motor_list, normalize=False
        )
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            # Implementation for recording ranges
            _ = display_values
            user_pressed_enter = True  # Placeholder

        same_min_max = [
            motor for motor in motor_list
            if mins.get(str(motor)) == maxes.get(str(motor))
        ]
        if same_min_max:
            logger.warning(f"No motion detected for motors: {same_min_max}")

        return mins, maxes

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError("No calibration available")

        normalized_values = {}
        for id_, val in ids_values.items():
            # Normalization logic here
            motor_name = self._id_to_name(id_)
            _ = motor_name  # Use motor_name for normalization logic
            normalized_values[id_] = float(val)

        return normalized_values

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError("No calibration available")

        unnormalized_values = {}
        for id_, val in ids_values.items():
            # Unnormalization logic here
            motor_name = self._id_to_name(id_)
            _ = motor_name  # Use motor_name for unnormalization logic
            unnormalized_values[id_] = int(val)

        return unnormalized_values

    def _encode_sign(
        self, data_name: str, ids_values: dict[int, int]
    ) -> dict[int, int]:
        for id_ in ids_values:
            # Sign encoding logic specific to data_name and model
            pass
        return ids_values

    def _decode_sign(
        self, data_name: str, ids_values: dict[int, int]
    ) -> dict[int, int]:
        for id_ in ids_values:
            # Sign decoding logic specific to data_name and model
            pass
        return ids_values

    def _serialize_data(self, value: int, length: int) -> list[int]:
        """Convert unsigned integer value into list of byte-sized integers."""
        if value < 0:
            raise ValueError("Value must be non-negative")

        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
        if max_value is None:
            raise ValueError(f"Unsupported length: {length}")

        if value > max_value:
            raise ValueError(
                f"Value {value} exceeds maximum for length {length}"
            )

        return self._split_into_byte_chunks(value, length)

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        # Little-endian byte splitting for Feetech
        chunks = []
        for i in range(length):
            chunks.append((value >> (8 * i)) & 0xFF)
        return chunks

    def ping(
        self,
        motor: str | int,
        num_retry: int = 0,
        raise_on_error: bool = False,
    ) -> int | None:
        """Ping a single motor and return its model number."""
        if self.packet_handler is None or self.port_handler is None:
            if raise_on_error:
                raise RuntimeError("SDK not initialized")
            return None

        id_ = self._get_motor_id(motor)
        for n_try in range(1 + num_retry):
            model_number, comm, error = self.packet_handler.ping(
                self.port_handler, id_
            )
            if self._is_comm_success(comm) and not self._is_error(error):
                return model_number

        if raise_on_error:
            raise RuntimeError(f"Failed to ping motor {motor}")
        return None

    def broadcast_ping(
        self, num_retry: int = 0, raise_on_error: bool = False
    ) -> dict[int, int] | None:
        """Broadcast ping to discover all motors on the bus."""
        self._assert_protocol_is_compatible("broadcast_ping")
        
        for n_try in range(1 + num_retry):
            data_list, comm = self._broadcast_ping()
            if self._is_comm_success(comm):
                return data_list

        if raise_on_error:
            raise RuntimeError("Broadcast ping failed")
        return None

    def _broadcast_ping(self) -> tuple[dict[int, int], int]:
        # Implementation of broadcast ping protocol
        data_list = {}
        comm = self._comm_success if self._comm_success is not None else 0
        return data_list, comm

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> int | float:
        """Read a register from a motor."""
        if not self.is_connected:
            raise RuntimeError("Bus not connected")

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = (
            f"Failed to read '{data_name}' on {id_=} "
            f"after {num_retry + 1} tries."
        )
        value, _, _ = self._read(
            addr, length, id_, num_retry=num_retry,
            raise_on_error=True, err_msg=err_msg
        )

        id_value = self._decode_sign(data_name, {id_: value})

        if normalize and data_name in self.normalized_data:
            id_value = self._normalize(id_value)

        return id_value[id_]

    def _read(
        self,
        address: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int, int]:
        if self.packet_handler is None or self.port_handler is None:
            if raise_on_error:
                raise RuntimeError("SDK not initialized")
            return 0, 0, 1

        comm = 0
        error = 1
        for n_try in range(1 + num_retry):
            if length == 1:
                value, comm, error = self.packet_handler.read1ByteTxRx(
                    self.port_handler, motor_id, address
                )
            elif length == 2:
                value, comm, error = self.packet_handler.read2ByteTxRx(
                    self.port_handler, motor_id, address
                )
            elif length == 4:
                value, comm, error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, motor_id, address
                )
            else:
                raise ValueError(f"Unsupported read length: {length}")

            if self._is_comm_success(comm) and not self._is_error(error):
                return value, comm, error

        if raise_on_error:
            raise RuntimeError(
                err_msg or f"Read failed after {num_retry + 1} tries"
            )

        return 0, comm, error

    def write(
        self,
        data_name: str,
        motor: str,
        value: int | float,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write a value to a single motor's register."""
        if not self.is_connected:
            raise RuntimeError("Bus not connected")

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        if normalize and data_name in self.normalized_data:
            value = self._unnormalize({id_: value})[id_]

        value = self._encode_sign(data_name, {id_: int(value)})[id_]

        err_msg = (
            f"Failed to write '{data_name}' on {id_=} with '{value}' "
            f"after {num_retry + 1} tries."
        )
        self._write(
            addr, length, id_, value, num_retry=num_retry,
            raise_on_error=True, err_msg=err_msg
        )

    def _write(
        self,
        addr: int,
        length: int,
        motor_id: int,
        value: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        if self.packet_handler is None or self.port_handler is None:
            if raise_on_error:
                raise RuntimeError("SDK not initialized")
            return 0, 1

        comm = 0
        error = 1
        for _ in range(1 + num_retry):
            if length == 1:
                comm, error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, addr, value
                )
            elif length == 2:
                comm, error = self.packet_handler.write2ByteTxRx(
                    self.port_handler, motor_id, addr, value
                )
            elif length == 4:
                comm, error = self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, addr, value
                )
            else:
                raise ValueError(f"Unsupported write length: {length}")

            if self._is_comm_success(comm) and not self._is_error(error):
                return comm, error

        if raise_on_error:
            raise RuntimeError(
                err_msg or f"Write failed after {num_retry + 1} tries"
            )

        return comm, error

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, int | float]:
        """Read the same register from several motors at once."""
        if not self.is_connected:
            raise RuntimeError("Bus not connected")

        self._assert_protocol_is_compatible("sync_read")

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]
        models = [self.motors[motor].model for motor in names]

        if self._has_different_ctrl_tables:
            raise ValueError(
                "Cannot sync read motors with different control tables"
            )

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = (
            f"Failed to sync read '{data_name}' on {ids=} "
            f"after {num_retry + 1} tries."
        )
        ids_values, _ = self._sync_read(
            addr, length, ids, num_retry=num_retry,
            raise_on_error=True, err_msg=err_msg
        )

        ids_values = self._decode_sign(data_name, ids_values)

        if normalize and data_name in self.normalized_data:
            ids_values = self._normalize(ids_values)

        return {
            self._id_to_name(id_): value
            for id_, value in ids_values.items()
        }

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[dict[int, int], int]:
        if self.sync_reader is None:
            if raise_on_error:
                raise RuntimeError("Sync reader not initialized")
            return {}, 0

        self._setup_sync_reader(motor_ids, addr, length)
        comm = 0
        for _ in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break

        if not self._is_comm_success(comm) and raise_on_error:
            raise RuntimeError(
                err_msg or f"Sync read failed after {num_retry + 1} tries"
            )

        values = {
            id_: self.sync_reader.getData(id_, addr, length)
            for id_ in motor_ids
        }
        return values, comm

    def _setup_sync_reader(
        self, motor_ids: list[int], addr: int, length: int
    ) -> None:
        if self.sync_reader is None:
            raise RuntimeError("Sync reader not initialized")
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length = length
        for id_ in motor_ids:
            self.sync_reader.addParam(id_)

    def sync_write(
        self,
        data_name: str,
        values: int | float | dict[str, int | float],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write the same register on multiple motors."""
        if not self.is_connected:
            raise RuntimeError("Bus not connected")

        ids_values = self._get_ids_values_dict(values)
        models = [self._id_to_model(id_) for id_ in ids_values]
        if self._has_different_ctrl_tables:
            raise ValueError(
                "Cannot sync write motors with different control tables"
            )

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        if normalize and data_name in self.normalized_data:
            ids_values = self._unnormalize(ids_values)

        int_ids_values = {
            id_: int(value) for id_, value in ids_values.items()
        }
        int_ids_values = self._encode_sign(data_name, int_ids_values)

        err_msg = (
            f"Failed to sync write '{data_name}' with {int_ids_values=} "
            f"after {num_retry + 1} tries."
        )
        _ = self._sync_write(
            addr, length, int_ids_values, num_retry=num_retry,
            raise_on_error=True, err_msg=err_msg
        )

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> int:
        if self.sync_writer is None:
            if raise_on_error:
                raise RuntimeError("Sync writer not initialized")
            return 0

        self._setup_sync_writer(ids_values, addr, length)
        comm = 0
        for _ in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._is_comm_success(comm):
                break

        if not self._is_comm_success(comm) and raise_on_error:
            raise RuntimeError(
                err_msg or f"Sync write failed after {num_retry + 1} tries"
            )

        return comm

    def _setup_sync_writer(
        self, ids_values: dict[int, int], addr: int, length: int
    ) -> None:
        if self.sync_writer is None:
            raise RuntimeError("Sync writer not initialized")
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length = length
        for id_, value in ids_values.items():
            data = self._serialize_data(value, length)
            self.sync_writer.addParam(id_, data)

    def _read_firmware_version(
        self, motor_ids: list[int], raise_on_error: bool = False
    ) -> dict[int, str]:
        # Implementation for reading firmware versions
        firmware_versions = {}
        for motor_id in motor_ids:
            # Read firmware version from motor
            firmware_versions[motor_id] = "1.0.0"  # Placeholder
        return firmware_versions

    def _read_model_number(
        self, motor_ids: list[int], raise_on_error: bool = False
    ) -> dict[int, int]:
        # Implementation for reading model numbers
        model_numbers: dict[int, int] = {}
        for motor_id in motor_ids:
            model_number: int | None = self.ping(
                motor=motor_id, raise_on_error=raise_on_error
            )
            if model_number is not None:
                model_numbers[motor_id] = model_number
        return model_numbers
