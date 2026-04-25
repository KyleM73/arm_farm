from dataclasses import dataclass

@dataclass(kw_only=True)
class CameraCfg:
    type: str
    fps: int | None = None
    width: int | None = None
    height: int | None = None
