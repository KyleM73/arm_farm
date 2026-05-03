"""Load ``<repo>/.env`` and default ``HF_LEROBOT_HOME``; existing env vars win."""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def load_env() -> None:
    root = _repo_root()
    env_path = root / ".env"
    if env_path.is_file():
        for key, val in _parse_env_file(env_path).items():
            os.environ.setdefault(key, val)
    os.environ.setdefault("HF_LEROBOT_HOME", str(root / "data"))
