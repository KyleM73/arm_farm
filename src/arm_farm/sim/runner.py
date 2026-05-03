"""ManipulationOnPolicyRunner with per-checkpoint ONNX naming and verbose
ONNX-export errors. Upstream overwrites a single .onnx and swallows export
failures with ``repr(exc)``; we mirror the .pt stem and print tracebacks."""

from __future__ import annotations

import traceback
from pathlib import Path

import wandb
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner


class ArmFarmManipulationRunner(ManipulationOnPolicyRunner):
    @staticmethod
    def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
        pt_path = Path(checkpoint_path)
        onnx_path = pt_path.with_suffix(".onnx")
        return pt_path.parent, onnx_path.name, onnx_path

    def save(self, path: str, infos=None) -> None:  # type: ignore[override]
        # .pt is mandatory; ONNX path is best-effort.
        super(ManipulationOnPolicyRunner, self).save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            run_name: str = (
                wandb.run.name
                if self.logger.logger_type == "wandb" and wandb.run and wandb.run.name
                else "local"
            )
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception:
            print(f"[arm-farm] ONNX export failed for {path}; .pt was saved. Traceback:")
            traceback.print_exc()
