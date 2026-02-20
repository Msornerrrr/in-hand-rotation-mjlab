from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


_SCHEMA_VERSION = 1
_TERM_KEY_PREFIX = "joint_position_commands__"


@dataclass(frozen=True)
class ReplayTrajectory:
  step_dt: float
  raw_action: torch.Tensor
  joint_position_commands: dict[str, torch.Tensor]
  task_id: str | None = None
  checkpoint_file: str | None = None

  @property
  def num_steps(self) -> int:
    return int(self.raw_action.shape[0])


def _as_2d(x: np.ndarray, *, name: str) -> np.ndarray:
  if x.ndim == 1:
    return x[None, :]
  if x.ndim != 2:
    raise ValueError(f"Expected '{name}' to have 1D/2D shape, got {x.shape}.")
  return x


def save_replay_trajectory(path: str | Path, trajectory: ReplayTrajectory) -> Path:
  out_path = Path(path).expanduser().resolve()
  out_path.parent.mkdir(parents=True, exist_ok=True)

  raw_action = _as_2d(
    trajectory.raw_action.detach().to(device="cpu").contiguous().numpy(),
    name="raw_action",
  ).astype(np.float32, copy=False)

  if len(trajectory.joint_position_commands) == 0:
    raise ValueError("Replay trajectory must include at least one action term.")

  action_terms = tuple(trajectory.joint_position_commands.keys())
  num_steps = int(raw_action.shape[0])
  payload: dict[str, np.ndarray] = {
    "schema_version": np.asarray([_SCHEMA_VERSION], dtype=np.int32),
    "step_dt": np.asarray([float(trajectory.step_dt)], dtype=np.float32),
    "num_steps": np.asarray([num_steps], dtype=np.int64),
    "action_terms": np.asarray(action_terms, dtype=np.str_),
    "raw_action": raw_action,
    "task_id": np.asarray(trajectory.task_id or "", dtype=np.str_),
    "checkpoint_file": np.asarray(trajectory.checkpoint_file or "", dtype=np.str_),
  }

  for term_name, term_cmd in trajectory.joint_position_commands.items():
    cmd = _as_2d(
      term_cmd.detach().to(device="cpu").contiguous().numpy(),
      name=f"joint_position_commands[{term_name}]",
    ).astype(np.float32, copy=False)
    if int(cmd.shape[0]) != num_steps:
      raise ValueError(
        "All replay action-term arrays must have the same length as raw_action. "
        f"Term '{term_name}' has {int(cmd.shape[0])} steps, expected {num_steps}."
      )
    payload[f"{_TERM_KEY_PREFIX}{term_name}"] = cmd

  np.savez_compressed(str(out_path), **payload)
  return out_path


def load_replay_trajectory(path: str | Path, *, device: str = "cpu") -> ReplayTrajectory:
  in_path = Path(path).expanduser().resolve()
  if not in_path.exists():
    raise FileNotFoundError(f"Replay trajectory not found: {in_path}")

  with np.load(str(in_path), allow_pickle=False) as data:
    schema_version = int(np.asarray(data["schema_version"]).reshape(-1)[0])
    if schema_version != _SCHEMA_VERSION:
      raise ValueError(
        f"Unsupported replay trajectory schema version: {schema_version}. "
        f"Expected {_SCHEMA_VERSION}."
      )

    step_dt = float(np.asarray(data["step_dt"]).reshape(-1)[0])
    raw_action_np = _as_2d(np.asarray(data["raw_action"], dtype=np.float32), name="raw_action")
    num_steps = int(raw_action_np.shape[0])

    action_terms = tuple(str(v) for v in np.asarray(data["action_terms"]).tolist())
    if len(action_terms) == 0:
      raise ValueError("Replay trajectory has no action terms.")

    commands: dict[str, torch.Tensor] = {}
    for term_name in action_terms:
      key = f"{_TERM_KEY_PREFIX}{term_name}"
      if key not in data:
        raise KeyError(
          f"Replay trajectory missing key '{key}' for action term '{term_name}'."
        )
      cmd_np = _as_2d(
        np.asarray(data[key], dtype=np.float32),
        name=f"joint_position_commands[{term_name}]",
      )
      if int(cmd_np.shape[0]) != num_steps:
        raise ValueError(
          "Replay trajectory term length mismatch. "
          f"Term '{term_name}' has {int(cmd_np.shape[0])} steps, expected {num_steps}."
        )
      commands[term_name] = torch.from_numpy(cmd_np.copy()).to(device=device)

    raw_action = torch.from_numpy(raw_action_np.copy()).to(device=device)
    task_id = str(np.asarray(data["task_id"]).item()) if "task_id" in data else ""
    checkpoint_file = (
      str(np.asarray(data["checkpoint_file"]).item()) if "checkpoint_file" in data else ""
    )

  return ReplayTrajectory(
    step_dt=step_dt,
    raw_action=raw_action,
    joint_position_commands=commands,
    task_id=task_id or None,
    checkpoint_file=checkpoint_file or None,
  )
