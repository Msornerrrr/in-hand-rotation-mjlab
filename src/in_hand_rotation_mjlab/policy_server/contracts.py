from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionTermMetadata:
  term_name: str
  target_joint_names: tuple[str, ...]
  target_local_joint_ids: torch.Tensor
  default_joint_pos: torch.Tensor
  hard_joint_pos_limits: torch.Tensor
  soft_joint_pos_limits: torch.Tensor


@dataclass(frozen=True)
class ObservationPacket:
  # Raw per-term observations before clip/scale/delay/history transforms.
  term_values: dict[str, torch.Tensor]
  # Current measured joint positions for each action term.
  action_term_joint_pos: dict[str, torch.Tensor]


@dataclass(frozen=True)
class ServerActionPacket:
  # Raw policy output after actor inference (and optional clip_actions).
  raw_action: torch.Tensor
  # Per-action-term joint position targets in policy command frame.
  joint_position_commands: dict[str, torch.Tensor]

