from __future__ import annotations

import base64
from typing import Any

import numpy as np
import torch

from in_hand_rotation_mjlab.policy_server.contracts import (
  ActionTermMetadata,
  ObservationPacket,
  ServerActionPacket,
)


def _encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
  array = tensor.detach().to(device="cpu").contiguous().numpy()
  return {
    "dtype": str(array.dtype),
    "shape": list(array.shape),
    "data_b64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
  }


def _decode_tensor(payload: dict[str, Any], device: str = "cpu") -> torch.Tensor:
  dtype = np.dtype(str(payload["dtype"]))
  shape = tuple(int(v) for v in payload["shape"])
  raw = base64.b64decode(str(payload["data_b64"]))
  array = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
  return torch.from_numpy(array).to(device=device)


def encode_action_metadata_by_term(
  metadata_by_term: dict[str, ActionTermMetadata],
) -> dict[str, Any]:
  encoded: dict[str, Any] = {}
  for term_name, metadata in metadata_by_term.items():
    encoded[term_name] = {
      "term_name": metadata.term_name,
      "target_joint_names": list(metadata.target_joint_names),
      "target_local_joint_ids": _encode_tensor(metadata.target_local_joint_ids),
      "default_joint_pos": _encode_tensor(metadata.default_joint_pos),
      "hard_joint_pos_limits": _encode_tensor(metadata.hard_joint_pos_limits),
      "soft_joint_pos_limits": _encode_tensor(metadata.soft_joint_pos_limits),
    }
  return encoded


def encode_observation_packet(observation: ObservationPacket) -> dict[str, Any]:
  return {
    "term_values": {k: _encode_tensor(v) for k, v in observation.term_values.items()},
    "action_term_joint_pos": {
      k: _encode_tensor(v) for k, v in observation.action_term_joint_pos.items()
    },
  }


def decode_observation_packet(
  payload: dict[str, Any],
  *,
  device: str = "cpu",
) -> ObservationPacket:
  return ObservationPacket(
    term_values={
      k: _decode_tensor(v, device=device) for k, v in payload["term_values"].items()
    },
    action_term_joint_pos={
      k: _decode_tensor(v, device=device)
      for k, v in payload["action_term_joint_pos"].items()
    },
  )


def encode_server_action_packet(action: ServerActionPacket) -> dict[str, Any]:
  return {
    "raw_action": _encode_tensor(action.raw_action),
    "joint_position_commands": {
      k: _encode_tensor(v) for k, v in action.joint_position_commands.items()
    },
  }


def decode_server_action_packet(
  payload: dict[str, Any],
  *,
  device: str = "cpu",
) -> ServerActionPacket:
  return ServerActionPacket(
    raw_action=_decode_tensor(payload["raw_action"], device=device),
    joint_position_commands={
      k: _decode_tensor(v, device=device)
      for k, v in payload["joint_position_commands"].items()
    },
  )
