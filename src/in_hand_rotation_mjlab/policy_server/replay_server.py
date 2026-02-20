from __future__ import annotations

from in_hand_rotation_mjlab.policy_server.contracts import (
  ActionTermMetadata,
  ObservationPacket,
  ServerActionPacket,
)
from in_hand_rotation_mjlab.policy_server.replay import ReplayTrajectory


class ReplayPolicyServer:
  """Serves pre-recorded actions with the same infer/reset interface."""

  def __init__(
    self,
    *,
    action_term_metadata: dict[str, ActionTermMetadata],
    trajectory: ReplayTrajectory,
    loop_trajectory: bool = True,
  ):
    self.action_term_metadata = action_term_metadata
    self.trajectory = trajectory
    self.loop_trajectory = loop_trajectory
    self._step_idx = 0

    if trajectory.num_steps <= 0:
      raise ValueError("Replay trajectory is empty.")

    self._validate_against_metadata()

  def _validate_against_metadata(self) -> None:
    term_names = tuple(self.trajectory.joint_position_commands.keys())
    if set(term_names) != set(self.action_term_metadata.keys()):
      raise ValueError(
        "Replay trajectory action terms do not match server action metadata. "
        f"Trajectory terms: {sorted(term_names)}, "
        f"Metadata terms: {sorted(self.action_term_metadata.keys())}."
      )
    for term_name, cmd in self.trajectory.joint_position_commands.items():
      meta = self.action_term_metadata[term_name]
      expected_dim = len(meta.target_joint_names)
      actual_dim = int(cmd.shape[1])
      if actual_dim != expected_dim:
        raise ValueError(
          "Replay trajectory dimension mismatch for action term "
          f"'{term_name}': expected {expected_dim}, got {actual_dim}."
        )

  def reset(self, observation: ObservationPacket) -> None:
    del observation
    self._step_idx = 0

  def infer(self, observation: ObservationPacket) -> ServerActionPacket:
    del observation
    idx = self._step_idx
    if idx >= self.trajectory.num_steps:
      if self.loop_trajectory:
        idx = idx % self.trajectory.num_steps
      else:
        idx = self.trajectory.num_steps - 1

    raw_action = self.trajectory.raw_action[idx : idx + 1]
    commands = {
      name: cmd[idx : idx + 1]
      for name, cmd in self.trajectory.joint_position_commands.items()
    }
    self._step_idx += 1
    return ServerActionPacket(
      raw_action=raw_action,
      joint_position_commands=commands,
    )
