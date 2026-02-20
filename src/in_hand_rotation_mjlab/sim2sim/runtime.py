from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PolicyRolloutBackend(Protocol):
  """Backend contract for task-config driven policy rollouts.

  Sim2sim and sim2real can share the same rollout loop by implementing this
  interface and swapping only the backend-specific I/O.
  """

  task_id: str
  device: str
  decimation: int
  step_dt: float
  num_steps: int

  @property
  def expected_actor_obs_dim(self) -> int | None: ...

  def rollout_description(self) -> str: ...

  def reset_rollout(self) -> torch.Tensor: ...

  def infer_action(self, obs: torch.Tensor) -> torch.Tensor: ...

  def step_rollout(self, action: torch.Tensor) -> torch.Tensor: ...

  def maybe_capture_frame(self) -> None: ...

  def finish_rollout(self, success: bool) -> None: ...


def run_policy_rollout(backend: PolicyRolloutBackend) -> None:
  """Run a deterministic rollout loop using a backend implementation."""

  print(backend.rollout_description())
  success = False
  try:
    obs = backend.reset_rollout()

    expected_obs_dim = backend.expected_actor_obs_dim
    if expected_obs_dim is not None:
      actual_obs_dim = int(obs.shape[-1])
      if actual_obs_dim != expected_obs_dim:
        raise ValueError(
          "Actor observation dimension mismatch between backend adapter and "
          f"checkpoint. Expected {expected_obs_dim}, got {actual_obs_dim}."
        )

    with torch.no_grad():
      for _ in range(backend.num_steps):
        action = backend.infer_action(obs)
        obs = backend.step_rollout(action)
        backend.maybe_capture_frame()
    success = True
  finally:
    backend.finish_rollout(success=success)
