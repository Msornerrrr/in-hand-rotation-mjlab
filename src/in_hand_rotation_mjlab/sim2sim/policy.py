from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.tasks.registry import load_runner_cls
from in_hand_rotation_mjlab.sim2sim.native.config import _PolicyLoadVecEnv, _prepare_agent_cfg


def _infer_obs_dim(ckpt: dict[str, Any], key: str) -> int:
  state_dict = ckpt.get(key, {})
  if "obs_normalizer._mean" in state_dict:
    tensor = state_dict["obs_normalizer._mean"]
    return int(tensor.shape[-1])
  if "mlp.0.weight" in state_dict:
    tensor = state_dict["mlp.0.weight"]
    return int(tensor.shape[1])
  raise ValueError(f"Unable to infer observation dimension from checkpoint entry '{key}'.")


def load_inference_policy(
  *,
  task_id: str,
  checkpoint_file: str,
  env_cfg: Any,
  agent_cfg: Any,
  num_actions: int,
  step_dt: float,
  max_episode_length: int,
  device: str,
) -> tuple[Callable[[Any], torch.Tensor], int]:
  """Load RSL-RL inference policy and checkpoint actor-observation dimension."""

  checkpoint = Path(checkpoint_file)
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
  actor_obs_dim = _infer_obs_dim(ckpt, "actor_state_dict")
  critic_obs_dim = _infer_obs_dim(ckpt, "critic_state_dict")

  obs_groups = {
    "actor": torch.zeros((1, actor_obs_dim), device=device),
    "critic": torch.zeros((1, critic_obs_dim), device=device),
  }

  loader_env = _PolicyLoadVecEnv(
    env_cfg=env_cfg,
    obs=obs_groups,
    num_actions=num_actions,
    step_dt=step_dt,
    max_episode_length=max_episode_length,
    device=device,
  )
  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(loader_env, _prepare_agent_cfg(agent_cfg), device=device)
  try:
    runner.load(str(checkpoint), map_location=device)
  except RuntimeError as exc:
    raise RuntimeError(
      "Failed to load checkpoint into policy network. "
      "This usually means checkpoint/task config mismatch (e.g., different "
      "observation or action dimensions). "
      f"Checkpoint: {checkpoint}"
    ) from exc
  policy = runner.get_inference_policy(device=device)
  policy.eval()
  return policy, actor_obs_dim
