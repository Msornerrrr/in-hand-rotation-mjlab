from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from mjlab.tasks.registry import load_runner_cls


def _prepare_agent_cfg(agent_cfg) -> dict[str, Any]:
  """Convert dataclass config to runner dict and drop incompatible MLP fields."""
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      if model_cfg.get("class_name", "MLPModel") != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
  return cfg_dict


class _PolicyLoadVecEnv:
  """Minimal VecEnv-like object used only to construct/load the policy network."""

  def __init__(
    self,
    env_cfg: Any,
    obs: dict[str, torch.Tensor],
    num_actions: int,
    step_dt: float,
    max_episode_length: int,
    device: str,
  ):
    self.cfg = env_cfg
    self._obs = obs
    self.num_envs = 1
    self.num_actions = num_actions
    self.step_dt = step_dt
    self.max_episode_length = max_episode_length
    self.device = torch.device(device)
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      (1,), device=self.device, dtype=torch.long
    )

  @property
  def unwrapped(self) -> "_PolicyLoadVecEnv":
    return self

  def get_observations(self) -> TensorDict:
    cloned = {k: v.clone() for k, v in self._obs.items()}
    return TensorDict(cloned, batch_size=[1])

  def reset(self):
    return self.get_observations(), {}

  def close(self) -> None:
    return None


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

