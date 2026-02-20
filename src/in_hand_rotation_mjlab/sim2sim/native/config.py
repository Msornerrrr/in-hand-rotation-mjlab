from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from tensordict import TensorDict


def _prepare_agent_cfg(agent_cfg) -> dict[str, Any]:
  """Convert dataclass config to runner dict and drop incompatible MLP fields."""
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      if model_cfg.get("class_name", "MLPModel") != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
  return cfg_dict


@dataclass(frozen=True)
class NativeSim2SimConfig:
  checkpoint_file: str
  device: str | None = None
  play: bool = True
  # If set, use an external policy server over ZMQ instead of in-process inference.
  server_endpoint: str | None = None
  server_timeout_ms: int = 30000
  shutdown_server_on_finish: bool = False
  num_steps: int | None = None
  # If None, defaults to:
  #   {checkpoint_run_dir}/videos/sim2sim/sim2sim_{checkpoint_stem}.mp4
  video_file: str | None = None
  video_height: int = 720
  video_width: int = 1280
  video_fps: int | None = None
  camera: int | str | None = None


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
