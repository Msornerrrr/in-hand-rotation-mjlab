from __future__ import annotations

import math
from typing import Any

import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from in_hand_rotation_mjlab.policy_server.actions import JointPositionActionMapper
from in_hand_rotation_mjlab.policy_server.config import PolicyServerConfig
from in_hand_rotation_mjlab.policy_server.contracts import (
  ActionTermMetadata,
  ObservationPacket,
  ServerActionPacket,
)
from in_hand_rotation_mjlab.policy_server.loader import load_inference_policy
from in_hand_rotation_mjlab.policy_server.observation import ActorObservationProcessor


class MjlabPolicyServer:
  """Server-side observation processing + policy inference + action mapping."""

  def __init__(
    self,
    task_id: str,
    cfg: PolicyServerConfig,
    *,
    action_term_metadata: dict[str, ActionTermMetadata],
    step_dt: float,
    max_episode_steps: int,
  ):
    self.task_id = task_id
    self.cfg = cfg
    self.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    self.env_cfg = load_env_cfg(task_id, play=cfg.play)
    self.agent_cfg = load_rl_cfg(task_id)

    self.action_mapper = JointPositionActionMapper(
      actions_cfg=self.env_cfg.actions,
      term_metadata=action_term_metadata,
      device=self.device,
    )

    self.policy, self._expected_actor_obs_dim = load_inference_policy(
      task_id=self.task_id,
      checkpoint_file=self.cfg.checkpoint_file,
      env_cfg=self.env_cfg,
      agent_cfg=self.agent_cfg,
      num_actions=self.action_mapper.total_action_dim,
      step_dt=step_dt,
      max_episode_length=max_episode_steps,
      device=self.device,
    )

    actor_obs_cfg = self.env_cfg.observations["actor"]
    # For obs terms whose name matches an action term, subtract that action
    # term's default_joint_pos so the client can send raw joint positions.
    term_offsets = {
      ts.term_name: ts.mapper.metadata.default_joint_pos
      for ts in self.action_mapper._term_mappers
      if ts.term_name in actor_obs_cfg.terms
    }
    self.obs_processor = ActorObservationProcessor(
      obs_group_cfg=actor_obs_cfg,
      device=self.device,
      term_offsets=term_offsets or None,
    )
    self.clip_actions = self.agent_cfg.clip_actions
    self._obs_dim_checked = False

  @property
  def expected_actor_obs_dim(self) -> int | None:
    return self._expected_actor_obs_dim

  @staticmethod
  def default_max_episode_steps(task_id: str, play: bool) -> tuple[float, int]:
    env_cfg = load_env_cfg(task_id, play=play)
    step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    if play and env_cfg.episode_length_s > 1e6:
      train_env_cfg = load_env_cfg(task_id, play=False)
      train_step_dt = train_env_cfg.sim.mujoco.timestep * train_env_cfg.decimation
      return train_step_dt, math.ceil(train_env_cfg.episode_length_s / train_step_dt)
    return step_dt, math.ceil(env_cfg.episode_length_s / step_dt)

  def reset(self, observation: ObservationPacket) -> None:
    self.obs_processor.reset()
    self.action_mapper.reset(observation.action_term_joint_pos)
    self._obs_dim_checked = False

  def infer(self, observation: ObservationPacket) -> ServerActionPacket:
    actor_obs = self.obs_processor.compute(observation.term_values)
    if not self._obs_dim_checked:
      expected = self.expected_actor_obs_dim
      if expected is not None and int(actor_obs.shape[-1]) != expected:
        raise ValueError(
          "Actor observation dimension mismatch between server processor and checkpoint. "
          f"Expected {expected}, got {int(actor_obs.shape[-1])}."
        )
      self._obs_dim_checked = True

    policy_obs = TensorDict({"actor": actor_obs}, batch_size=[1])
    action = self.policy(policy_obs)
    if self.clip_actions is not None:
      action = torch.clamp(action, -self.clip_actions, self.clip_actions)
    return self.action_mapper.map_policy_action(action)
