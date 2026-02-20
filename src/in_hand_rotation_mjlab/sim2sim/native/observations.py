from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.buffers import CircularBuffer, DelayBuffer
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from in_hand_rotation_mjlab.sim2sim.native.actions import _ActionAdapter
from in_hand_rotation_mjlab.sim2sim.native.state import _NativeState

if TYPE_CHECKING:
  from in_hand_rotation_mjlab.sim2sim.native.plugins import NativeTaskPlugin


@dataclass
class _ObsTermState:
  name: str
  cfg: ObservationTermCfg
  evaluator: Any
  scale: torch.Tensor | None
  delay_buffer: DelayBuffer | None
  history_buffer: CircularBuffer | None


class _ObservationAdapter:
  def __init__(
    self,
    obs_group_cfg: ObservationGroupCfg,
    state: _NativeState,
    action_adapter: _ActionAdapter,
    device: str,
    plugins: list[NativeTaskPlugin] | None = None,
  ):
    self.cfg = copy.deepcopy(obs_group_cfg)
    self.state = state
    self.action_adapter = action_adapter
    self.device = device
    self._plugins = plugins or []
    self._cache: dict[str, Any] = {}
    self._terms: list[_ObsTermState] = []

    if not self.cfg.concatenate_terms:
      raise NotImplementedError(
        "Native sim2sim currently supports concatenate_terms=True only."
      )
    if self.cfg.concatenate_dim not in (-1, 1):
      raise NotImplementedError(
        "Native sim2sim currently supports concatenate_dim=-1 only."
      )

    for term_name, term_cfg in self.cfg.terms.items():
      term_cfg = copy.deepcopy(term_cfg)
      for value in term_cfg.params.values():
        if isinstance(value, SceneEntityCfg):
          value.resolve(self.state.scene)
      if self.cfg.history_length is not None:
        term_cfg.history_length = self.cfg.history_length
        term_cfg.flatten_history_dim = self.cfg.flatten_history_dim
      if self.cfg.enable_corruption and term_cfg.noise is not None:
        raise ValueError(
          "Observation noise corruption is enabled. "
          "Use play=True for deployment/sim2sim configs."
        )

      evaluator = self._build_evaluator(term_cfg.func, term_cfg.params)
      scale = None
      if term_cfg.scale is not None:
        scale = torch.tensor(term_cfg.scale, device=self.device, dtype=torch.float32)

      delay_buffer = None
      if term_cfg.delay_max_lag > 0:
        delay_buffer = DelayBuffer(
          min_lag=term_cfg.delay_min_lag,
          max_lag=term_cfg.delay_max_lag,
          batch_size=1,
          device=self.device,
          per_env=term_cfg.delay_per_env,
          hold_prob=term_cfg.delay_hold_prob,
          update_period=term_cfg.delay_update_period,
          per_env_phase=term_cfg.delay_per_env_phase,
        )

      history_buffer = None
      if term_cfg.history_length > 0:
        history_buffer = CircularBuffer(
          max_len=term_cfg.history_length,
          batch_size=1,
          device=self.device,
        )

      self._terms.append(
        _ObsTermState(
          name=term_name,
          cfg=term_cfg,
          evaluator=evaluator,
          scale=scale,
          delay_buffer=delay_buffer,
          history_buffer=history_buffer,
        )
      )

  @staticmethod
  def _func_key(func: Any) -> str:
    return f"{func.__module__}:{func.__name__}"

  def _build_evaluator(self, func: Any, params: dict[str, Any]):
    key = self._func_key(func)

    if key == self._func_key(envs_mdp.joint_pos_rel):
      return lambda: self._obs_joint_pos_rel(**params)
    if key == self._func_key(envs_mdp.joint_vel_rel):
      return lambda: self._obs_joint_vel_rel(**params)
    if key == self._func_key(envs_mdp.last_action):
      return lambda: self._obs_last_action(**params)
    if key == self._func_key(envs_mdp.base_lin_vel):
      return lambda: self._obs_base_lin_vel(**params)
    if key == self._func_key(envs_mdp.base_ang_vel):
      return lambda: self._obs_base_ang_vel(**params)
    if key == self._func_key(envs_mdp.projected_gravity):
      return lambda: self._obs_projected_gravity(**params)

    for plugin in self._plugins:
      evaluator = plugin.build_observation_evaluator(self, func, params)
      if evaluator is not None:
        return evaluator

    raise NotImplementedError(
      "Unsupported observation function for native sim2sim: "
      f"{self._func_key(func)}"
    )

  def reset(self) -> None:
    for term in self._terms:
      if term.delay_buffer is not None:
        term.delay_buffer.reset()
      if term.history_buffer is not None:
        term.history_buffer.reset()

  def compute_raw_terms(self) -> dict[str, torch.Tensor]:
    """Compute raw per-term observations before post-processing transforms."""
    out: dict[str, torch.Tensor] = {}
    for term in self._terms:
      out[term.name] = cast(torch.Tensor, term.evaluator()).clone()
    return out

  def compute(self) -> torch.Tensor:
    out_terms: list[torch.Tensor] = []
    for term in self._terms:
      obs = cast(torch.Tensor, term.evaluator()).clone()
      if term.cfg.clip is not None:
        obs = obs.clip(min=term.cfg.clip[0], max=term.cfg.clip[1])
      if term.scale is not None:
        obs = obs * term.scale
      if term.delay_buffer is not None:
        term.delay_buffer.append(obs)
        obs = term.delay_buffer.compute()
      if term.history_buffer is not None:
        term.history_buffer.append(obs)
        if term.cfg.flatten_history_dim:
          obs = term.history_buffer.buffer.reshape(1, -1)
        else:
          obs = term.history_buffer.buffer
      out_terms.append(obs)
    return torch.cat(out_terms, dim=-1)

  def _select_joint_ids(
    self,
    asset_cfg: SceneEntityCfg,
  ) -> list[int] | slice:
    return asset_cfg.joint_ids

  def _single_body_local_id(self, asset_cfg: SceneEntityCfg) -> int:
    if isinstance(asset_cfg.body_ids, slice):
      body_ids = list(range(*asset_cfg.body_ids.indices(len(self.state.entity(asset_cfg.name).body_names))))
    else:
      body_ids = list(asset_cfg.body_ids)
    if len(body_ids) != 1:
      raise ValueError(
        f"Expected exactly one body in SceneEntityCfg('{asset_cfg.name}'), got {body_ids}"
      )
    return body_ids[0]

  def _obs_joint_pos_rel(
    self,
    asset_cfg: SceneEntityCfg,
    biased: bool = False,
  ) -> torch.Tensor:
    ent = self.state.entity(asset_cfg.name)
    ids = self._select_joint_ids(asset_cfg)
    joint_pos = self.state.joint_pos(asset_cfg.name)
    if biased:
      joint_pos = joint_pos + ent.encoder_bias
    return joint_pos[:, ids] - ent.default_joint_pos[:, ids]

  def _obs_joint_vel_rel(
    self,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    ent = self.state.entity(asset_cfg.name)
    ids = self._select_joint_ids(asset_cfg)
    return self.state.joint_vel(asset_cfg.name)[:, ids] - ent.default_joint_vel[:, ids]

  def _obs_last_action(self, action_name: str | None = None) -> torch.Tensor:
    if action_name is None:
      return self.action_adapter.action
    return self.action_adapter.term_raw_action(action_name)

  def _commanded_joint_pos(
    self,
    action_name: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    ent = self.state.entity(asset_cfg.name)
    joint_ids = self._select_joint_ids(asset_cfg)
    ids, cmd = self.action_adapter.term_commanded_joint_pos(action_name)
    if isinstance(joint_ids, slice):
      selected_joint_ids = list(range(*joint_ids.indices(len(ent.joint_names))))
    else:
      selected_joint_ids = list(joint_ids)
    commanded_full = self.state.joint_pos(asset_cfg.name).clone()
    commanded_full[:, ids] = cmd
    return commanded_full[:, selected_joint_ids]

  def _obs_joint_pos_commanded(
    self,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  ) -> torch.Tensor:
    return self._commanded_joint_pos(action_name=action_name, asset_cfg=asset_cfg)

  def _obs_joint_pos_command_error(
    self,
    action_name: str = "joint_pos",
    biased: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  ) -> torch.Tensor:
    ent = self.state.entity(asset_cfg.name)
    ids = self._select_joint_ids(asset_cfg)
    measured = self.state.joint_pos(asset_cfg.name)[:, ids]
    if biased:
      measured = measured + ent.encoder_bias[:, ids]
    commanded = self._commanded_joint_pos(action_name=action_name, asset_cfg=asset_cfg)
    return commanded - measured

  def _obs_base_lin_vel(self, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat_w = self.state.root_link_pose_w(asset_cfg.name)[:, 3:7]
    lin_vel_w = self.state.root_link_vel_w(asset_cfg.name)[:, 0:3]
    return quat_apply_inverse(quat_w, lin_vel_w)

  def _obs_base_ang_vel(self, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat_w = self.state.root_link_pose_w(asset_cfg.name)[:, 3:7]
    ang_vel_w = self.state.root_link_vel_w(asset_cfg.name)[:, 3:6]
    return quat_apply_inverse(quat_w, ang_vel_w)

  def _obs_projected_gravity(self, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat_w = self.state.root_link_pose_w(asset_cfg.name)[:, 3:7]
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
    return quat_apply_inverse(quat_w, gravity)

  def _palm_center_pose_w(
    self,
    hand_cfg: SceneEntityCfg,
    palm_center_geom_expr: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    palm_body_local_id = self._single_body_local_id(hand_cfg)
    palm_pose = self.state.body_link_pose_w(hand_cfg.name, [palm_body_local_id]).squeeze(1)
    palm_pos_w = palm_pose[:, 0:3]
    palm_quat_w = palm_pose[:, 3:7]

    cache_key = f"palm_center_geom_ids::{hand_cfg.name}::{palm_center_geom_expr}"
    if cache_key not in self._cache:
      hand_entity = self.state.scene.entities[hand_cfg.name]
      local_geom_ids, _ = hand_entity.find_geoms(
        palm_center_geom_expr, preserve_order=True
      )
      self._cache[cache_key] = local_geom_ids
    local_geom_ids = cast(list[int], self._cache[cache_key])
    if len(local_geom_ids) == 0:
      return palm_pos_w, palm_quat_w

    palm_geom_pos = self.state.geom_pos_w(hand_cfg.name, local_geom_ids)
    palm_center_w = palm_geom_pos.mean(dim=1)
    return palm_center_w, palm_quat_w

  def _obs_cube_pose_in_palm_frame(
    self,
    object_name: str,
    hand_cfg: SceneEntityCfg,
    palm_center_geom_expr: str,
  ) -> torch.Tensor:
    palm_center_w, palm_quat_w = self._palm_center_pose_w(
      hand_cfg=hand_cfg,
      palm_center_geom_expr=palm_center_geom_expr,
    )
    cube_pose = self.state.root_link_pose_w(object_name)
    cube_pos_w = cube_pose[:, 0:3]
    cube_quat_w = cube_pose[:, 3:7]
    q_inv = quat_inv(palm_quat_w)
    pos_palm = quat_apply(q_inv, cube_pos_w - palm_center_w)
    quat_palm = quat_mul(q_inv, cube_quat_w)
    return torch.cat([pos_palm, quat_palm], dim=-1)

  def _obs_cube_lin_vel_in_palm_frame(
    self,
    object_name: str,
    hand_cfg: SceneEntityCfg,
    palm_center_geom_expr: str,
  ) -> torch.Tensor:
    palm_body_local_id = self._single_body_local_id(hand_cfg)
    palm_pose = self.state.body_link_pose_w(hand_cfg.name, [palm_body_local_id]).squeeze(1)
    palm_vel = self.state.body_link_vel_w(hand_cfg.name, [palm_body_local_id]).squeeze(1)
    palm_pos_w = palm_pose[:, 0:3]
    palm_quat_w = palm_pose[:, 3:7]
    palm_lin_vel_w = palm_vel[:, 0:3]
    palm_ang_vel_w = palm_vel[:, 3:6]

    palm_center_w, _ = self._palm_center_pose_w(
      hand_cfg=hand_cfg,
      palm_center_geom_expr=palm_center_geom_expr,
    )
    center_offset_w = palm_center_w - palm_pos_w
    palm_center_lin_vel_w = palm_lin_vel_w + torch.cross(
      palm_ang_vel_w, center_offset_w, dim=-1
    )
    cube_lin_vel_w = self.state.root_link_vel_w(object_name)[:, 0:3]
    rel_lin_vel_w = cube_lin_vel_w - palm_center_lin_vel_w
    return quat_apply(quat_inv(palm_quat_w), rel_lin_vel_w)

  def _obs_cube_ang_vel_in_palm_frame(
    self,
    object_name: str,
    hand_cfg: SceneEntityCfg,
    palm_center_geom_expr: str,
  ) -> torch.Tensor:
    del palm_center_geom_expr
    palm_body_local_id = self._single_body_local_id(hand_cfg)
    palm_pose = self.state.body_link_pose_w(hand_cfg.name, [palm_body_local_id]).squeeze(1)
    palm_vel = self.state.body_link_vel_w(hand_cfg.name, [palm_body_local_id]).squeeze(1)
    palm_quat_w = palm_pose[:, 3:7]
    palm_ang_vel_w = palm_vel[:, 3:6]
    cube_ang_vel_w = self.state.root_link_vel_w(object_name)[:, 3:6]
    rel_ang_vel_w = cube_ang_vel_w - palm_ang_vel_w
    return quat_apply(quat_inv(palm_quat_w), rel_ang_vel_w)

  def _obs_cube_size(
    self,
    object_name: str,
    geom_name: str = "cube_geom",
  ) -> torch.Tensor:
    cache_key = f"cube_size_geom::{object_name}::{geom_name}"
    if cache_key not in self._cache:
      entity = self.state.scene.entities[object_name]
      local_geom_ids, _ = entity.find_geoms(geom_name, preserve_order=True)
      if len(local_geom_ids) != 1:
        raise ValueError(
          f"Expected one geom for '{geom_name}', got {len(local_geom_ids)}."
        )
      self._cache[cache_key] = local_geom_ids[0]
    local_geom_id = cast(int, self._cache[cache_key])
    return self.state.geom_scalar_field(
      entity_name=object_name,
      geom_local_id=local_geom_id,
      field="geom_size",
      axis=0,
    )

  def _obs_cube_mass(
    self,
    object_name: str,
    body_name: str = "cube",
  ) -> torch.Tensor:
    cache_key = f"cube_mass_body::{object_name}::{body_name}"
    if cache_key not in self._cache:
      entity = self.state.scene.entities[object_name]
      local_body_ids, _ = entity.find_bodies(body_name, preserve_order=True)
      if len(local_body_ids) != 1:
        raise ValueError(
          f"Expected one body for '{body_name}', got {len(local_body_ids)}."
        )
      self._cache[cache_key] = local_body_ids[0]
    local_body_id = cast(int, self._cache[cache_key])
    return self.state.body_field(object_name, local_body_id, "body_mass")

  def _obs_cube_com_offset_b(
    self,
    object_name: str,
    body_name: str = "cube",
  ) -> torch.Tensor:
    cache_key = f"cube_com_body::{object_name}::{body_name}"
    if cache_key not in self._cache:
      entity = self.state.scene.entities[object_name]
      local_body_ids, _ = entity.find_bodies(body_name, preserve_order=True)
      if len(local_body_ids) != 1:
        raise ValueError(
          f"Expected one body for '{body_name}', got {len(local_body_ids)}."
        )
      self._cache[cache_key] = local_body_ids[0]
    local_body_id = cast(int, self._cache[cache_key])
    return self.state.body_field(object_name, local_body_id, "body_ipos")

  def _obs_cube_friction_coeff(
    self,
    object_name: str,
    geom_name: str = "cube_geom",
    axis: int = 0,
  ) -> torch.Tensor:
    cache_key = f"cube_friction_geom::{object_name}::{geom_name}"
    if cache_key not in self._cache:
      entity = self.state.scene.entities[object_name]
      local_geom_ids, _ = entity.find_geoms(geom_name, preserve_order=True)
      if len(local_geom_ids) != 1:
        raise ValueError(
          f"Expected one geom for '{geom_name}', got {len(local_geom_ids)}."
        )
      self._cache[cache_key] = local_geom_ids[0]
    local_geom_id = cast(int, self._cache[cache_key])
    return self.state.geom_scalar_field(
      entity_name=object_name,
      geom_local_id=local_geom_id,
      field="geom_friction",
      axis=axis,
    )
