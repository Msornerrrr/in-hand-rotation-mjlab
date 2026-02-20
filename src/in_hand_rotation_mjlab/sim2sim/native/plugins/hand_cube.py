from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp

from in_hand_rotation_mjlab.sim2sim.native.plugins.base import NativeTaskPlugin
from in_hand_rotation_mjlab.sim2sim.native.plugins.registry import register_native_task_plugin

if TYPE_CHECKING:
  from in_hand_rotation_mjlab.sim2sim.native.observations import _ObservationAdapter
  from in_hand_rotation_mjlab.sim2sim.native.client import NativeMujocoClient


def _func_key(func: Any) -> str:
  return f"{func.__module__}:{func.__name__}"


class HandCubeNativeTaskPlugin(NativeTaskPlugin):
  """Task plugin for hand-cube reset + observation behavior."""

  def __init__(self, task_id: str):
    self.task_id = task_id
    self._grasp_cache: dict[str, torch.Tensor] | None = None
    self._grasp_cache_path: str | None = None
    self._grasp_cache_missing_warned = False
    self._event_handlers: dict[str, Any] = {
      _func_key(hand_cube_mdp.reset_from_grasp_cache): self._apply_reset_from_grasp_cache,
    }
    self._obs_handlers: dict[str, str] = {
      _func_key(hand_cube_mdp.cube_pose_in_palm_frame): "_obs_cube_pose_in_palm_frame",
      _func_key(hand_cube_mdp.cube_lin_vel_in_palm_frame): "_obs_cube_lin_vel_in_palm_frame",
      _func_key(hand_cube_mdp.cube_ang_vel_in_palm_frame): "_obs_cube_ang_vel_in_palm_frame",
      _func_key(hand_cube_mdp.cube_size): "_obs_cube_size",
      _func_key(hand_cube_mdp.cube_mass): "_obs_cube_mass",
      _func_key(hand_cube_mdp.cube_com_offset_b): "_obs_cube_com_offset_b",
      _func_key(hand_cube_mdp.cube_friction_coeff): "_obs_cube_friction_coeff",
      _func_key(hand_cube_mdp.joint_pos_commanded): "_obs_joint_pos_commanded",
      _func_key(hand_cube_mdp.joint_pos_command_error): "_obs_joint_pos_command_error",
    }

  def apply_event(
    self,
    runner: NativeMujocoClient,
    term_name: str,
    term_cfg: Any,
    params: dict[str, Any],
  ) -> bool:
    del term_name
    key = _func_key(term_cfg.func)
    handler = self._event_handlers.get(key)
    if handler is None:
      return False
    handler(runner, params)
    return True

  def build_observation_evaluator(
    self,
    adapter: _ObservationAdapter,
    func: Any,
    params: dict[str, Any],
  ) -> Any | None:
    method_name = self._obs_handlers.get(_func_key(func))
    if method_name is None:
      return None
    method = getattr(adapter, method_name)
    return lambda: method(**params)

  def _load_grasp_cache(
    self,
    runner: NativeMujocoClient,
    cache_file: str,
  ) -> dict[str, torch.Tensor] | None:
    cache_path = str(Path(cache_file).expanduser())
    if self._grasp_cache is not None and self._grasp_cache_path == cache_path:
      return self._grasp_cache

    if not Path(cache_path).exists():
      if not self._grasp_cache_missing_warned:
        print(
          f"[hand_cube] grasp cache missing at '{cache_path}', using default reset events."
        )
        self._grasp_cache_missing_warned = True
      self._grasp_cache = None
      self._grasp_cache_path = cache_path
      return None

    with np.load(cache_path, allow_pickle=False) as data:
      size_key = next(
        (
          k
          for k in ("cube_size", "cube_half_size", "cube_size_half", "size", "sizes")
          if k in data
        ),
        None,
      )
      if size_key is None:
        raise ValueError(
          f"Grasp cache '{cache_path}' missing cube size key. "
          "Expected cube_size/cube_half_size/size/sizes."
        )

      if "cube_pose_rel" in data:
        cube_pose_rel = data["cube_pose_rel"]
      elif "cube_pose" in data:
        cube_pose_rel = data["cube_pose"]
      elif "cube_pos_rel" in data and "cube_quat" in data:
        cube_pose_rel = np.concatenate([data["cube_pos_rel"], data["cube_quat"]], axis=-1)
      else:
        raise ValueError(
          f"Grasp cache '{cache_path}' missing cube pose key. "
          "Expected cube_pose_rel or cube_pose (Nx7)."
        )

      sizes = np.asarray(data[size_key], dtype=np.float32).reshape(-1)
      cube_pose_rel = np.asarray(cube_pose_rel, dtype=np.float32)

    if cube_pose_rel.ndim != 2 or cube_pose_rel.shape[1] != 7:
      raise ValueError(f"cube pose must be shape (N, 7), got {cube_pose_rel.shape}")
    if cube_pose_rel.shape[0] != sizes.shape[0]:
      raise ValueError(
        "Grasp cache arrays must have matching first dimension: "
        f"sizes={sizes.shape}, cube_pose={cube_pose_rel.shape}"
      )

    self._grasp_cache = {
      "cube_size": torch.from_numpy(sizes).to(device=runner.device),
      "cube_pose_rel": torch.from_numpy(cube_pose_rel).to(device=runner.device),
    }
    self._grasp_cache_path = cache_path
    print(f"[hand_cube] loaded grasp cache '{cache_path}' with {sizes.shape[0]} entries")
    return self._grasp_cache

  def _apply_reset_from_grasp_cache(
    self,
    runner: NativeMujocoClient,
    params: dict[str, Any],
  ) -> None:
    cache = self._load_grasp_cache(runner, params["cache_file"])
    if cache is None:
      return

    scale_list = params.get("scale_list")
    scale_jitter = float(params.get("scale_jitter", 0.025))
    pose_range = params.get("pose_range") or {}
    cube_cfg = params.get(
      "cube_cfg",
      SceneEntityCfg("cube", geom_names=("cube_geom",)),
    )
    assert isinstance(cube_cfg, SceneEntityCfg)
    cube_ent = runner.state.entity(cube_cfg.name)

    if cube_ent.root_qpos_adr is None or cube_ent.root_dof_adr is None:
      raise ValueError(f"Cube entity '{cube_cfg.name}' is not floating-base.")

    geom_local_ids = runner._select_local_ids(cube_cfg.geom_ids, len(cube_ent.geom_names))
    geom_ids = cube_ent.geom_model_ids[geom_local_ids]
    cache_sizes = cache["cube_size"]
    cache_pose_rel = cache["cube_pose_rel"]
    if cache_sizes.numel() == 0:
      raise ValueError("Grasp cache is empty.")

    if scale_list is not None and len(scale_list) > 0:
      scales = torch.tensor(scale_list, device=runner.device, dtype=cache_sizes.dtype)
      base_scale = scales[0]
      sampled_scale = sample_uniform(
        base_scale - scale_jitter,
        base_scale + scale_jitter,
        (1,),
        device=runner.device,
      )
      default_geom_size = runner._get_default_model_field("geom_size")[geom_ids, 0].mean()
      selected_sizes = sampled_scale * default_geom_size

      sorted_sizes, sorted_order = torch.sort(cache_sizes)
      insert_idx = torch.searchsorted(sorted_sizes, selected_sizes)
      left_idx = torch.clamp(insert_idx - 1, min=0)
      right_idx = torch.clamp(insert_idx, max=sorted_sizes.numel() - 1)
      left_diff = torch.abs(sorted_sizes[left_idx] - selected_sizes)
      right_diff = torch.abs(sorted_sizes[right_idx] - selected_sizes)
      nearest_sorted_idx = torch.where(left_diff <= right_diff, left_idx, right_idx)
      chosen_cache_id = int(sorted_order[nearest_sorted_idx][0].item())
      selected_size = float(selected_sizes[0].item())
    else:
      chosen_cache_id = int(
        torch.randint(cache_sizes.numel(), (1,), device=runner.device)[0].item()
      )
      selected_size = float(cache_sizes[chosen_cache_id].item())

    geom_ids_np = geom_ids.to(device="cpu", dtype=torch.long).numpy()
    runner.model.geom_size[geom_ids_np, :] = selected_size

    cube_pose_w = cache_pose_rel[chosen_cache_id : chosen_cache_id + 1].clone()
    cube_pose_w[:, 0:3] += runner._env_origin().unsqueeze(0)

    pos_delta = torch.stack(
      [
        sample_uniform(*pose_range.get("x", (0.0, 0.0)), 1, device=runner.device),
        sample_uniform(*pose_range.get("y", (0.0, 0.0)), 1, device=runner.device),
        sample_uniform(*pose_range.get("z", (0.0, 0.0)), 1, device=runner.device),
      ],
      dim=-1,
    )
    cube_pose_w[:, 0:3] += pos_delta

    roll_delta = sample_uniform(
      *pose_range.get("roll", (0.0, 0.0)),
      1,
      device=runner.device,
    )
    pitch_delta = sample_uniform(
      *pose_range.get("pitch", (0.0, 0.0)),
      1,
      device=runner.device,
    )
    yaw_delta = sample_uniform(
      *pose_range.get("yaw", (0.0, 0.0)),
      1,
      device=runner.device,
    )
    delta_quat = quat_from_euler_xyz(roll_delta, pitch_delta, yaw_delta)
    cube_pose_w[:, 3:7] = quat_mul(cube_pose_w[:, 3:7], delta_quat)

    runner.data.qpos[cube_ent.root_qpos_adr : cube_ent.root_qpos_adr + 7] = (
      cube_pose_w[0].cpu().numpy()
    )
    runner.data.qvel[cube_ent.root_dof_adr : cube_ent.root_dof_adr + 6] = 0.0

def _is_hand_cube_task(task_id: str) -> bool:
  return "HandCube" in task_id


register_native_task_plugin(_is_hand_cube_task, HandCubeNativeTaskPlugin)
