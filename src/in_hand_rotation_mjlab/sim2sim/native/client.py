from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, cast

import mediapy as media
import mujoco
import numpy as np
import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from in_hand_rotation_mjlab.policy_server.contracts import (
  ActionTermMetadata,
  ObservationPacket,
  ServerActionPacket,
)
from in_hand_rotation_mjlab.sim2sim.native.actions import _ActionAdapter
from in_hand_rotation_mjlab.sim2sim.native.config import NativeSim2SimConfig
from in_hand_rotation_mjlab.sim2sim.native.observations import _ObservationAdapter
from in_hand_rotation_mjlab.sim2sim.native.plugins import resolve_native_task_plugins
from in_hand_rotation_mjlab.sim2sim.native.state import _NativeState


class NativeMujocoClient:
  """Minimal native MuJoCo client for server-driven sim2sim rollouts.

  This client intentionally excludes domain-randomization logic. It only keeps
  reset/event behavior needed for deterministic sim2sim evaluation.
  """

  def __init__(self, task_id: str, cfg: NativeSim2SimConfig):
    self.task_id = task_id
    self.cfg = cfg
    self.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    self.env_cfg = load_env_cfg(task_id, play=cfg.play)
    self.agent_cfg = load_rl_cfg(task_id)
    self.env_cfg.scene.num_envs = 1

    self.scene = Scene(self.env_cfg.scene, device="cpu")
    self.model = self.scene.compile()
    self.env_cfg.sim.mujoco.apply(self.model)
    self.data = mujoco.MjData(self.model)
    self._init_key_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_KEY, "init_state"
    )
    self._default_model_fields: dict[str, torch.Tensor] = {}
    self._startup_events_applied = False

    self.state = _NativeState(
      scene=self.scene,
      model=self.model,
      data=self.data,
      device=self.device,
    )
    self._reset_to_keyframe()
    self.action_adapter = _ActionAdapter(self.env_cfg.actions, self.state)
    self.decimation = self.env_cfg.decimation
    self.step_dt = self.env_cfg.sim.mujoco.timestep * self.decimation
    self.max_episode_steps = math.ceil(self.env_cfg.episode_length_s / self.step_dt)
    self.num_steps = self._resolve_default_num_steps(cfg.num_steps)
    self.video_path = self._resolve_default_video_path(cfg.video_file)

    self._task_plugins = resolve_native_task_plugins(self.task_id)
    actor_obs_cfg = self.env_cfg.observations["actor"]
    self.obs_adapter = _ObservationAdapter(
      obs_group_cfg=actor_obs_cfg,
      state=self.state,
      action_adapter=self.action_adapter,
      device=self.device,
      plugins=self._task_plugins,
    )

    self._video_renderer: mujoco.Renderer | None = None
    self._render_camera: int | str | mujoco.MjvCamera | None = None
    self._video_frames: list[Any] = []

  def _resolve_default_num_steps(self, requested_num_steps: int | None) -> int:
    if requested_num_steps is not None:
      return requested_num_steps

    # Play configs often set episode_length_s to a huge value; in that case, use
    # the nominal training horizon by default.
    if self.cfg.play and self.env_cfg.episode_length_s > 1e6:
      train_env_cfg = load_env_cfg(self.task_id, play=False)
      train_step_dt = train_env_cfg.sim.mujoco.timestep * train_env_cfg.decimation
      return math.ceil(train_env_cfg.episode_length_s / train_step_dt)
    return self.max_episode_steps

  def _resolve_default_video_path(self, video_file: str | None) -> Path:
    if video_file is not None:
      return Path(video_file)
    ckpt_path = Path(self.cfg.checkpoint_file).resolve()
    run_dir = ckpt_path.parent
    return run_dir / "videos" / "sim2sim" / f"sim2sim_{ckpt_path.stem}.mp4"

  def _resolve_render_camera(self) -> int | str | mujoco.MjvCamera | None:
    if self.cfg.camera is not None:
      return self.cfg.camera

    viewer = self.env_cfg.viewer
    cam = mujoco.MjvCamera()
    cam.distance = float(viewer.distance)
    cam.azimuth = float(viewer.azimuth)
    cam.elevation = float(viewer.elevation)
    cam.lookat[:] = viewer.lookat

    # Try to follow the task's configured viewer origin by default.
    if viewer.origin_type.name == "ASSET_BODY":
      if viewer.entity_name is not None and viewer.body_name is not None:
        body_name = f"{viewer.entity_name}/{viewer.body_name}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
          cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
          cam.trackbodyid = body_id
          return cam
    if viewer.origin_type.name == "ASSET_ROOT":
      if viewer.entity_name is not None and viewer.entity_name in self.scene.entities:
        ent = self.scene.entities[viewer.entity_name]
        root_body_local = ent.root_body.name.split("/")[-1]
        body_name = f"{viewer.entity_name}/{root_body_local}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
          cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
          cam.trackbodyid = body_id
          return cam

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    return cam

  @staticmethod
  def _func_key(func: Any) -> str:
    return f"{func.__module__}:{func.__name__}"

  @staticmethod
  def _should_skip_event_for_sim2sim(term_name: str, term_cfg: Any) -> bool:
    # Sim2sim intentionally skips all domain-randomization style events.
    if term_name.startswith("dr_"):
      return True
    if bool(getattr(term_cfg, "domain_randomization", False)):
      return True
    func_name = getattr(term_cfg.func, "__name__", "")
    if func_name.startswith("randomize_"):
      return True
    if func_name in {
      "set_actuator_effort_limits",
      "inject_random_cube_pose",
      "apply_random_cube_wrench",
    }:
      return True
    return False

  @staticmethod
  def _select_local_ids(ids: list[int] | slice, count: int) -> list[int]:
    if isinstance(ids, slice):
      return list(range(*ids.indices(count)))
    return list(ids)

  def _resolve_scene_cfg_values(self, value: Any) -> Any:
    if isinstance(value, SceneEntityCfg):
      cfg = copy.deepcopy(value)
      cfg.resolve(self.scene)
      return cfg
    if isinstance(value, dict):
      return {k: self._resolve_scene_cfg_values(v) for k, v in value.items()}
    if isinstance(value, list):
      return [self._resolve_scene_cfg_values(v) for v in value]
    if isinstance(value, tuple):
      return tuple(self._resolve_scene_cfg_values(v) for v in value)
    return value

  def _iter_event_terms(self, mode: str):
    for term_name, term_cfg in self.env_cfg.events.items():
      if term_cfg is None or term_cfg.mode != mode:
        continue
      if self._should_skip_event_for_sim2sim(term_name, term_cfg):
        continue
      raw_params = copy.deepcopy(term_cfg.params) if term_cfg.params is not None else {}
      params = cast(dict[str, Any], self._resolve_scene_cfg_values(raw_params))
      yield term_name, term_cfg, params

  def _get_default_model_field(self, field: str) -> torch.Tensor:
    if field not in self._default_model_fields:
      if not hasattr(self.model, field):
        raise ValueError(f"Field '{field}' not found in MuJoCo model.")
      raw = np.array(getattr(self.model, field), copy=True)
      self._default_model_fields[field] = torch.as_tensor(raw, device=self.device)
    return self._default_model_fields[field]

  def _env_origin(self) -> torch.Tensor:
    try:
      origins = self.scene.env_origins
      return origins[0].to(device=self.device, dtype=torch.float32)
    except Exception:
      return torch.zeros((3,), device=self.device, dtype=torch.float32)

  def _apply_reset_root_state_uniform(self, params: dict[str, Any]) -> None:
    pose_range = params["pose_range"]
    velocity_range = params.get("velocity_range") or {}
    asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
    assert isinstance(asset_cfg, SceneEntityCfg)
    ent = self.state.entity(asset_cfg.name)

    ranges = torch.tensor(
      [pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
      device=self.device,
      dtype=torch.float32,
    )
    pose_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (1, 6), device=self.device
    )

    root_state = ent.default_root_state.clone()
    positions = root_state[:, 0:3] + pose_samples[:, 0:3] + self._env_origin().unsqueeze(0)
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_state[:, 3:7], orientations_delta)

    if ent.is_fixed_base:
      if not ent.is_mocap or ent.mocap_id is None:
        raise ValueError(
          f"Cannot reset fixed-base non-mocap entity '{asset_cfg.name}' in native sim2sim."
        )
      self.data.mocap_pos[ent.mocap_id] = positions[0].cpu().numpy()
      self.data.mocap_quat[ent.mocap_id] = orientations[0].cpu().numpy()
      return

    if ent.root_qpos_adr is None or ent.root_dof_adr is None:
      raise ValueError(f"Entity '{asset_cfg.name}' has no floating-base root joint.")
    self.data.qpos[ent.root_qpos_adr : ent.root_qpos_adr + 7] = (
      torch.cat([positions, orientations], dim=-1)[0].cpu().numpy()
    )

    vel_ranges = torch.tensor(
      [
        velocity_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
      dtype=torch.float32,
    )
    vel_samples = sample_uniform(
      vel_ranges[:, 0], vel_ranges[:, 1], (1, 6), device=self.device
    )
    velocities = root_state[:, 7:13] + vel_samples
    ang_vel_b = quat_apply_inverse(orientations, velocities[:, 3:6])
    velocity_qvel = torch.cat([velocities[:, 0:3], ang_vel_b], dim=-1)
    self.data.qvel[ent.root_dof_adr : ent.root_dof_adr + 6] = (
      velocity_qvel[0].cpu().numpy()
    )

  def _apply_reset_joints_by_offset(self, params: dict[str, Any]) -> None:
    position_range = params["position_range"]
    velocity_range = params["velocity_range"]
    asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
    assert isinstance(asset_cfg, SceneEntityCfg)
    ent = self.state.entity(asset_cfg.name)

    joint_ids = self._select_local_ids(asset_cfg.joint_ids, len(ent.joint_names))
    if len(joint_ids) == 0:
      return

    default_joint_pos = ent.default_joint_pos[:, joint_ids].clone()
    default_joint_vel = ent.default_joint_vel[:, joint_ids].clone()
    soft_limits = ent.soft_joint_pos_limits[joint_ids]

    joint_pos = default_joint_pos + sample_uniform(
      position_range[0], position_range[1], default_joint_pos.shape, device=self.device
    )
    joint_pos = joint_pos.clamp(soft_limits[:, 0], soft_limits[:, 1])
    joint_vel = default_joint_vel + sample_uniform(
      velocity_range[0], velocity_range[1], default_joint_vel.shape, device=self.device
    )

    qpos_adr = ent.joint_qpos_adr[joint_ids].to(device="cpu", dtype=torch.long).numpy()
    qvel_adr = ent.joint_dof_adr[joint_ids].to(device="cpu", dtype=torch.long).numpy()
    self.data.qpos[qpos_adr] = joint_pos[0].cpu().numpy()
    self.data.qvel[qvel_adr] = joint_vel[0].cpu().numpy()

  def _apply_event_term(self, term_name: str, term_cfg: Any, params: dict[str, Any]) -> None:
    key = self._func_key(term_cfg.func)
    if key == self._func_key(envs_mdp.reset_scene_to_default):
      # Native reset already restores the init_state keyframe first.
      return
    if key == self._func_key(envs_mdp.reset_root_state_uniform):
      self._apply_reset_root_state_uniform(params)
      return
    if key == self._func_key(envs_mdp.reset_joints_by_offset):
      self._apply_reset_joints_by_offset(params)
      return
    for plugin in self._task_plugins:
      if plugin.apply_event(self, term_name, term_cfg, params):
        return
    raise NotImplementedError(
      "Unsupported event term in native sim2sim: "
      f"{term_name} ({key})."
    )

  def _apply_event_mode(self, mode: str) -> None:
    for term_name, term_cfg, params in self._iter_event_terms(mode):
      self._apply_event_term(term_name, term_cfg, params)

  def rollout_description(self) -> str:
    return (
      "[INFO] Native sim2sim client: "
      f"task={self.task_id}, steps={self.num_steps}, decimation={self.decimation}, "
      f"step_dt={self.step_dt:.6f}, device={self.device}\n"
      f"[INFO] Sim2sim video path: {self.video_path}"
    )

  def build_server_action_metadata(self) -> dict[str, ActionTermMetadata]:
    return self.action_adapter.build_server_action_metadata()

  def _reset_to_keyframe(self) -> None:
    if self._init_key_id >= 0:
      mujoco.mj_resetDataKeyframe(self.model, self.data, self._init_key_id)
    else:
      mujoco.mj_resetData(self.model, self.data)

  def reset_native_state(self) -> None:
    self._reset_to_keyframe()
    if not self._startup_events_applied:
      self._apply_event_mode("startup")
      self._startup_events_applied = True
    self._apply_event_mode("reset")
    mujoco.mj_forward(self.model, self.data)

  def _make_renderer(self) -> mujoco.Renderer | None:
    if self.video_path is None:
      return None
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    mujoco_gl = os.environ.get("MUJOCO_GL")
    if not has_display and mujoco_gl not in {"egl", "osmesa"}:
      raise RuntimeError(
        "Video requested in headless mode, but MUJOCO_GL is not set to a headless "
        "backend. Export MUJOCO_GL=egl (or osmesa) before launching this script."
      )
    try:
      return mujoco.Renderer(
        self.model,
        height=self.cfg.video_height,
        width=self.cfg.video_width,
      )
    except Exception as exc:
      raise RuntimeError(
        "Failed to create MuJoCo renderer for video capture. "
        "Set MUJOCO_GL=egl (or osmesa) and ensure the backend is available."
      ) from exc

  def reset_rollout(self) -> ObservationPacket:
    self.reset_native_state()
    self.action_adapter.reset(self.state)
    self.obs_adapter.reset()
    self._video_renderer = self._make_renderer()
    self._render_camera = (
      self._resolve_render_camera() if self._video_renderer is not None else None
    )
    self._video_frames = []
    return ObservationPacket(
      term_values=self.obs_adapter.compute_raw_terms(),
      action_term_joint_pos=self.action_adapter.current_joint_pos_by_term(),
    )

  def step_rollout(self, action_packet: ServerActionPacket) -> ObservationPacket:
    self.action_adapter.apply_server_action(
      raw_action=action_packet.raw_action,
      commanded_joint_pos=action_packet.joint_position_commands,
    )
    for substep_idx in range(self.decimation):
      self.action_adapter.apply_substep(self.data, substep_idx, self.decimation)
      mujoco.mj_step(self.model, self.data)
    # Keep derived quantities fresh for observation terms.
    mujoco.mj_forward(self.model, self.data)
    self.maybe_capture_frame()
    return ObservationPacket(
      term_values=self.obs_adapter.compute_raw_terms(),
      action_term_joint_pos=self.action_adapter.current_joint_pos_by_term(),
    )

  def maybe_capture_frame(self) -> None:
    if self._video_renderer is None:
      return
    if self._render_camera is None:
      self._video_renderer.update_scene(self.data)
    else:
      self._video_renderer.update_scene(self.data, camera=self._render_camera)
    self._video_frames.append(self._video_renderer.render())

  def finish_rollout(self, success: bool) -> None:
    if self._video_renderer is not None and len(self._video_frames) > 0:
      video_path = self.video_path
      video_path.parent.mkdir(parents=True, exist_ok=True)
      fps = self.cfg.video_fps or max(1, int(round(1.0 / self.step_dt)))
      media.write_video(str(video_path), self._video_frames, fps=fps)
      print(f"[INFO] Saved native sim2sim video: {video_path} (fps={fps})")
    self._video_renderer = None
    self._render_camera = None
    self._video_frames = []
