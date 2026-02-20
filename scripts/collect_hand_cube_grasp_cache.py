"""Collect size-conditioned hand-cube grasp cache by passive settling.

For each cube half-size on a regular grid, place the cube at the default
spawn pose with the hand at its home keyframe, simulate passive settling,
and record (cube_size, joint_pos, cube_pose_rel) for stable grasps.

The output cache is used by ``hand_cube_mdp.reset_from_grasp_cache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class CollectConfig:
  out_file: str = "src/in_hand_rotation_mjlab/tasks/hand_cube/cache/leap_left_grasp_cache.npz"
  num_envs: int = 256
  settle_steps: int = 240
  device: str | None = None
  seed: int = 1

  # Cube size grid (half-sizes). Default: 0.0375 * [0.9, 1.1].
  size_min: float = 0.03375
  size_max: float = 0.04125
  num_sizes: int = 32

  # Stability filters
  lin_vel_threshold: float = 0.08
  ang_vel_threshold: float = 1.2
  palm_center_geom_expr: str = "palm_collision_.*"
  max_palm_distance: float = 0.14
  min_cube_height: float = 0.05


def run_collect(task_id: str, cfg: CollectConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=False)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs

  # Disable terminations that would reset envs during settling.
  if env_cfg.terminations is not None:
    for key in ("cube_fell", "cube_far_from_palm"):
      if key in env_cfg.terminations:
        env_cfg.terminations[key] = None

  # Disable cache-based resets and startup size randomization (we set sizes
  # explicitly from a grid).
  if env_cfg.events is not None:
    for key in ("reset_from_grasp_cache", "randomize_cube_size_once"):
      if key in env_cfg.events:
        env_cfg.events[key] = None

  step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  required_ep_len_s = (cfg.settle_steps + 2) * step_dt
  env_cfg.episode_length_s = max(env_cfg.episode_length_s, required_ep_len_s)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  robot = env.scene["robot"]
  cube = env.scene["cube"]
  action_dim = env.action_manager.total_action_dim

  # Resolve palm geom/body IDs for distance checks.
  palm_geom_ids, _ = robot.find_geoms(
    cfg.palm_center_geom_expr, preserve_order=True
  )
  palm_body_id, _ = robot.find_bodies("palm", preserve_order=True)
  if len(palm_body_id) != 1:
    raise RuntimeError("Could not uniquely resolve palm body.")
  palm_body_id = palm_body_id[0]
  use_palm_geom_center = len(palm_geom_ids) > 0

  # Resolve cube geom ID for setting sizes.
  cube_geom_local_id, _ = cube.find_geoms("cube_geom", preserve_order=True)
  if len(cube_geom_local_id) != 1:
    raise RuntimeError("Could not uniquely resolve cube geom.")
  cube_geom_world_id = int(cube.indexing.geom_ids[cube_geom_local_id[0]].item())

  # Build the size grid.
  sizes = np.linspace(cfg.size_min, cfg.size_max, cfg.num_sizes, dtype=np.float32)

  size_buf: list[np.ndarray] = []
  joint_buf: list[np.ndarray] = []
  pose_rel_buf: list[np.ndarray] = []

  zero_action = torch.zeros((cfg.num_envs, action_dim), device=env.device)
  all_env_ids = torch.arange(cfg.num_envs, device=env.device)

  total_kept = 0
  for si, half_size in enumerate(sizes):
    # Set all envs to this cube size (isotropic box).
    env.sim.model.geom_size[all_env_ids, cube_geom_world_id, :] = float(half_size)

    # Reset (hand goes to home keyframe + small noise, cube to spawn pos + small noise).
    env.reset()

    # Passive settle: zero-action lets the cube rest in the hand.
    for _ in range(cfg.settle_steps):
      env.step(zero_action)

    # --- stability check ---
    cube_pos_w = cube.data.root_link_pos_w
    cube_quat_w = cube.data.root_link_quat_w
    cube_lin_vel_w = cube.data.root_link_lin_vel_w
    cube_ang_vel_w = cube.data.root_link_ang_vel_w

    if use_palm_geom_center:
      palm_center_w = robot.data.geom_pos_w[:, palm_geom_ids].mean(dim=1)
    else:
      palm_center_w = robot.data.body_link_pos_w[:, palm_body_id]
    cube_dist = torch.norm(cube_pos_w - palm_center_w, dim=-1)

    stable = torch.ones(cfg.num_envs, dtype=torch.bool, device=env.device)
    stable &= env.episode_length_buf >= cfg.settle_steps
    stable &= torch.norm(cube_lin_vel_w, dim=-1) <= cfg.lin_vel_threshold
    stable &= torch.norm(cube_ang_vel_w, dim=-1) <= cfg.ang_vel_threshold
    stable &= cube_dist <= cfg.max_palm_distance
    stable &= cube_pos_w[:, 2] >= cfg.min_cube_height

    stable_ids = stable.nonzero(as_tuple=False).squeeze(-1)
    kept = int(stable_ids.numel())

    if kept > 0:
      cube_pose_rel = torch.cat(
        [
          cube_pos_w[stable_ids] - env.scene.env_origins[stable_ids],
          cube_quat_w[stable_ids],
        ],
        dim=-1,
      )
      size_buf.append(np.full(kept, half_size, dtype=np.float32))
      joint_buf.append(robot.data.joint_pos[stable_ids].cpu().numpy())
      pose_rel_buf.append(cube_pose_rel.cpu().numpy())

    total_kept += kept
    print(
      f"[collect] size {si + 1}/{cfg.num_sizes}  "
      f"half_size={half_size:.5f}  kept={kept}/{cfg.num_envs}  total={total_kept}"
    )

  env.close()

  if total_kept == 0:
    raise RuntimeError(
      "No stable samples collected. Check spawn pose, settle_steps, or thresholds."
    )

  out_path = Path(cfg.out_file)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  np.savez_compressed(
    out_path,
    cube_size=np.concatenate(size_buf).astype(np.float32),
    joint_pos=np.concatenate(joint_buf).astype(np.float32),
    cube_pose_rel=np.concatenate(pose_rel_buf).astype(np.float32),
    joint_names=np.asarray(robot.joint_names),
  )
  print(f"[collect] wrote {total_kept} samples ({cfg.num_sizes} sizes) to {out_path}")


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  cfg = tyro.cli(
    CollectConfig,
    args=remaining_args,
    default=CollectConfig(),
    prog=f"scripts/collect_hand_cube_grasp_cache.py {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_collect(chosen_task, cfg)


if __name__ == "__main__":
  main()
