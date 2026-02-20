"""Record a replay trajectory by rolling out a trained policy in mjwarp env."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import sys
from typing import Any

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from in_hand_rotation_mjlab.policy_server.replay import ReplayTrajectory, save_replay_trajectory


@dataclass(frozen=True)
class RecordReplayTrajectoryCliConfig:
  checkpoint_file: str
  output_file: str | None = None
  device: str | None = None
  play: bool = True
  num_steps: int | None = None
  log_interval: int = 200
  save_video: bool = True
  video_file: str | None = None
  video_height: int = 720
  video_width: int = 1280


def _prepare_agent_cfg(agent_cfg) -> dict:
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      if model_cfg.get("class_name", "MLPModel") != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
  return cfg_dict


def _resolve_default_num_steps(task_id: str, play: bool, requested: int | None) -> int:
  if requested is not None:
    return int(requested)
  env_cfg = load_env_cfg(task_id, play=play)
  step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  if play and env_cfg.episode_length_s > 1e6:
    train_cfg = load_env_cfg(task_id, play=False)
    train_dt = train_cfg.sim.mujoco.timestep * train_cfg.decimation
    return math.ceil(train_cfg.episode_length_s / train_dt)
  return math.ceil(env_cfg.episode_length_s / step_dt)


def _default_output_path(checkpoint_file: str, num_steps: int) -> Path:
  ckpt = Path(checkpoint_file).expanduser().resolve()
  run_dir = ckpt.parent
  return run_dir / "trajectories" / f"replay_{ckpt.stem}_{num_steps}steps.npz"


def _default_video_path(checkpoint_file: str, num_steps: int) -> Path:
  ckpt = Path(checkpoint_file).expanduser().resolve()
  run_dir = ckpt.parent
  return run_dir / "videos" / "replay" / f"replay_{ckpt.stem}_{num_steps}steps.mp4"


def main() -> None:
  # Import tasks to populate registry.
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    RecordReplayTrajectoryCliConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  num_steps = _resolve_default_num_steps(chosen_task, args.play, args.num_steps)

  env_cfg = load_env_cfg(chosen_task, play=args.play)
  env_cfg.scene.num_envs = 1
  if args.video_height is not None:
    env_cfg.viewer.height = args.video_height
  if args.video_width is not None:
    env_cfg.viewer.width = args.video_width

  render_mode = "rgb_array" if args.save_video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if args.save_video:
    video_path = (
      Path(args.video_file).expanduser().resolve()
      if args.video_file is not None
      else _default_video_path(args.checkpoint_file, num_steps)
    )
    env = VideoRecorder(
      env,
      video_folder=video_path.parent,
      step_trigger=lambda step: step == 0,
      video_length=num_steps,
      disable_logger=True,
      name_prefix=video_path.stem,
    )
    print(f"[INFO] Replay recording video path: {video_path}")

  agent_cfg = load_rl_cfg(chosen_task)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(chosen_task) or OnPolicyRunner
  runner = runner_cls(vec_env, _prepare_agent_cfg(agent_cfg), device=device)
  runner.load(str(Path(args.checkpoint_file).expanduser().resolve()), map_location=device)
  policy = runner.get_inference_policy(device=device)
  policy.eval()

  print(
    "[INFO] Recording replay trajectory in mjwarp: "
    f"task={chosen_task}, steps={num_steps}, step_dt={vec_env.unwrapped.step_dt:.6f}"
  )
  step_dt = float(vec_env.unwrapped.step_dt)

  obs, _ = vec_env.reset()
  term_names = list(vec_env.unwrapped.action_manager.active_terms)
  raw_actions: list[torch.Tensor] = []
  commands_by_term: dict[str, list[torch.Tensor]] = {name: [] for name in term_names}

  with torch.no_grad():
    for step_idx in range(num_steps):
      policy_action = policy(obs)
      if agent_cfg.clip_actions is not None:
        action = torch.clamp(policy_action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
      else:
        action = policy_action

      obs, _, _, _ = vec_env.step(action)

      raw_actions.append(action[0:1].detach().to(device="cpu").clone())
      for term_name in term_names:
        term = vec_env.unwrapped.action_manager.get_term(term_name)
        processed = getattr(term, "_processed_actions", None)
        if processed is None:
          raise RuntimeError(
            f"Action term '{term_name}' has no '_processed_actions' buffer."
          )
        commands_by_term[term_name].append(processed[0:1].detach().to(device="cpu").clone())

      if step_idx == 0 or ((step_idx + 1) % max(1, int(args.log_interval)) == 0):
        print(f"[INFO] Recorded step {step_idx + 1}/{num_steps}")

  vec_env.close()

  raw_action = torch.cat(raw_actions, dim=0)
  joint_commands = {
    term_name: torch.cat(cmd_list, dim=0)
    for term_name, cmd_list in commands_by_term.items()
  }
  trajectory = ReplayTrajectory(
    step_dt=step_dt,
    raw_action=raw_action,
    joint_position_commands=joint_commands,
    task_id=chosen_task,
    checkpoint_file=str(Path(args.checkpoint_file).expanduser().resolve()),
  )

  out_path = (
    Path(args.output_file).expanduser().resolve()
    if args.output_file is not None
    else _default_output_path(args.checkpoint_file, trajectory.num_steps)
  )
  saved = save_replay_trajectory(out_path, trajectory)
  print(
    "[INFO] Saved replay trajectory: "
    f"{saved} (steps={trajectory.num_steps}, terms={list(joint_commands.keys())})"
  )


if __name__ == "__main__":
  main()
