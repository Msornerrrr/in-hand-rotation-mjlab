"""Train a PPO policy (RSL-RL) on a task.

  uv run python scripts/train.py Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear --env.scene.num-envs 4096
  uv run python scripts/train.py <task> --gpu-ids all                     # multi-GPU (torchrunx)
  uv run python scripts/train.py <task> --init-checkpoint-file <ckpt>     # warm start (fine-tune stages)

Runs log to logs/rsl_rl/<experiment>/<date>_<run name>/.
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from in_hand_rotation_mjlab.utils.nan_guard import NanSafeVecEnvWrapper
from in_hand_rotation_mjlab.utils.plant import check_checkpoint_plant, write_run_plant
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder

from in_hand_rotation_mjlab.utils.distributed_nan_guard import enable_nan_guard_for_rank


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  video: bool = False
  # None maps to env.max_episode_length (full nominal horizon) in run_train().
  video_length: int | None = None
  video_interval: int = 2000
  enable_nan_guard: bool = False
  # Wrap the env in NanSafeVecEnvWrapper (force-resets envs with non-finite obs / reward instead of crashing).
  wrap_nan_safe: bool = False
  init_checkpoint_file: str | None = None
  init_checkpoint_load_critic: bool = False
  init_checkpoint_load_optimizer: bool = False
  resume_env_step_count: int | None = None
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _restore_env_step_count(env, cfg: "TrainConfig") -> None:
  """Set the env's control-step counter (``--resume-env-step-count``; step-counted curricula read it)."""
  if cfg.resume_env_step_count is None:
    return
  if cfg.resume_env_step_count < 0:
    raise ValueError("resume_env_step_count must be nonnegative")
  env.unwrapped.common_step_counter = cfg.resume_env_step_count
  env.unwrapped._sim_step_counter = cfg.resume_env_step_count * cfg.env.decimation
  print(f"[INFO]: Restored environment control step: {cfg.resume_env_step_count}")


def _prepare_agent_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict:
  """Convert dataclass config to runner dict and drop incompatible MLP fields."""
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      class_name = model_cfg.get("class_name", "MLPModel")
      if class_name != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
      if class_name != "RNNModel":
        model_cfg.pop("rnn_type", None)
        model_cfg.pop("rnn_hidden_dim", None)
        model_cfg.pop("rnn_num_layers", None)
  return cfg_dict


def _restore_adaptive_learning_rate(runner: OnPolicyRunner) -> float | None:
  """Synchronize RSL-RL's adaptive-KL scalar after optimizer resume.

  PPO checkpoints store the optimizer (including its current learning rate),
  but upstream RSL-RL keeps a second ``algorithm.learning_rate`` scalar that
  is not serialized.  Without this synchronization, the first resumed PPO
  update starts adapting from the config's initial rate instead of the saved
  rate.
  """
  algorithm = runner.alg
  optimizer = getattr(algorithm, "optimizer", None)
  if optimizer is None or not optimizer.param_groups:
    return None
  learning_rate = float(optimizer.param_groups[0]["lr"])
  if hasattr(algorithm, "learning_rate"):
    algorithm.learning_rate = learning_rate
    return learning_rate
  return None


def _set_algorithm_learning_rate(
  runner: OnPolicyRunner, learning_rate: float
) -> None:
  """Set both RSL-RL's scalar and every optimizer parameter-group rate."""
  runner.alg.learning_rate = learning_rate
  for group in runner.alg.optimizer.param_groups:
    group["lr"] = learning_rate


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    output_dir = enable_nan_guard_for_rank(
      cfg.env, rank=rank, world_size=world_size
    )
    print(f"[INFO] NaN guard enabled, output dir: {output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  init_path: Path | None = None
  if cfg.init_checkpoint_load_optimizer and not cfg.init_checkpoint_load_critic:
    raise ValueError(
      "--init-checkpoint-load-optimizer requires "
      "--init-checkpoint-load-critic True because PPO uses one optimizer for "
      "both actor and critic parameters."
    )
  if cfg.init_checkpoint_load_optimizer and cfg.init_checkpoint_file is None:
    raise ValueError(
      "--init-checkpoint-load-optimizer requires --init-checkpoint-file."
    )
  if cfg.agent.resume:
    if cfg.init_checkpoint_file is not None:
      raise ValueError(
        "Use either --agent.resume for same-experiment resume or "
        "--init-checkpoint-file for cross-task warm start, not both."
      )
    resume_path = get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
  elif cfg.init_checkpoint_file is not None:
    init_path = Path(cfg.init_checkpoint_file).expanduser().resolve()
    if not init_path.exists():
      raise FileNotFoundError(f"Init checkpoint file not found: {init_path}")

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    # Default to a full horizon clip when video_length is not explicitly set.
    # This avoids ultra-short clips when env[0] terminates early.
    video_length = (
      cfg.video_length
      if cfg.video_length is not None
      else env.unwrapped.max_episode_length
    )
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      # `step` is the environment control-step index on this rank.
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=video_length,
      disable_logger=True,
      log_to_wandb=cfg.agent.logger == "wandb",
      wandb_log_key="videos/train",
    )
    print(
      "[INFO] Video config: "
      f"interval_steps={cfg.video_interval}, length_frames={video_length}"
    )
    print("[INFO] Recording videos during training.")

  wrapper_cls = NanSafeVecEnvWrapper if cfg.wrap_nan_safe else RslRlVecEnvWrapper
  env = wrapper_cls(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = _prepare_agent_cfg(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    check_checkpoint_plant(resume_path, cfg.env, task=task_id)
    # A distributed checkpoint is written by rank 0, so its optimizer tensors
    # are tagged with cuda:0.  Remap them to each worker's local CUDA device;
    # otherwise rank > 0 fails its first Adam step with mixed-device state and
    # rank 0 eventually times out in the corresponding NCCL all-reduce.
    runner.load(str(resume_path), map_location=device)
    restored_lr = _restore_adaptive_learning_rate(runner)
    if restored_lr is not None:
      print(f"[INFO]: Restored adaptive learning rate: {restored_lr:.9g}")
    # RSL-RL saves ``iter`` after completing that iteration, but its learning
    # loop starts at ``current_learning_iteration``.  Advance once so resuming
    # model_600 continues with update 601 instead of repeating update 600.
    completed_iteration = runner.current_learning_iteration
    runner.current_learning_iteration = completed_iteration + 1
    _restore_env_step_count(env, cfg)
    print(
      f"[INFO]: Resuming after completed iteration {completed_iteration}; "
      f"next iteration is {runner.current_learning_iteration}"
    )
  elif init_path is not None:
    check_checkpoint_plant(init_path, cfg.env, task=task_id)
    target_learning_rate = float(runner.alg.learning_rate)
    print(
      f"[INFO]: Warm-starting actor"
      f"{' and critic' if cfg.init_checkpoint_load_critic else ''} "
      f"{'with optimizer state ' if cfg.init_checkpoint_load_optimizer else ''}"
      f"from checkpoint: {init_path}"
    )
    runner.load(
      str(init_path),
      load_cfg={
        "actor": True,
        "critic": cfg.init_checkpoint_load_critic,
        "optimizer": cfg.init_checkpoint_load_optimizer,
        "iteration": False,
        "rnd": False,
      },
      map_location=device,
    )
    # A warm start into a task with step-counted curricula continues them from --resume-env-step-count.
    _restore_env_step_count(env, cfg)
    if cfg.init_checkpoint_load_optimizer:
      # Keep Adam moments/step counters from the source policy, but honor the
      # target task's deliberately conservative fine-tuning rate.
      _set_algorithm_learning_rate(runner, target_learning_rate)
      print(
        "[INFO]: Restored optimizer state with target-task learning rate: "
        f"{target_learning_rate:.9g}"
      )
    # Preserve the exact post-load/pre-update state. model_0.pt is saved after
    # PPO iteration zero and is therefore not a trustworthy initialization
    # baseline when diagnosing catastrophic forgetting.
    if rank == 0:
      log_dir.mkdir(parents=True, exist_ok=True)
      initial_state = runner.alg.save()
      initial_state["iter"] = -1
      initial_state["infos"] = {"source_checkpoint": str(init_path)}
      torch.save(initial_state, log_dir / "model_init.pt")
      print(f"[INFO]: Saved exact warm-start baseline: {log_dir / 'model_init.pt'}")

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
    write_run_plant(log_dir, cfg.env)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = [t for t in list_tasks() if t.startswith("Mjlab-Leap")]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
