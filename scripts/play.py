"""Play a PPO checkpoint (or a zero / random agent) in a task's play env, in the native viewer, a viser web viewer, or to
an mp4.

  uv run python scripts/play.py Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear --checkpoint-file ckpts/yaw/ppo_gear_yaw.pt
  MUJOCO_GL=egl uv run python scripts/play.py <task> --checkpoint-file <ckpt> --num-envs 1 --video True --viewer none

World-model actors (``actor_*.pt``) use their own observation adapters: see scripts/render/ and scripts/eval/.
"""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from in_hand_rotation_mjlab.utils.plant import check_checkpoint_plant
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser", "none"] = "auto"
  """"none": no viewer; with --video True, record the clip and exit (headless machines)."""


class TaskCenteredViserPlayViewer(ViserPlayViewer):
  """Viser viewer that honors the task viewer body as its tracked center."""

  def setup(self) -> None:
    super().setup()
    self._center_on_viewer_body()

  def _center_on_viewer_body(self) -> None:
    viewer_cfg = self.cfg
    scene = getattr(self, "_scene", None)
    server = getattr(self, "_server", None)
    if scene is None or server is None:
      return

    body_id = -1
    if getattr(viewer_cfg, "entity_name", None) and getattr(viewer_cfg, "body_name", None):
      model_body_name = f"{viewer_cfg.entity_name}/{viewer_cfg.body_name}"
      body_id = mujoco.mj_name2id(
        scene.mj_model, mujoco.mjtObj.mjOBJ_BODY, model_body_name
      )
      if body_id < 0:
        body_id = mujoco.mj_name2id(
          scene.mj_model, mujoco.mjtObj.mjOBJ_BODY, viewer_cfg.body_name
        )

    if body_id >= 0:
      scene._tracked_body_id = body_id
      scene.request_update()

    azimuth = np.deg2rad(float(viewer_cfg.azimuth))
    elevation = np.deg2rad(float(viewer_cfg.elevation))
    distance = float(viewer_cfg.distance)
    camera_offset = np.array(
      [
        -np.cos(elevation) * np.cos(azimuth),
        -np.cos(elevation) * np.sin(azimuth),
        # ViewerConfig follows MuJoCo's convention: negative elevation means
        # above the look-at point. Viser uses an explicit Cartesian offset.
        -np.sin(elevation),
      ]
    ) * distance

    def apply_camera(client) -> None:
      client.camera.position = camera_offset
      client.camera.look_at = np.zeros(3)

    for client in server.get_clients().values():
      apply_camera(client)

    @server.on_client_connect
    def _(client) -> None:
      apply_camera(client)


def _prepare_agent_cfg(agent_cfg) -> dict:
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


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    del log_root_path
    if cfg.checkpoint_file is None:
      raise ValueError("--checkpoint-file is required for the trained agent (or use --agent zero / random)")
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    # a training run's directory keeps its videos; a loose checkpoint (e.g. ckpts/) writes under outputs/videos/<name>
    log_dir = resume_path.parent if (resume_path.parent / "params").is_dir() else Path("outputs/videos") / resume_path.stem

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    elif cfg.agent == "random":

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
    else:
      raise RuntimeError(f"Unsupported dummy agent type: {cfg.agent}")
  else:
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, _prepare_agent_cfg(agent_cfg), device=device)
    # the run recorded its training cfg, so compare against the task's training plant
    check_checkpoint_plant(resume_path, load_env_cfg(task_id), task=task_id)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "none":
    obs = env.get_observations()
    with torch.no_grad():
      for _ in range(cfg.video_length + 1):
        obs, _, _, _ = env.step(policy(obs))
    print(f"[INFO]: Ran {cfg.video_length} steps without a viewer" + (f"; video in {log_dir / 'videos' / 'play'}" if cfg.video and log_dir else ""))
  elif resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    TaskCenteredViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


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

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
