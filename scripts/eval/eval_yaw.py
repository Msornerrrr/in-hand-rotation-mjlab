"""Evaluate a palm-up cube-yaw policy (PPO rsl_rl checkpoint or a world-model-trained actor) the same way: the play
env of ``Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear`` (plant, contact and object randomization on), N envs for S
seconds with the deterministic policy, auto-reset on drops.

Metrics: the yaw rate the task rewards (world yaw, sign of the left hand, rad/s) averaged over all env-steps, per
15 s window; drops per env-minute and the share of envs that dropped at least once; the env's own mean step reward
(per second, i.e. / dt) and each term.

  MUJOCO_GL=egl uv run python scripts/eval/eval_yaw.py --policy <model_2999.pt | actor.pt> \
      --out outputs/policy_eval/<name>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import in_hand_rotation_mjlab.tasks  # noqa: F401
from in_hand_rotation_mjlab.utils.rl import prepare_agent_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

TASK = "Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear"


def load_policy(path: str, runner_factory, device: str):
  sd = torch.load(path, map_location=device, weights_only=False)
  if isinstance(sd, dict) and sd.get("kind") == "wm_actor":
    from in_hand_rotation_mjlab.world_model.policy import Actor

    actor = Actor.from_checkpoint(sd).to(device).eval()
    return lambda obs: actor(obs["actor"] if not torch.is_tensor(obs) else obs)
  runner = runner_factory()
  runner.load(path, map_location=device)
  return runner.get_inference_policy(device=device)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", required=True)
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--envs", type=int, default=512)
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  dev = args.device
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.envs
  env_cfg.seed = args.seed
  env_cfg.episode_length_s = args.seconds + 5.0
  env = ManagerBasedRlEnv(cfg=env_cfg, device=dev)
  agent_cfg = load_rl_cfg(args.task)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  from rsl_rl.runners import OnPolicyRunner

  policy = load_policy(args.policy, lambda: OnPolicyRunner(wenv, prepare_agent_cfg(agent_cfg), device=dev), dev)
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  cube = env.scene["cube"]
  steps = int(round(args.seconds / env.step_dt))
  win = int(round(15.0 / env.step_dt))
  prev_yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)[2].clone()
  yaw_sum = torch.zeros(steps, device=dev)
  drops = torch.zeros(args.envs, device=dev)
  rew_sum, terms = 0.0, {}
  t0 = time.time()
  with torch.no_grad():
    for t in range(steps):
      a = policy(obs)
      obs, r, d, _ = wenv.step(a)
      yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)[2]
      dyaw = -wrap_to_pi(yaw - prev_yaw)  # left hand: clockwise positive (negate_yaw_rate)
      d = d.bool()
      term = env.reset_terminated.bool() & d
      drops += term.float()
      dyaw = torch.where(d, torch.zeros_like(dyaw), dyaw)  # a reset is not rotation
      yaw_sum[t] = dyaw.mean()
      prev_yaw = yaw.clone()
      rew_sum += float(r.mean())
      rm = env.reward_manager  # _step_reward: (N, terms) reward rates (raw x weight, per second)
      for i, k in enumerate(rm._term_names):
        terms[k] = terms.get(k, 0.0) + float(rm._step_reward[:, i].mean())
  dt = env.step_dt
  rate = yaw_sum / dt
  windows = [float(rate[i:i + win].mean()) for i in range(0, steps, win)]
  res = dict(policy=args.policy, envs=args.envs, seconds=args.seconds, yaw_rate=float(rate.mean()), yaw_rate_windows=windows,
             drops_per_env_min=float(drops.sum()) / args.envs / (args.seconds / 60.0),
             share_dropped=float((drops > 0).float().mean()), reward_per_s=rew_sum / steps / dt,
             reward_terms_per_s={k: v / steps for k, v in terms.items()}, eval_seconds=time.time() - t0)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=1))
  print(f"[eval_yaw] {Path(args.policy).name}: yaw {res['yaw_rate']:.3f} rad/s (15 s windows "
        f"{', '.join(f'{w:.3f}' for w in windows)}), drops/env-min {res['drops_per_env_min']:.3f}, "
        f"dropped {100 * res['share_dropped']:.1f}%, reward/s {res['reward_per_s']:.3f}")


if __name__ == "__main__":
  main()
