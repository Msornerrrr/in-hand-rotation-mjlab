"""Filmstrip PNG + MP4 of a palm-up cube-yaw policy (PPO checkpoint or world-model actor) on the play env.

  MUJOCO_GL=egl uv run python scripts/render/render_yaw.py --policy <ckpt> --out outputs/video/name
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

import in_hand_rotation_mjlab.tasks  # noqa: F401
from in_hand_rotation_mjlab.utils.rl import prepare_agent_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("eval_yaw", Path(__file__).resolve().parents[1] / "eval" / "eval_yaw.py")
_ev = importlib.util.module_from_spec(_spec)
sys.modules["eval_yaw"] = _ev
_spec.loader.exec_module(_ev)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", required=True)
  ap.add_argument("--envs", type=int, default=3)
  ap.add_argument("--seconds", type=float, default=12.0)
  ap.add_argument("--frames", type=int, default=8, help="filmstrip frames per env")
  ap.add_argument("--size", type=int, default=320)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  dev = args.device
  cfg = load_env_cfg(_ev.TASK, play=True)
  cfg.scene.num_envs = args.envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.seconds + 5
  env = ManagerBasedRlEnv(cfg=cfg, device=dev)
  agent_cfg = load_rl_cfg(_ev.TASK)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  from rsl_rl.runners import OnPolicyRunner

  policy = _ev.load_policy(args.policy, lambda: OnPolicyRunner(wenv, prepare_agent_cfg(agent_cfg), device=dev), dev)
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  qs = []
  with torch.no_grad():
    for _ in range(int(args.seconds / env.step_dt)):
      obs, *_ = wenv.step(policy(obs))
      qs.append(env.sim.data.qpos.clone().cpu().numpy())
  qs = np.stack(qs)  # (T, N, nq)
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  # each env sits at its own world offset: place the cube relative to that env's hand base
  robot, cube_ent = env.scene["robot"], env.scene["cube"]
  base = robot.data.root_link_pos_w.cpu().numpy()
  oq = int(cube_ent.indexing.free_joint_q_adr[0])
  ren = mujoco.Renderer(m, args.size, args.size)
  cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube/cube")
  root_id = int(robot.indexing.body_ids[0])
  d.qpos[:] = qs[0, 0]
  mujoco.mj_forward(m, d)
  root_pos = d.xpos[root_id].copy()  # the hand base in the native (single-world) model
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 0.32, -50.0, 150.0
  video, strips = [], []
  for e in range(args.envs):
    row = []
    for t in range(0, len(qs), 2):
      d.qpos[:] = qs[t, e]
      d.qpos[oq:oq + 3] = qs[t, e, oq:oq + 3] - base[e] + root_pos
      mujoco.mj_forward(m, d)
      cam.lookat[:] = d.xpos[cube]
      ren.update_scene(d, cam)
      img = ren.render()
      if t // 2 < len(video):
        video[t // 2] = np.concatenate([video[t // 2], img], 1)
      else:
        video.append(img)
      row.append(img)
    pick = np.linspace(0, len(row) - 1, args.frames).astype(int)
    strip = Image.fromarray(np.concatenate([row[i] for i in pick], 1))
    dr = ImageDraw.Draw(strip)
    for j, i in enumerate(pick):
      dr.text((j * args.size + 5, 5), f"env {e}  t={2 * i * env.step_dt:.1f}s", fill=(255, 255, 0))
    strips.append(np.asarray(strip))
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(np.concatenate(strips, 0)).save(out.with_suffix(".png"))
  mediapy.write_video(str(out.with_suffix(".mp4")), video, fps=int(round(1 / (2 * env.step_dt))))
  print(f"[render_yaw] wrote {out.with_suffix('.png')} and .mp4")


if __name__ == "__main__":
  main()
