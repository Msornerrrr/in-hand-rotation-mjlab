"""MP4s of palm-down mixed-object fingertip-rotation rollouts recorded by ``eval_ftrot.py --record``.

Each recording holds every env's qpos per step; env e holds object e mod 22. Panels are replayed on a CPU copy of the
scene with that env's object geometry, two views each (a 3/4 view from below, and straight up the palm normal, which
shows the yaw). Overlays: policy, object, pad friction, time, net yaw, per-pad contact force bars (I M R T) and a
DROPPED banner from the first drop on (the env then restarts from a new hold).

  # one video per object, policies side by side (the same rollout index r of each recording):
  python scripts/render/render_ftrot.py --rec "World model B=B.rec.pt" "Student=S.rec.pt" --rollout 0 --out-dir vids/
  # one grid of every object for one recording:
  python scripts/render/render_ftrot.py --rec "World model B=B.rec.pt" --grid --out vids/grid.mp4
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

AX = [(0.95, 0.15, 0.15, 1), (0.15, 0.85, 0.15, 1), (0.2, 0.35, 1.0, 1)]
VIEWS = [(110, 20, 0.36, 0.03), (90, 89, 0.28, 0.0)]  # palm down: from below at 20 deg (look-at raised toward the palm),
# and straight up the palm normal; (azimuth, elevation, distance, look-at height offset)
_MODEL = {}


def _model(task):
  if task not in _MODEL:
    import in_hand_rotation_mjlab.tasks  # noqa: F401
    from mjlab.scene import Scene
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    _MODEL[task] = Scene(cfg.scene, device="cpu").compile()
  return _MODEL[task]


def _load(path):
  import torch

  r = torch.load(path, weights_only=False, map_location="cpu")
  return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in r.items()}


def render_env(rec_path, env, size, every, seconds, label):
  """Frames (T, 2*size, size, 3) of one env of one recording, both views stacked."""
  import mujoco
  from PIL import Image, ImageDraw

  r = _load(rec_path)
  m = _model(r["task"])
  d = mujoco.MjData(m)
  gid = list(r["geom_ids"])
  m.geom_size[gid] = r["geom_size"][env]
  for g in gid:  # the multi-shape object: hide the inactive primitives, keep the active one half transparent
    active = m.geom_size[g][0] > 0.002
    m.geom_rgba[g] = (0.95, 0.75, 0.25, 0.55) if active else (0, 0, 0, 0)
  for g in range(m.ngeom):  # readable on a light background: grey links, no floor plane
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
      m.geom_rgba[g, 3] = 0.0
    elif g not in gid and m.geom_rgba[g, 3] > 0 and m.geom_rgba[g, :3].max() < 0.3:
      m.geom_rgba[g, :3] = (0.42, 0.45, 0.52)
  m.vis.headlight.ambient[:] = 0.45
  m.vis.headlight.diffuse[:] = 0.6
  cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube/cube")
  rd = mujoco.Renderer(m, size, size)
  cam = mujoco.MjvCamera()
  dt = float(r["step_dt"])
  T = min(len(r["qpos"]), int(round(seconds / dt)) + 1)
  qpos = r["qpos"][:, env].astype(np.float64).copy()
  qpos[:, -7:-4] -= r["mocap_pos"][env, 0] - m.body("robot/mocap_base").pos  # into the single model's frame
  look = qpos[0, -7:-4].copy()
  t_fail = float(r["t_fail"][env])
  obj, mu = r["obj"][env], float(r["mu"][env])
  frames = []
  for t in range(0, T, every):
    d.qpos[:] = qpos[t]
    mujoco.mj_forward(m, d)
    views = []
    for az, el, dist, dz in VIEWS:
      cam.lookat[:] = look + (0.0, 0.0, dz)
      cam.distance, cam.azimuth, cam.elevation = dist, az, el
      rd.update_scene(d, cam)
      c, R = d.xpos[cube], d.xmat[cube].reshape(3, 3)
      for k in range(3):
        if rd.scene.ngeom < rd.scene.maxgeom:
          g = rd.scene.geoms[rd.scene.ngeom]
          mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(AX[k], np.float32))
          mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.003, c, c + 0.06 * R[:, k])
          rd.scene.ngeom += 1
      px = rd.render().copy()
      px[(px == 0).all(-1)] = (236, 238, 242)  # empty background
      views.append(px)
    img = Image.fromarray(np.concatenate(views, 0))
    dr = ImageDraw.Draw(img)
    tt = t * dt
    dropped = 0 <= t_fail <= tt + 1e-6
    dr.rectangle((0, 0, size, 48), fill=(0, 0, 0))
    dr.text((6, 3), label, fill=(255, 255, 255))
    dr.text((6, 18), f"{obj}  pad mu {mu:.2f}", fill=(200, 220, 255))
    dr.text((6, 33), f"t {tt:4.1f}s  yaw {np.degrees(float(r['yaw'][t, env])):+5.0f} deg", fill=(255, 230, 120))
    for i, (lab, f) in enumerate(zip("IMRT", r["pads"][t, env])):
      x0 = size - 66 + i * 16
      h = min(30.0, 6.0 * float(f))
      dr.rectangle((x0, 46 - h, x0 + 10, 46), fill=(90, 200, 255))
      dr.text((x0 + 1, 50), lab, fill=(255, 255, 255))
    if dropped:
      dr.rectangle((0, size - 22, size, size), fill=(170, 20, 20))
      dr.text((6, size - 18), f"DROPPED at {t_fail:.1f}s (restarted)", fill=(255, 255, 255))
    frames.append(np.asarray(img))
  return frames


def _job(a):
  return render_env(*a)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rec", nargs="+", required=True, help='"Label=path.rec.pt" (or a path)')
  ap.add_argument("--rollout", type=int, default=0, help="rollout index per object (env = rollout * 22 + object)")
  ap.add_argument("--objects", nargs="*", default=None, help="object names (default: all)")
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--size", type=int, default=240)
  ap.add_argument("--every", type=int, default=1, help="render every k-th control step (fps = 20 / k)")
  ap.add_argument("--grid", action="store_true", help="one video, all objects of the first recording in a grid")
  ap.add_argument("--cols", type=int, default=6)
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--out-dir", default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--crf", type=int, default=30)
  a = ap.parse_args()
  import mediapy

  recs = [(s.split("=", 1) if "=" in s else (Path(s).stem, s)) for s in a.rec]
  r0 = _load(recs[0][1])
  names = list(dict.fromkeys(r0["obj"]))
  n_obj = len(names)
  objs = a.objects or names
  fps = 20.0 / a.every
  with ProcessPoolExecutor(a.workers) as pool:
    if a.grid:
      label, path = recs[0]
      jobs = [(path, a.rollout * n_obj + names.index(o), a.size, a.every, a.seconds, f"{label}") for o in objs]
      panels = list(pool.map(_job, jobs))
      n = min(len(p) for p in panels)
      rows = []
      blank = np.zeros_like(panels[0][0])
      cols = a.cols
      grid = []
      for t in range(n):
        tiles = [p[t] for p in panels] + [blank] * (-len(panels) % cols)
        rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
        grid.append(np.concatenate(rows, 0))
      mediapy.write_video(a.out, grid, fps=fps, crf=a.crf)
      print(a.out, len(grid), "frames")
      return
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs, keys = [], []
    for o in objs:
      for label, path in recs:
        jobs.append((path, a.rollout * n_obj + names.index(o), a.size, a.every, a.seconds, label))
        keys.append(o)
    res = list(pool.map(_job, jobs))
    meta = {}
    k = len(recs)
    for i, o in enumerate(objs):
      panels = res[i * k:(i + 1) * k]
      n = min(len(p) for p in panels)
      sep = np.full((panels[0][0].shape[0], 4, 3), 40, np.uint8)
      video = [np.concatenate(sum(([p[t], sep] for p in panels), [])[:-1], 1) for t in range(n)]
      f = out / f"{o}.mp4"
      mediapy.write_video(str(f), video, fps=fps, crf=a.crf)
      meta[o] = {lab: dict(yaw_deg=float(np.degrees(_load(p)["yaw_final"][a.rollout * n_obj + names.index(o)])),
                           dropped_at=float(_load(p)["t_fail"][a.rollout * n_obj + names.index(o)]))
                 for lab, p in recs}
      print(f, n, "frames", os.path.getsize(f) // 1024, "KB")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
  main()
