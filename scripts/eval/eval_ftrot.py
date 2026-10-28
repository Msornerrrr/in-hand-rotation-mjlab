"""Palm-down 22-object fingertip rotation: one evaluation for PPO checkpoints (teacher / student, the env's own
observation) and world-model actors (the ``ftrot_mix`` adapter's observation), as ``fingertip_rotate/eval_policy.py
--mixed`` scores them: the play env (plant / contact / object randomization on, no pushes), env e holds object e mod 22,
``--per-obj`` rollouts each, ``--seconds`` long; a rollout's yaw stops counting at its first drop.

Per object and per shape: net yaw over the run (deg), the share of rollouts that dropped, the share of time all four
fingers took part (each pressed within the last second) and pads_low.

  MUJOCO_GL=egl uv run python scripts/eval/eval_ftrot.py --policy <ckpt> \
      [--task Mjlab-...-PalmDown-MixTeacherF | -PalmDown-Mix] --out <json>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import in_hand_rotation_mjlab.tasks  # noqa: F401
from in_hand_rotation_mjlab.tasks.fingertip_rotate import rl_mdp as M
from in_hand_rotation_mjlab.tasks.fingertip_rotate.env_cfg import PAD_SENSOR
from in_hand_rotation_mjlab.tasks.fingertip_rotate.objects import OBJECTS
from in_hand_rotation_mjlab.utils.rl import prepare_agent_cfg
from in_hand_rotation_mjlab.world_model import record as R
from in_hand_rotation_mjlab.world_model.ftrot_mix import H_OBS, FtrotMix, StateEstimator
from in_hand_rotation_mjlab.world_model.policy import Actor
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

DEG = 180 / math.pi


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", required=True)
  ap.add_argument("--task", default="Mjlab-Leap-Left-Custom-Fingertip-Rotate-PalmDown-MixTeacherF")
  ap.add_argument("--per-obj", type=int, default=16)
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--seed", type=int, default=5)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", required=True)
  ap.add_argument("--dump", default=None,
                  help="world-model actors: save (actor history input, object drift, object position, pad mu, live) every "
                       "--dump-every steps to this .npz, for observability probes")
  ap.add_argument("--dump-every", type=int, default=10)
  ap.add_argument("--record", default=None,
                  help="save every env's qpos per step (+ object geom sizes, pad mu, yaw, pad forces, first-drop time) to this "
                       ".pt for render_ftrot.py; world-model actors then reset twice like the PPO path, so both start alike")
  ap.add_argument("--start-squeeze", type=float, default=None,
                  help="scale every trained grasp's command toward its joint positions: target = q + s (target - q). The real "
                       "hand started at s ~ 0.1-0.3 (squeeze ~0.01-0.025 rad on mf_pip/rf_pip/th_mcp against 0.087 trained)")
  ap.add_argument("--pad-mu", type=float, default=None, help="pin the pad friction of every rollout to this value")
  ap.add_argument("--max-drift", type=float, default=None,
                  help="override the drop threshold (m, drift from the reset position; the task's is 0.03): a larger one "
                       "separates objects that fell from objects only pushed away from where they started")
  args = ap.parse_args()
  dev = args.device
  ck = torch.load(args.policy, map_location=dev, weights_only=False)
  wm_actor = isinstance(ck, dict) and ck.get("kind") == "wm_actor"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.episode_length_s = args.seconds + 5.0
  if args.max_drift is not None:
    for grp in (env_cfg.terminations, env_cfg.rewards):
      if "dropped" in grp:
        grp["dropped"].params["max_drift"] = args.max_drift
  env_cfg.seed = args.seed
  if args.pad_mu is not None:
    env_cfg.events["dr_split_contact_friction"].params["pad_range"] = (args.pad_mu, args.pad_mu)
  ev = env_cfg.events["reset_from_pool"]
  if args.start_squeeze is not None:  # the pool is cached: edit it before the first reset reads it
    pool = M._mixed_pool(ev.params["pool_path"], "down", dev)
    jp, jt0 = pool["joint_pos"], pool["joint_target"]
    assert jp.shape == jt0.shape, (jp.shape, jt0.shape)
    pool["joint_target"] = jp + args.start_squeeze * (jt0 - jp)
  ev.params["assign_objects"] = True
  ev.params["stage2_step"] = 0
  names = [str(x) for x in np.load(ev.params["pool_path"])["obj_names"]]
  n_obj = len(names)
  envs = n_obj * args.per_obj
  env_cfg.scene.num_envs = envs
  # eval-only per-contact pad sensor (contact frame: force[0] = normal, force[1:] = friction) for the grip numbers
  pad_cfg = next(s for s in env_cfg.scene.sensors if s.name == PAD_SENSOR)
  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + (dataclasses.replace(
    pad_cfg, name="pad_contact_cf", fields=("found", "force", "normal", "tangent"), reduce="maxforce", num_slots=2),)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=dev)
  env.reset()  # the task's per-episode state (reset position p0, ...) is only set by a reset: without it every env
  # "drops" on its first step (the rsl-rl wrapper resets on construction; a bare env does not)
  if wm_actor and args.record:
    env.reset()  # the PPO path's rsl-rl wrapper resets once more: same seed -> same starts, pad mu and masses
  st = M.ft(env, "down")
  if wm_actor:
    actor = Actor.from_checkpoint(ck).to(dev).eval()
    L = R.layout(env)
    nj = L.nj
    full, cn = R.context_full(env)
    cnames = ck["ctx_names"]
    cols = [cn.index(n) for n in cnames]
    robot = env.scene["robot"]
    n_hand = quat_apply_inverse(L.base_q[:1], st.n_w[None])[0]
    weights = {k: v.weight for k, v in env_cfg.rewards.items()}
    task = FtrotMix(nj, robot.data.default_joint_pos[0, L.term.target_ids].clone(), L.base_q[0].clone(), n_hand,
                    env.step_dt, weights, cnames)
    priv = ck.get("actor_kind") == "priv"
    s = R.read_state(env, L)
    ctx = full[:, cols].clone()
    info = torch.cat([s[:, :nj], s[:, task.S["p"]], st.extent[:, None]], -1)
    fr = task.obs_frame(s[:, :nj], s[:, task.S["cmd"]], ctx, noise=False)
    hist = fr[:, None].expand(-1, H_OBS, -1).clone()
    q_prev, lag = s[:, :nj].clone(), torch.zeros(envs, dtype=torch.bool, device=dev)

    est = None
    if "est_state" in ck:
      est = StateEstimator(*ck["est_dims"], hidden=ck["est_hidden"]).to(dev).eval()
      est.load_state_dict(ck["est_state"])

    def act(_obs):
      x = task.flatten(hist)
      if est is not None:
        x = torch.cat([x, est(x)], -1)
      if priv:
        x = torch.cat([x, task.priv(s, info, ctx)], -1)
      return actor(x)
  else:
    agent_cfg = load_rl_cfg(args.task)
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(wenv, prepare_agent_cfg(agent_cfg), device=dev)
    if "student_state_dict" in ck:
      runner.alg.actor.load_state_dict(ck["student_state_dict"])
    else:
      runner.load(args.policy, map_location=dev)
    pol = runner.get_inference_policy(device=dev)
    obs = wenv.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs

    def act(o):
      return pol(o)

  mu = M.pad_friction(env)[:, 0].clone()  # this rollout's pad friction (counting stops at the first drop)
  steps = int(args.seconds / env.step_dt)
  failed = torch.zeros(envs, dtype=torch.bool, device=dev)
  yaw = torch.zeros(envs, device=dev)
  live_steps = torch.zeros(envs, device=dev)
  all4 = torch.zeros(envs, device=dev)
  low = torch.zeros(envs, device=dev)
  idle = torch.zeros(envs, 4, device=dev)
  capped = torch.zeros(envs, device=dev)  # rotation the rotate reward pays for: per-step rate clipped to +-0.3 rad/s
  t_fail = torch.full((envs,), -1.0, device=dev)  # time of the first drop (s), -1 if none
  # grip (how the pads hold, per live step, from the per-contact pad sensor): total pad normal force, pads pressing,
  # friction-cone use = total friction / (mu * total normal), the share of steps with some pressing contact at > 90% of
  # its cone (about to slide), and the share of the pads' vertical (anti-gravity) force carried by normal force
  g_fn = torch.zeros(envs, device=dev)
  g_np = torch.zeros(envs, device=dev)
  g_cu = torch.zeros(envs, device=dev)
  g_cn = torch.zeros(envs, device=dev)
  g_sl = torch.zeros(envs, device=dev)
  g_nz = torch.zeros(envs, device=dev)
  g_dr = torch.zeros(envs, device=dev)  # drift from the reset position (mm), summed over live steps
  g_d15 = torch.zeros(envs, device=dev)  # live steps with drift > 15 mm
  # the 3 s before each first drop: pads pressing, total pad normal force, cone use, pads_low, object height change and
  # drift from the reset position (mm; the episode ends at 30 mm drift)
  TR = int(round(3.0 / env.step_dt))
  ring = torch.full((TR, envs, 6), float("nan"), device=dev)
  traces = torch.full((envs, TR, 6), float("nan"), device=dev)
  z0 = env.scene["cube"].data.root_link_pos_w[:, 2].clone()
  yaw_prev = st.yaw.clone()
  jt = env.action_manager.get_term("joint_pos")
  jn = list(env.scene["robot"].joint_names)
  sq_ids = [i for i, n in enumerate([jn[j] for j in jt._target_ids.tolist()]) if n in ("mf_pip", "rf_pip", "th_mcp")]
  squeeze = []  # start squeeze target - q (physical; the encoder bias cancels) on the deploy diagnostic's joints
  dump = {k: [] for k in ("x", "drift", "pos", "live", "t")}
  rec = None
  if args.record:
    from in_hand_rotation_mjlab.tasks.fingertip_rotate.objects import MIXED_GEOM_ORDER
    gid = [env.sim.mj_model.geom(f"cube/{g}").id for g in MIXED_GEOM_ORDER]
    rec = dict(qpos=[env.sim.data.qpos.float().cpu()], yaw=[st.yaw.cpu()], pads=[M.pad_forces_w(env).norm(dim=-1).cpu()],
               geom_ids=gid, geom_size=env.sim.model.geom_size[:, gid].cpu(), mu=M.pad_friction(env)[:, 0].cpu(),
               obj=[names[e % n_obj] for e in range(envs)], step_dt=env.step_dt, policy=args.policy, task=args.task,
               mocap_pos=env.sim.data.mocap_pos.float().cpu())  # per env: the hand base, offset by the env's origin
  t0 = time.time()
  with torch.no_grad():
    for t in range(steps):
      if wm_actor:
        if args.dump and t % args.dump_every == 0:
          dump["x"].append(task.flatten(hist).half().cpu())
          dump["drift"].append((M.obj_pos_hand(env, st) - st.p0).cpu())
          dump["pos"].append(M.obj_pos_hand(env, st).cpu())
          dump["live"].append((~failed).cpu())
          dump["t"].append(t)
        a = act(None)
        _, _, term, trunc, _ = env.step(a)
        d = (term | trunc).bool()
        s_new = R.read_state(env, L)
        q_prev = torch.where(d[:, None], s_new[:, :nj], s[:, :nj])
        s = s_new
        if d.any():
          ids = d.nonzero().squeeze(-1)
          f2, _ = R.context_full(env)
          ctx[ids] = f2[ids][:, cols]
          info[ids] = torch.cat([s[ids, :nj], s[ids][:, task.S["p"]], st.extent[ids, None]], -1)
        lag = torch.where(torch.rand(envs, device=dev) < 0.01, torch.rand(envs, device=dev) < 0.5, lag)
        fr = task.obs_frame(torch.where(lag[:, None], q_prev, s[:, :nj]), s[:, task.S["cmd"]], ctx, noise=False)
        hist = torch.cat([hist[:, 1:], fr[:, None]], 1)
        if d.any():
          hist[ids] = fr[ids, None].expand(-1, H_OBS, -1)
      else:
        a = act(obs)
        obs, _, d, _ = wenv.step(a)
        d = d.bool()
      live = ~failed & ~d
      if t < 40:
        rq = env.scene["robot"].data.joint_pos[:, jt._target_ids]
        squeeze.append((jt._target - env.scene["robot"].data.encoder_bias[:, jt._target_ids] - rq)[:, sq_ids].mean(-1))
      if rec is not None:
        rec["qpos"].append(env.sim.data.qpos.float().cpu())
        rec["yaw"].append(torch.where(live, st.yaw, yaw).cpu())
        rec["pads"].append(M.pad_forces_w(env).norm(dim=-1).cpu())
      capped += live.float() * ((st.yaw - yaw_prev) / env.step_dt).clamp(-0.3, 0.3) * env.step_dt
      yaw_prev = st.yaw.clone()
      yaw = torch.where(live, st.yaw, yaw)
      t_fail = torch.where(d & ~failed, torch.full_like(t_fail, (t + 1) * env.step_dt), t_fail)
      failed |= d
      lv = (~failed).float()
      on = M.pad_forces_w(env).norm(dim=-1) > 0.1
      idle = torch.where(on, torch.zeros_like(idle), idle + env.step_dt)
      live_steps += lv
      all4 += lv * (idle <= 1.0).all(-1).float()
      low += lv * M.pads_low(env, "down")
      cd = env.scene.sensors["pad_contact_cf"].data
      f, nv, tv = cd.force, cd.normal, cd.tangent  # (B, N, 3)
      fn, fs = f[..., 0], f[..., 1:].norm(dim=-1)
      pr = (cd.found > 0) & (fn > 0.05)
      fn, fs = fn * pr, fs * pr
      t2 = torch.cross(nv, tv, dim=-1)
      vz = fn * nv[..., 2] + f[..., 1] * pr * tv[..., 2] + f[..., 2] * pr * t2[..., 2]  # vertical force, per contact
      g_fn += lv * fn.sum(-1)
      g_np += lv * on.sum(-1)
      hp = lv * pr.any(-1).float()
      g_cu += hp * fs.sum(-1) / (mu * fn.sum(-1)).clamp(min=1e-6)
      g_cn += hp
      g_sl += lv * (pr & (fs > 0.9 * mu[:, None] * fn)).any(-1).float()
      g_nz += hp * ((fn * nv[..., 2]).sum(-1) / vz.sum(-1).abs().clamp(min=1e-3)).clamp(-2, 2)
      new_fail = d & (t_fail == (t + 1) * env.step_dt)
      if new_fail.any():  # the ring holds steps up to t-1 (this step's readings are already the reset episode's)
        k = torch.arange(TR, device=dev)
        order = (t + k) % TR  # oldest first
        traces[new_fail] = ring[order][:, new_fail].transpose(0, 1)
      zc = env.scene["cube"].data.root_link_pos_w[:, 2]
      z0 = torch.where(d, zc, z0)
      dr_mm = (M.obj_pos_hand(env, st) - st.p0).norm(dim=-1) * 1000.0
      g_dr += lv * dr_mm
      g_d15 += lv * (dr_mm > 15.0).float()
      ring[t % TR] = torch.stack([on.sum(-1).float(), fn.sum(-1), fs.sum(-1) / (mu * fn.sum(-1)).clamp(min=1e-6),
                                  M.pads_low(env, "down"), (zc - z0) * 1000.0,
                                  dr_mm], -1)
  ls = live_steps.clamp(min=1)
  per_obj = {}
  for i, n in enumerate(names):
    sel = torch.arange(i, envs, n_obj, device=dev)
    per_obj[n] = dict(shape=OBJECTS[n].shape, yaw_deg=float(yaw[sel].mean() * DEG), dropped=float(failed[sel].float().mean()),
                      capped_yaw_deg=float(capped[sel].mean() * DEG),
                      all_four=float((all4[sel] / ls[sel]).mean()), pads_low=float((low[sel] / ls[sel]).mean()))
  shapes = {}
  for sh in ("sphere", "cube", "cylinder", "cuboid"):
    v = [x for x in per_obj.values() if x["shape"] == sh]
    shapes[sh] = {k: float(np.mean([x[k] for x in v])) for k in ("yaw_deg", "dropped", "capped_yaw_deg", "all_four", "pads_low")}
  # by pad friction (the real pads sit near 0.3-0.4): all objects and the non-spheres
  sph = torch.tensor([OBJECTS[names[e % n_obj]].shape == "sphere" for e in range(envs)], device=dev)
  edges = (0.0, 0.4, 0.55, 0.7, 0.85, 9.0)
  by_mu = {}
  for lo, hi in zip(edges[:-1], edges[1:]):
    b = (mu >= lo) & (mu < hi)
    nb = b & ~sph
    by_mu[f"{lo:.2f}-{hi:.2f}"] = dict(n=int(b.sum()), yaw_deg=float(yaw[b].mean() * DEG) if b.any() else None,
                                       dropped=float(failed[b].float().mean()) if b.any() else None,
                                       nonsphere_dropped=float(failed[nb].float().mean()) if nb.any() else None)
    tf = t_fail[b & failed]
    if len(tf):  # when the drops happen: early ones suggest the friction is not yet identified, late ones the gait
      by_mu[f"{lo:.2f}-{hi:.2f}"]["drop_t_s"] = [float(x) for x in torch.quantile(tf, torch.tensor([0.25, 0.5, 0.75], device=dev))]
      by_mu[f"{lo:.2f}-{hi:.2f}"]["drop_in_first_5s"] = float((tf <= 5.0).float().mean())
  shape_mu = {}  # dropped share per shape, low (< 0.4) vs high (>= 0.55) pad friction
  shp = [OBJECTS[names[e % n_obj]].shape for e in range(envs)]
  for sh in ("sphere", "cube", "cylinder", "cuboid"):
    m = torch.tensor([x == sh for x in shp], device=dev)
    shape_mu[sh] = {bn: (float(failed[m & bm].float().mean()) if (m & bm).any() else None)
                    for bn, bm in (("lo", mu < 0.4), ("hi", mu >= 0.55))}
  grip = {}
  for gname, gsh in (("prism", ("cube", "cuboid")), ("round", ("sphere", "cylinder"))):
    m = torch.tensor([x in gsh for x in shp], device=dev)
    for bn, bm in (("lo", mu < 0.4), ("hi", mu >= 0.55)):
      e = m & bm
      if e.any():
        grip[f"{gname}_{bn}"] = dict(n=int(e.sum()), normal_N=float((g_fn[e] / ls[e]).mean()), pads_pressing=float((g_np[e] / ls[e]).mean()),
                                     cone_use=float((g_cu[e] / g_cn[e].clamp(min=1)).mean()), near_slip=float((g_sl[e] / ls[e]).mean()),
                                     weight_by_normal=float((g_nz[e] / g_cn[e].clamp(min=1)).mean()),
                                     drift_mm=float((g_dr[e] / ls[e]).mean()), drift_over_15mm=float((g_d15[e] / ls[e]).mean()),
                                     pads_low=float((low[e] / ls[e]).mean()), dropped=float(failed[e].float().mean()))
  drop_trace = {}
  ts = [-3.0 + (i + 1) * env.step_dt for i in range(TR)]
  pick = list(range(4, TR, 5))  # every 0.25 s
  for gname, gsh in (("prism", ("cube", "cuboid")), ("round", ("sphere", "cylinder"))):
    m = torch.tensor([x in gsh for x in shp], device=dev)
    for bn, bm in (("lo", mu < 0.4), ("hi", mu >= 0.55)):
      e = m & bm & failed & (t_fail > 0)
      if e.any():
        tr = traces[e]  # (n, TR, 5)
        mean = torch.nanmean(tr, 0)
        drop_trace[f"{gname}_{bn}"] = dict(n=int(e.sum()), t_s=[round(ts[i], 2) for i in pick],
                                           **{f: [float(mean[i, j]) for i in pick]
                                              for j, f in enumerate(("pads", "normal_N", "cone_use", "pads_low", "dz_mm", "drift_mm"))})
  res = dict(policy=args.policy, task=args.task, seconds=args.seconds, per_obj_rollouts=args.per_obj,
             yaw_deg=float(yaw.mean() * DEG), dropped=float(failed.float().mean()), by_shape=shapes, by_object=per_obj,
             capped_yaw_deg=float(capped.mean() * DEG),
             by_pad_friction=by_mu, pad_friction_range=[float(mu.min()), float(mu.max())], by_shape_pad_friction=shape_mu, grip=grip, drop_trace=drop_trace,
             meta={k: ck.get(k) for k in ("update", "env_steps", "wall_s")} if wm_actor else {}, eval_s=time.time() - t0,
             start_squeeze=dict(t_s=[0.05, 0.5, 1.0, 2.0], rad=[float(torch.stack(squeeze)[min(i, len(squeeze) - 1)].mean())
                                                                for i in (0, 9, 19, 39)], scale=args.start_squeeze))
  if args.dump and dump["x"]:
    np.savez_compressed(args.dump, x=torch.stack(dump["x"]).numpy(), drift=torch.stack(dump["drift"]).numpy(),
                        pos=torch.stack(dump["pos"]).numpy(), live=torch.stack(dump["live"]).numpy(), t=np.array(dump["t"]),
                        mu=mu.cpu().numpy(), shape=np.array(shp), frame_dim=task.frame_dim, step_dt=env.step_dt)
  if rec is not None:
    rec.update(qpos=torch.stack(rec["qpos"]), yaw=torch.stack(rec["yaw"]), pads=torch.stack(rec["pads"]), t_fail=t_fail.cpu(),
               yaw_final=yaw.cpu(), failed=failed.cpu())
    torch.save(rec, args.record)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=1))
  print(f"[eval_ftrot] {Path(args.policy).name}: yaw {res['yaw_deg']:.0f} deg/{args.seconds:.0f}s, dropped "
        f"{100 * res['dropped']:.1f}%  | " + "  ".join(f"{k} {v['yaw_deg']:.0f}deg/{100 * v['dropped']:.0f}%"
                                                         for k, v in shapes.items()))
  print(f"[eval_ftrot] {Path(args.policy).name} start squeeze (target - q, mf_pip/rf_pip/th_mcp) at 0.05/0.5/1/2 s: "
        + "/".join(f"{x:.3f}" for x in res["start_squeeze"]["rad"]) + f" rad (start squeeze x{args.start_squeeze})")
  print(f"[eval_ftrot] {Path(args.policy).name} rewarded (0.3 rad/s-capped) yaw {res['capped_yaw_deg']:.0f} deg/{args.seconds:.0f}s"
        f" of max {0.3 * args.seconds * DEG:.0f}: " + "  ".join(f"{k} {v['capped_yaw_deg']:.0f}" for k, v in shapes.items()))
  print(f"[eval_ftrot] {Path(args.policy).name} by pad mu (n, dropped all/non-sphere): " + "  ".join(
    f"{k} {v['n']} {100 * v['dropped']:.0f}/{100 * (v['nonsphere_dropped'] or 0):.0f}%" for k, v in by_mu.items() if v["n"]))
  print(f"[eval_ftrot] {Path(args.policy).name} drop time median s by pad mu: " + "  ".join(
    f"{k} {v['drop_t_s'][1]:.0f} ({100 * v['drop_in_first_5s']:.0f}% <5s)" for k, v in by_mu.items() if "drop_t_s" in v))
  print(f"[eval_ftrot] {Path(args.policy).name} grip (normal N / pads / cone use / near-slip / weight by normal / low / drift mm / >15 mm / dropped): " + "  ".join(
    f"{k} {v['normal_N']:.2f}/{v['pads_pressing']:.1f}/{v['cone_use']:.2f}/{v['near_slip']:.2f}/{v['weight_by_normal']:.2f}/"
    f"{v['pads_low']:.2f}/{v['drift_mm']:.1f}/{v['drift_over_15mm']:.2f}/{100 * v['dropped']:.0f}%"
    for k, v in grip.items()))
  for k, v in drop_trace.items():  # -3, -1, -0.5, -0.25 s and the last step before the drop
    ix = [0, 7, 9, 10, 11]
    print(f"[eval_ftrot] {Path(args.policy).name} before drop {k} (n {v['n']}; t -3/-1/-.5/-.25/-.05 s): " + "  ".join(
      f"{f} " + "/".join(f"{v[f][i]:.2f}" for i in ix) for f in ("pads", "normal_N", "cone_use", "pads_low", "dz_mm", "drift_mm")))


if __name__ == "__main__":
  main()
