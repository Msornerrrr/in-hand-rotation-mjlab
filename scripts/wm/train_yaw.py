"""SuperMimic-style policy learning (arXiv 2610.28480) on the palm-up cube-yaw task: a deterministic actor trained by
backpropagating imagined returns through a world model that is refit online on the actor's own data.

One loop iteration = one vectorised env step of the actor (+ N(0, sigma^2) action noise) on the training env (all of
its randomization), written to a GPU replay ring. Every 1/UTD env steps one update:
  1. world model: ``--wm-steps`` Adam steps on k-step open-loop windows (+ termination BCE, + the learned
     torque / work penalty head); commands rebuilt from the predicted joints;
  2. actor: ``--starts`` imagination starts from the ring (each with its observation history, previous yaw
     differences, episode info and context), H imagined steps, J = sum gamma^t c_t r_t + gamma^H c_H V(s_H) with the
     survival weight c from the termination head, gradient clipped by norm 1 then by value 0.1;
  3. critic: TD(lambda) targets on the detached imagined trajectory.
The first ``--wm-only`` updates fit the world model alone. Actor checkpoints go to <out>/ckpt every few minutes for
``eval_yaw.py``; <out>/log.jsonl records losses and the real (collected) reward next to the imagined one.

  MUJOCO_GL=egl uv run python scripts/wm/train_yaw.py --out outputs/wm/yaw
  ... --parity --collect-policy <ppo model.pt>   # check the reward / observation adapter against the env, then exit
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import in_hand_rotation_mjlab.tasks  # noqa: F401
from in_hand_rotation_mjlab.world_model import record as R
from in_hand_rotation_mjlab.world_model.gear_yaw import H_OBS, GearYaw
from in_hand_rotation_mjlab.world_model.model import CommandMap, StateError, WMConfig, WorldModel, mlp
from in_hand_rotation_mjlab.world_model.policy import Actor, Critic, RunningNorm
from in_hand_rotation_mjlab.world_model.replay import Replay
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear"
REW_NAMES = ("rotate_finite_diff", "object_linvel_penalty", "pose_diff_penalty", "torque_penalty", "work_penalty",
             "object_fallen")


def actor_obs(o):
  return o["actor"] if not torch.is_tensor(o) else o


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--envs", type=int, default=4096)
  ap.add_argument("--replay", type=int, default=1000, help="steps per env kept")
  ap.add_argument("--hours", type=float, default=3.0, help="wall-clock budget")
  ap.add_argument("--horizon", type=int, default=16)
  ap.add_argument("--starts", type=int, default=4096)
  ap.add_argument("--wm-steps", type=int, default=10)
  ap.add_argument("--wm-batch", type=int, default=2048)
  ap.add_argument("--wm-k", type=int, default=8)
  ap.add_argument("--wm-hidden", default="1024,1024,1024")
  ap.add_argument("--wm-init", default=None, help="start from a fitted world model (wm.pt of train_wm.py)")
  ap.add_argument("--wm-only", type=int, default=300)
  ap.add_argument("--utd", type=float, default=0.4, help="updates per env step")
  ap.add_argument("--noise", type=float, default=0.1, help="collection action noise (added after the squash)")
  ap.add_argument("--imag-noise", type=float, default=0.0,
                  help="reparameterized pre-squash Gaussian on the imagined actions (randomized smoothing)")
  ap.add_argument("--imag-noise-final", type=float, default=None, help="linear decay of --imag-noise to this")
  ap.add_argument("--imag-noise-updates", type=int, default=6000)
  ap.add_argument("--gamma", type=float, default=0.99)
  ap.add_argument("--lam", type=float, default=0.95)
  ap.add_argument("--lr", type=float, default=2e-4)
  ap.add_argument("--warmup", type=int, default=64, help="env steps before the first update")
  ap.add_argument("--warmup-noise", type=float, default=0.5,
                  help="action noise of the warm-up steps, whose data sets the model's normalization (no expert data)")
  ap.add_argument("--ckpt-min", type=float, default=5.0)
  ap.add_argument("--log-every", type=int, default=50)
  ap.add_argument("--torque", choices=["analytic", "learned"], default="analytic",
                  help="torque / work penalties: the servo law on predicted states, or a learned head")
  ap.add_argument("--parity", action="store_true")
  ap.add_argument("--collect-policy", default=None, help="parity: collect with this rsl_rl checkpoint")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  dev = args.device
  out = Path(args.out)
  (out / "ckpt").mkdir(parents=True, exist_ok=True)
  (out / "args.json").write_text(json.dumps(vars(args), indent=1))

  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.scene.num_envs = args.envs
  env_cfg.seed = args.seed
  env_cfg.recorders = {"wm_terminal": RecorderTermCfg(func=R.TerminalRecorder)}
  env = ManagerBasedRlEnv(cfg=env_cfg, device=dev)
  wenv = RslRlVecEnvWrapper(env, clip_actions=1.0)
  L = R.layout(env)
  rec = env.recorder_manager.get_term("wm_terminal")
  rec.L = L
  nj, N, dt = L.nj, args.envs, env.step_dt
  robot = env.scene["robot"]
  task = GearYaw(nj, robot.data.default_joint_pos[0, L.term.target_ids].clone(), L.base_p[0].clone(),
                 L.base_q[0].clone(), dt)
  assert tuple(env.reward_manager._term_names) == REW_NAMES, env.reward_manager._term_names
  full, names = R.context_full(env)
  cidx, cnames = R.varying_columns(full, names)
  meta = dict(delta_min=L.term._delta_min.tolist(), delta_max=L.term._delta_max.tolist(),
              max_lead=None if L.term._max_lead is None else L.term._max_lead.tolist(),
              soft_limits=robot.data.soft_joint_pos_limits[0][L.term.target_ids].tolist(),
              clip_to_joint_limits=bool(L.term.cfg.clip_to_joint_limits), step_dt=dt, nj=nj)
  cmap = CommandMap(meta).to(dev)

  collect_policy = None
  if args.collect_policy:
    from rsl_rl.runners import OnPolicyRunner

    from in_hand_rotation_mjlab.utils.rl import prepare_agent_cfg
    from mjlab.tasks.registry import load_rl_cfg

    runner = OnPolicyRunner(wenv, prepare_agent_cfg(load_rl_cfg(TASK)), device=dev)
    runner.load(args.collect_policy, map_location=dev)
    collect_policy = runner.get_inference_policy(device=dev)

  actor = Actor(task.obs_dim, nj).to(dev)
  from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import LeapGearServoActuator

  servo = next(a for a in robot.actuators if isinstance(a, LeapGearServoActuator))
  assert list(servo._target_names) == list(L.term._target_names)
  plant_now = lambda: torch.cat([servo.current_gain, servo.current_damping, servo.current_limit,  # noqa: E731
                                 servo.torque_constant], -1).clone()
  replay = Replay(args.replay, N, L.dim, nj, len(cidx), 2 * nj, task.info_dim, len(REW_NAMES), dev, d_plant=4 * nj)

  # --- collection state ---
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  cur_s = R.read_state(env, L)
  ob = actor_obs(obs)
  newest = lambda o: torch.cat([o[:, (H_OBS - 1) * nj:H_OBS * nj], o[:, (2 * H_OBS - 1) * nj:]], -1)  # noqa: E731
  cur_frame = newest(ob)
  cur_epi = torch.arange(N, device=dev)
  n_epi = N
  cur_ctx = full[:, cidx].clone()
  cur_info = torch.zeros(N, task.info_dim, device=dev)
  cur_info[:, :nj] = cur_s[:, :nj]
  cur_ok = torch.zeros(N, dtype=torch.bool, device=dev)
  cur_step = env.episode_length_buf.clone().long()
  cur_plant = plant_now()
  ep_drops = 0

  noise = args.noise

  def env_step():
    nonlocal obs, cur_s, cur_frame, n_epi, cur_step, ep_drops, cur_plant
    with torch.no_grad():
      ob = actor_obs(obs)
      if collect_policy is not None:
        a = collect_policy(obs)
      else:
        actor.norm.update(ob)
        a = actor(ob)
      a = (a + noise * torch.randn_like(a)).clamp(-1.0, 1.0)
      rec.clear()
      obs2, _, d, _ = wenv.step(a)
      d = d.bool()
      s_post = R.read_state(env, L)
      s_true = s_post.clone()
      if d.any():
        s_true[rec.ids] = rec.rows
      done = torch.where(d, torch.where(env.reset_time_outs.bool(), 2, 1), 0).to(torch.uint8)
      ep_drops += int((done == 1).sum())
      replay.add(s=cur_s, s_next=s_true, a=a, ctx=cur_ctx, frame=cur_frame, info=cur_info, info_ok=cur_ok, plant=cur_plant,
                 rew=env.reward_manager._step_reward.clone(), done=done, epi=cur_epi, ep_step=cur_step)
      cur_s = s_post
      obs = obs2
      cur_frame = newest(actor_obs(obs2))
      cur_step = env.episode_length_buf.clone().long()
      first = (cur_step == 1) & ~d  # the yaw reward anchors on the state after the episode's first step
      if first.any():
        cur_info[first, nj:] = task.anchor(s_post[first])
        cur_ok[first] = True
      if d.any():
        ids = d.nonzero().squeeze(-1)
        cur_epi[ids] = torch.arange(n_epi, n_epi + len(ids), device=dev)
        n_epi += len(ids)
        f2, _ = R.context_full(env)
        cur_ctx[ids] = f2[ids][:, cidx]
        cur_info[ids, :nj] = s_post[ids, :nj]
        cur_ok[ids] = False
        cur_plant = plant_now()

  if args.parity:
    for _ in range(args.warmup):
      env_step()
    parity(replay, task, nj, dt, out)
    return

  # --- models ---
  wcfg = WMConfig(nj=nj, dc=len(cidx), hidden=tuple(int(h) for h in args.wm_hidden.split(",")), dt=dt)
  wm = WorldModel(wcfg).to(dev)
  err = StateError(nj).to(dev)
  c_mu = torch.zeros(len(cidx), device=dev)
  c_sd = torch.ones(len(cidx), device=dev)
  aux = mlp(2 * nj + len(cidx), (256, 256), 1, layer_norm=True).to(dev)
  aux_sd = torch.ones((), device=dev)
  critic_in = task.critic_features(cur_s[:2], cur_info[:2], torch.zeros(2, 3, device=dev), cur_ctx[:2]).shape[-1]
  critic = Critic(critic_in).to(dev)
  opt_wm = torch.optim.Adam(list(wm.parameters()) + list(aux.parameters()), lr=args.lr)
  opt_pi = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(0.9, 0.95))
  opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
  r_max = 0.4 * dt  # most the reward can pay per step (rotation 1.25 x 0.25 /s), for the value-target clamp

  noise = args.warmup_noise
  for _ in range(args.warmup):
    env_step()
  noise = args.noise
  # normalization from the warm-up data (fixed afterwards, as the reproductions do)
  with torch.no_grad():
    G = replay.G
    rows = torch.arange(G, device=dev) % replay.L
    s = replay.s[rows].reshape(-1, L.dim)
    sn = replay.s_next[rows].reshape(-1, L.dim)
    cx = replay.ctx[rows].reshape(-1, len(cidx))
    c_mu.copy_(cx.mean(0))
    c_sd.copy_(cx.std(0).clamp_min(1e-6))
    if args.wm_init:
      ck = torch.load(args.wm_init, map_location=dev, weights_only=False)
      assert ck["ctx_names"] == cnames, "the pre-fitted model has other context columns"
      wm.load_state_dict(ck["state_dict"])
      c_mu.copy_(ck["c_mu"].to(dev))
      c_sd.copy_(ck["c_sd"].to(dev))
    else:
      wm.fit_stats(s, sn[:, wm.S["cmd"]], (cx - c_mu) / c_sd, sn)
    err.fit(s, sn)
    aux_t = replay.rew[rows][..., 3:5].sum(-1).reshape(-1)
    aux_sd.copy_(aux_t.std().clamp_min(1e-6))
  norm_c = lambda c: (c - c_mu) / c_sd  # noqa: E731

  log = (out / "log.jsonl").open("a")
  t0 = time.time()
  last_ckpt = t0
  n_up, acc, hist = 0, 0.0, []
  last_G, last_drops, last_rew = replay.G, ep_drops, None

  def set_grad(mods, flag):
    for m in mods:
      for p in m.parameters():
        p.requires_grad_(flag)

  while time.time() - t0 < args.hours * 3600:
    env_step()
    acc += args.utd
    while acc >= 1.0:
      acc -= 1.0
      n_up += 1
      st = {}
      # 1. world model
      for _ in range(args.wm_steps):
        g, e = replay.sample_windows(args.wm_batch, args.wm_k)
        s0, acts, snx, cx, dn = replay.window(g, e, args.wm_k)
        c = norm_c(cx)
        sh, ld = s0, 0.0
        for j in range(args.wm_k):
          sh, _ = wm(sh, cmap(sh[..., wm.S["cmd"]], acts[j], sh[..., wm.S["q"]]), c)
          ld = ld + err(sh, snx[j]).mean()
        ld = ld / args.wm_k
        with torch.no_grad():
          from in_hand_rotation_mjlab.world_model.model import geodesic, quat_conj, quat_mul, twist_z
          Sx = wm.S
          q0c = quat_conj(s0[..., Sx["quat"]])
          tw = twist_z(quat_mul(sh[..., Sx["quat"]], q0c)) - twist_z(quat_mul(snx[-1][..., Sx["quat"]], q0c))
          tw = torch.remainder(tw + math.pi, 2 * math.pi) - math.pi
          acc_tw = float(tw.abs().mean() * 180 / math.pi)
          acc_pos = float((sh[..., Sx["p"]] - snx[-1][..., Sx["p"]]).norm(dim=-1).mean() * 1e3)
        # one-step: termination head (terminated rows oversampled to a quarter) + torque / work head
        term = (replay.done == 1).nonzero()
        nt = min(len(term), args.wm_batch // 4)
        rr = torch.randint(0, replay.L if replay.G >= replay.L else replay.G, (args.wm_batch - nt,), device=dev)
        ee = torch.randint(0, N, (args.wm_batch - nt,), device=dev)
        if nt:
          pick = term[torch.randint(len(term), (nt,), device=dev)]
          rr, ee = torch.cat([rr, pick[:, 0]]), torch.cat([ee, pick[:, 1]])
        s1, a1, n1, c1 = replay.s[rr, ee], replay.a[rr, ee], replay.s_next[rr, ee], norm_c(replay.ctx[rr, ee])
        cmd1 = cmap(s1[..., wm.S["cmd"]], a1, s1[..., wm.S["q"]])
        with torch.no_grad():
          p1, _ = wm(s1, cmd1, c1)
        lt = F.binary_cross_entropy_with_logits(wm.term((wm._term_feat(p1) - wm.t_mu) / wm.t_sd).squeeze(-1),
                                                (replay.done[rr, ee] == 1).float())
        la = (aux(task.aux_features(n1, cmd1, c1)).squeeze(-1) - replay.rew[rr, ee][:, 3:5].sum(-1) / aux_sd).pow(2).mean()
        opt_wm.zero_grad(set_to_none=True)
        (ld + lt + la).backward()
        torch.nn.utils.clip_grad_norm_(list(wm.parameters()) + list(aux.parameters()), 1.0)
        opt_wm.step()
      st.update(wm=float(ld), term=float(lt), aux=float(la), wm_twist_deg=acc_tw, wm_pos_mm=acc_pos)
      if n_up <= args.wm_only:
        hist.append(st)
        continue
      # 2. actor through imagination
      fin = args.imag_noise if args.imag_noise_final is None else args.imag_noise_final
      frac = min(1.0, max(0, n_up - args.wm_only) / max(1, args.imag_noise_updates))
      imag_sigma = args.imag_noise + (fin - args.imag_noise) * frac
      set_grad([wm, aux], False)
      g, e = replay.sample_starts(args.starts, H_OBS)
      r0 = g % replay.L
      s, info, ctx, plant = replay.s[r0, e], replay.info[r0, e], norm_c(replay.ctx[r0, e]), replay.plant[r0, e]
      frames = replay.history("frame", g, e, H_OBS)
      sh4 = replay.history("s", g, e, 4)
      dprev = torch.stack([task.dyaw(sh4[:, i], sh4[:, i + 1]) for i in range(3)], -1)
      J, surv, disc = 0.0, torch.ones(len(g), device=dev), 1.0
      feats, rews, conts = [], [], []
      for h in range(args.horizon):
        feats.append(task.critic_features(s, info, dprev, ctx).detach())
        a = actor(task.flatten(frames), noise=imag_sigma)
        cmd = cmap(s[..., wm.S["cmd"]], a, s[..., wm.S["q"]])
        s2, logit = wm(s, cmd, ctx)
        if args.torque == "analytic":
          aux_rate = task.torque_work_rate(s2, cmd, plant)
        else:
          aux_rate = aux(task.aux_features(s2, cmd, ctx)).squeeze(-1) * aux_sd
        terms, dprev = task.reward_rates(s, s2, info, dprev, aux_rate)
        r = sum(terms.values()) * dt
        p = torch.sigmoid(logit)
        J = J + disc * surv * r
        rews.append(r.detach()); conts.append((1 - p).detach())
        surv = surv * (1 - p)
        disc *= args.gamma
        lag = torch.rand(len(g), device=dev) < 0.5
        q_obs = torch.where(lag[:, None], s[..., wm.S["q"]], s2[..., wm.S["q"]])
        frames = torch.cat([frames[:, 1:], task.obs_frame(q_obs, cmd)[:, None]], 1)
        s = s2
      fH = task.critic_features(s, info, dprev, ctx)
      J = J + disc * surv * critic(fH)
      loss_pi = -J.mean()
      opt_pi.zero_grad(set_to_none=True)
      loss_pi.backward()
      gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
      torch.nn.utils.clip_grad_value_(actor.parameters(), 0.1)
      ok = all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
      if ok:
        opt_pi.step()
      set_grad([wm, aux], True)
      # 3. critic: TD(lambda) on the detached rollout
      with torch.no_grad():
        feats.append(fH.detach())
        X = torch.stack(feats)  # (H+1, B, d)
        critic.norm.update(X)
        V = critic(X)
        Rt = V[-1]
        tgt = []
        for h in reversed(range(args.horizon)):
          Rt = rews[h] + args.gamma * conts[h] * ((1 - args.lam) * V[h + 1] + args.lam * Rt)
          tgt.append(Rt)
        T = torch.stack(tgt[::-1]).clamp(-r_max / (1 - args.gamma), r_max / (1 - args.gamma))
      Xf, Tf = X[:-1].reshape(-1, X.shape[-1]), T.reshape(-1)
      lv = 0.0
      for _ in range(5):
        perm = torch.randperm(len(Xf), device=dev)
        for mb in perm.chunk(4):
          l = (critic(Xf[mb]) - Tf[mb]).pow(2).mean()
          opt_v.zero_grad(set_to_none=True)
          l.backward()
          torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
          opt_v.step()
          lv = float(l)
      st.update(J=float(J.mean()), imag_r=float(torch.stack(rews).mean() / dt), v=lv, gn=float(gn),
                surv=float(surv.mean()), finite=ok)
      hist.append(st)
    if n_up and n_up % args.log_every == 0 and hist:
      G = replay.G
      k = min(G - last_G, replay.L)
      rows = torch.arange(G - k, G, device=dev) % replay.L
      real = replay.rew[rows].mean((0, 1))
      wall = time.time() - t0
      row = dict(update=n_up, env_steps=G * N, wall_s=wall, s_per_update=wall / max(1, n_up),
                 real_reward_per_s=float(real.sum()), real_rotate=float(real[0]),
                 real_drops_per_env_min=(ep_drops - last_drops) / N / max(1e-9, (G - last_G) * dt / 60),
                 **{k2: sum(h[k2] for h in hist if k2 in h) / max(1, sum(k2 in h for h in hist))
                    for k2 in ("wm", "term", "aux", "J", "imag_r", "v", "gn", "surv", "wm_twist_deg", "wm_pos_mm")})
      log.write(json.dumps(row) + "\n"); log.flush()
      print(f"[train_policy] up {n_up} steps {G * N / 1e6:.1f}M {wall / 60:.1f} min  real r/s {row['real_reward_per_s']:.3f} "
            f"rot {row['real_rotate']:.3f} drops/min {row['real_drops_per_env_min']:.3f}  imag r/s {row.get('imag_r', 0):.3f} "
            f"{row['s_per_update']:.2f} s/up  wm {row['wm']:.2f} (twist@k {row['wm_twist_deg']:.2f} deg) term {row['term']:.3f} v {row.get('v', 0):.2e} surv {row.get('surv', 0):.3f}", flush=True)
      hist, last_G, last_drops = [], G, ep_drops
    if time.time() - last_ckpt > args.ckpt_min * 60 and n_up > args.wm_only:
      last_ckpt = time.time()
      path = out / "ckpt" / f"actor_{n_up:06d}.pt"
      torch.save(actor.checkpoint(update=n_up, env_steps=replay.G * N, wall_s=time.time() - t0), path)
  torch.save(actor.checkpoint(update=n_up, env_steps=replay.G * N, wall_s=time.time() - t0), out / "ckpt" / "actor_final.pt")
  torch.save(dict(wm=wm.state_dict(), cfg=wcfg.__dict__, critic=critic.state_dict(), aux=aux.state_dict(),
                  ctx_names=cnames, c_mu=c_mu, c_sd=c_sd, meta=meta), out / "models_final.pt")
  print("[train_policy] done")


@torch.no_grad()
def parity(replay: Replay, task: GearYaw, nj: int, dt: float, out: Path):
  """Analytic reward terms from recorded states vs the env's own, on rows with a full yaw window and known info."""
  G = replay.G
  g = torch.arange(4, G, device=replay.device)
  rows = g % replay.L
  res = {}
  e = torch.arange(replay.N, device=replay.device)
  gg = g[:, None].expand(-1, replay.N).reshape(-1)
  ee = e[None].expand(len(g), -1).reshape(-1)
  r = gg % replay.L
  ok = replay.info_ok[r, ee] & (replay.ep_step[r, ee] >= 4) & (replay.done[r, ee] == 0)
  gg, ee = gg[ok], ee[ok]
  s4 = replay.history("s", gg, ee, 4)
  dprev = torch.stack([task.dyaw(s4[:, i], s4[:, i + 1]) for i in range(3)], -1)
  r = gg % replay.L
  s, s2, info = replay.s[r, ee], replay.s_next[r, ee], replay.info[r, ee]
  env_rew = replay.rew[r, ee]
  cmd = s2[:, task.S["cmd"]]
  tw = task.torque_work_rate(s2, cmd, replay.plant[r, ee])
  a, b = tw, env_rew[:, 3:5].sum(-1)
  res["torque_work"] = dict(n=len(a), env_mean=float(b.mean()), ours_mean=float(a.mean()),
                            mean_abs_diff=float((a - b).abs().mean()), corr=float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
  print(f"[parity] torque+work: env {res['torque_work']['env_mean']:+.4f} ours {res['torque_work']['ours_mean']:+.4f}  "
        f"mean|diff| {res['torque_work']['mean_abs_diff']:.2e}  corr {res['torque_work']['corr']:.4f}")
  terms, _ = task.reward_rates(s, s2, info, dprev, env_rew[:, 3:5].sum(-1))
  for i, name in enumerate(("rotate_finite_diff", "object_linvel_penalty", "pose_diff_penalty")):
    a, b = terms[name], env_rew[:, i]
    res[name] = dict(n=len(a), env_mean=float(b.mean()), ours_mean=float(a.mean()), max_abs_diff=float((a - b).abs().max()),
                     mean_abs_diff=float((a - b).abs().mean()), corr=float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    print(f"[parity] {name}: env {res[name]['env_mean']:+.4f} ours {res[name]['ours_mean']:+.4f}  mean|diff| "
          f"{res[name]['mean_abs_diff']:.2e}  max|diff| {res[name]['max_abs_diff']:.2e}  corr {res[name]['corr']:.4f}")
  (out / "parity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
