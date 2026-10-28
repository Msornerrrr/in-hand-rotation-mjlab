"""SuperMimic-style policy learning on the palm-down 22-object fingertip rotation task (``-PalmDown-MixTeacherF``
rewards, from scratch): the loop of ``train_yaw.py`` with the task adapter ``world_model.ftrot_mix``.

Differences from the cube loop:
  - the actor observation is built by the adapter in collection and imagination alike (the deployable ``-Mix``
    actor's 768-D proprio history; ``--actor priv`` adds the object state and descriptors);
  - the world model gets a contact head (pad forces and heights, link and palm contact) trained on the task's sensors;
    the contact reward terms use its prediction in imagination;
  - the rotation reward's rate -> ratchet mix ramps on this loop's own update count (``--ramp-delay``, ``--ramp``);
  - random object pushes are off (a deterministic model cannot predict them); cubes and cuboids join the reset
    distribution after ``--stage2-steps`` env steps (the PPO teacher's curriculum).

  MUJOCO_GL=egl uv run python scripts/wm/train_ftrot.py --actor proprio --out ...
  ... --parity    # the reward reconstruction against the env (true contact signals), then exit
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import in_hand_rotation_mjlab.tasks  # noqa: F401
from in_hand_rotation_mjlab.tasks.fingertip_rotate import rl_mdp as M
from in_hand_rotation_mjlab.mdp.events import _resolve_env_ids
from in_hand_rotation_mjlab.world_model import record as R
from in_hand_rotation_mjlab.world_model.ftrot_mix import (CONTACT_DIM, DROP_DRIFT, H_OBS, PRESS_N, TERMS, ContactHead, FailHead,
                                            FtrotMix, StateEstimator, contact_signals)
from in_hand_rotation_mjlab.world_model.model import CommandMap, StateError, WMConfig, WorldModel
from in_hand_rotation_mjlab.world_model.policy import Actor, Critic
from in_hand_rotation_mjlab.world_model.replay import Replay
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

TASK = "Mjlab-Leap-Left-Custom-Fingertip-Rotate-PalmDown-MixTeacherF"
PALM = "down"


class TerminalWithContacts(R.TerminalRecorder):
  def record_pre_reset(self, env_ids):
    super().record_pre_reset(env_ids)
    if self.L is not None:
      self.contacts = contact_signals(self._env, PALM)[self.ids]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--actor", choices=["proprio", "priv"], default="proprio")
  ap.add_argument("--envs", type=int, default=4096)
  ap.add_argument("--replay", type=int, default=800)
  ap.add_argument("--hours", type=float, default=6.0)
  ap.add_argument("--max-updates", type=int, default=0,
                  help="> 0: also stop after this many updates in this run (counted from the init checkpoint's), for budget-matched runs across GPUs")
  ap.add_argument("--horizon", type=int, default=16)
  ap.add_argument("--starts", type=int, default=4096)
  ap.add_argument("--wm-steps", type=int, default=10)
  ap.add_argument("--wm-batch", type=int, default=2048)
  ap.add_argument("--wm-k", type=int, default=8)
  ap.add_argument("--wm-hidden", default="1024,1024,1024")
  ap.add_argument("--compile", type=int, default=0,
                  help="1: torch.compile the world model for the fitting windows and imagination (~2x faster WM fit). "
                       "Was broken by the per-update requires_grad toggling (rotation 0.51 vs 1.37 at update 1100); "
                       "imagination now runs through detached weights instead. Smoke (seed 3, 1024 envs, 600 updates): "
                       "no stall, rotation ahead of eager. Not yet validated at full scale.")
  ap.add_argument("--wm-bf16", type=int, default=0, help="1: the world model's MLP under bfloat16 autocast")
  ap.add_argument("--wm-only", type=int, default=300)
  ap.add_argument("--utd", type=float, default=0.4)
  ap.add_argument("--pi-steps", type=int, default=1, help="actor + critic steps per update (fresh imagination starts each)")
  ap.add_argument("--noise", type=float, default=0.1)
  ap.add_argument("--warmup", type=int, default=64)
  ap.add_argument("--warmup-noise", type=float, default=0.5)
  ap.add_argument("--gamma", type=float, default=0.99)
  ap.add_argument("--lam", type=float, default=0.95)
  ap.add_argument("--lr", type=float, default=2e-4)
  ap.add_argument("--ramp-delay", type=int, default=3000, help="updates of the yaw-rate reward before the ratchet ramp")
  ap.add_argument("--ramp", type=int, default=3000, help="updates over which the reward ramps to the ratchet")
  ap.add_argument("--stage2-steps", type=int, default=2500, help="env steps before cubes / cuboids join the resets")
  ap.add_argument("--ckpt-min", type=float, default=10.0)
  ap.add_argument("--log-every", type=int, default=50)
  ap.add_argument("--calibrate-term", type=int, default=1,
                  help="imagination's drop-logit correction: 0 none, 1 the prior shift of the oversampled training "
                       "batches, 2 empirical: the shift that makes the mean predicted drop probability on the newest "
                       "--recal-window real steps equal their drop rate (re-solved every --recal-every updates); 3: 2 plus an adaptive extra shift that "
                       "matches the imagined drop hazard to the newest real one (the actor otherwise steers into "
                       "states where the head under-predicts drops)")
  ap.add_argument("--drop-soft-tau", type=float, default=0.0,
                  help="> 0: imagination's drop probability also includes sigmoid((drift - 3 cm) / tau) of the predicted "
                       "object drift (the task's exact drop rule, smoothed), combined with the head as a union; "
                       "--calibrate-term 3 matches only the head's hazard")
  ap.add_argument("--wm-drop-frac", type=float, default=0.0,
                  help="share of the world model's k-step training windows that end in a drop (the deterministic "
                       "model otherwise smooths the rare fast falls away: ~10%% of real drops foreseen 8 steps ahead)")
  ap.add_argument("--fail-head", type=int, default=0,
                  help="1: imagination's drop probability from a separate FailHead (drift from reset, size, context) "
                       "trained on true and predicted next states of the world-model windows, importance-weighted "
                       "against the drop-window oversampling (use with --calibrate-term 0)")
  ap.add_argument("--fail-last", type=int, default=0,
                  help="1: train the FailHead on each window's last step only. Windows never cross a reset, so a drop "
                       "can only be a window's last step: labels over all k steps put the head's base rate ~k x low")
  ap.add_argument("--fail-clip", type=float, default=0.1, help="FailHead: max drop probability per imagined step")
  ap.add_argument("--ema", type=float, default=0.0,
                  help="> 0: checkpoints hold a Polyak average of the actor (decay per update); collection uses the raw one")
  ap.add_argument("--pad-friction", type=float, nargs=2, default=None,
                  help="override the pad sliding-friction randomization range (lo hi) of the training env (as ft_student.py)")
  ap.add_argument("--pad-friction-mix", type=float, nargs=3, default=None, metavar=("LO", "HI", "P"),
                  help="each reset draws pad friction from U(LO, HI) with probability P, else from the normal range")
  ap.add_argument("--pad-friction-mix-from", type=int, default=0,
                  help="> 0: the mix probability stays 0 until this update, then ramps linearly to P over "
                       "--pad-friction-mix-len (a from-scratch run learns the gait on the normal range first)")
  ap.add_argument("--pad-friction-mix-len", type=int, default=1000)
  ap.add_argument("--collect-ema", type=int, default=0,
                  help="1: collect real data with the --ema average (+ the same noise) instead of the raw actor")
  ap.add_argument("--lr-decay-start", type=int, default=0, help="> 0: the actor lr decays linearly from this update")
  ap.add_argument("--lr-decay-len", type=int, default=5000)
  ap.add_argument("--lr-min-frac", type=float, default=0.2, help="actor lr at the end of the decay, as a fraction of --lr")
  ap.add_argument("--drop-weight", type=float, default=1.0,
                  help="multiplies the drop penalty (-400 / s) in imagination only; the env's reward is unchanged")
  ap.add_argument("--drop-weight-from", type=int, default=0,
                  help="> 0: the drop weight stays 1 until this update, then ramps linearly to --drop-weight over "
                       "--drop-weight-len (learn to turn the prisms first; a high weight from the start freezes them)")
  ap.add_argument("--drop-weight-len", type=int, default=1000)
  ap.add_argument("--drift-penalty", type=float, default=0.0,
                  help="> 0: imagination-only shaping, a per-second cost of W x ((drift - free) / (3 cm - free)) clipped to "
                       "[0, 1]: the object's slide away from its reset position before the 2 cm gate and the 3 cm drop")
  ap.add_argument("--drift-free", type=float, default=0.01, help="drift (m) the --drift-penalty leaves free")
  ap.add_argument("--drift-penalty-from", type=int, default=0,
                  help="> 0: the drift cost stays 0 until this update, then ramps linearly to W over --drift-penalty-len")
  ap.add_argument("--drift-penalty-len", type=int, default=1000)
  ap.add_argument("--hard-starts", type=float, default=0.0,
                  help="> 0: this share of the imagination starts are the most-drifted of a 4x larger replay sample "
                       "(the object furthest from its reset position), the states where it is about to be lost")
  ap.add_argument("--few-pads-penalty", type=float, default=0.0,
                  help="> 0: imagination-only shaping, a per-second cost of W per pad short of three pressing (the contact "
                       "head's predicted pad forces, soft count)")
  ap.add_argument("--real-critic", type=int, default=0,
                  help="> 0: the critic also regresses TD(lambda) targets on this many real k-step replay windows per "
                       "update (imagination's reward on the true transitions, true drops), so the bootstrap carries the "
                       "real drop risk the world model under-predicts")
  ap.add_argument("--real-critic-k", type=int, default=32)
  ap.add_argument("--real-critic-window", type=int, default=200, help="newest replay steps the real windows come from")
  ap.add_argument("--init-models", default=None, help="the world-model checkpoint that goes with --init (default: beside it)")
  ap.add_argument("--init", default=None,
                  help="an actor_<N>.pt checkpoint: start from it and its models_<N>.pt (world model, heads, critic); "
                       "the update count continues from N (pair with --stage2-steps 0)")
  ap.add_argument("--estimator", type=int, default=0,
                  help="1 (proprio actor): a StateEstimator, trained on the replay, predicts the privileged features "
                       "from the history; the actor reads history + estimate (both deployable)")
  ap.add_argument("--guide", default=None,
                  help="a privileged wm_actor checkpoint: the (proprio) actor also imitates it on the imagined states "
                       "(DAgger in imagination: the student's own rollouts, the teacher's actions as labels)")
  ap.add_argument("--guide-beta", type=float, default=10.0, help="weight of the imitation loss (actions in [-1, 1])")
  ap.add_argument("--guide-decay", type=int, default=0, help="> 0: beta decays linearly to 0 over this many updates")
  ap.add_argument("--hazard-lr", type=float, default=0.3, help="--calibrate-term 3: log-rate step per re-solve")
  ap.add_argument("--calibrate-from", type=int, default=0,
                  help="updates before --calibrate-term applies (mode 0 until then): learn to rotate with an optimistic "
                       "drop model first, then match the imagined drop hazard to the real one")
  ap.add_argument("--recal-window", type=int, default=200, help="newest replay steps (x envs) for --calibrate-term 2")
  ap.add_argument("--recal-every", type=int, default=25)
  ap.add_argument("--profile", action="store_true", help="time the loop phases (cuda-synchronized) into the log")
  ap.add_argument("--parity", action="store_true")
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
  env_cfg.events.pop("random_push", None)
  env_cfg.events["reset_from_pool"].params["stage2_step"] = args.stage2_steps
  if args.pad_friction is not None:
    env_cfg.events["dr_split_contact_friction"].params["pad_range"] = tuple(args.pad_friction)
  mp_target = args.pad_friction_mix[2] if args.pad_friction_mix is not None else 0.0
  mix_p_at = lambda n: (mp_target if args.pad_friction_mix_from <= 0 else  # noqa: E731
                        mp_target * min(1.0, max(0.0, (n - args.pad_friction_mix_from) / max(1, args.pad_friction_mix_len))))
  mix_p = [mix_p_at(0)]  # updated every update in the training loop
  if args.pad_friction_mix is not None:  # a mixture: keep the full range in the data, oversample low friction
    ev = env_cfg.events["dr_split_contact_friction"]
    base, (mlo, mhi, mp) = ev.func, args.pad_friction_mix

    def mixed_friction(env, env_ids, pad_range, other_range):
      base(env, env_ids, pad_range, other_range)
      ids = _resolve_env_ids(env, env_ids)
      sub = ids[torch.rand(len(ids), device=env.device) < mix_p[0]]
      if len(sub):
        base(env, sub, (mlo, mhi), other_range)

    mixed_friction.model_fields, mixed_friction.recompute = base.model_fields, base.recompute
    ev.func = mixed_friction
  env_cfg.recorders = {"wm_terminal": RecorderTermCfg(func=TerminalWithContacts)}
  weights = {k: v.weight for k, v in env_cfg.rewards.items()}
  env = ManagerBasedRlEnv(cfg=env_cfg, device=dev)
  env.reset()  # sets the task's per-episode state (p0 ...); without it every env reports a drop on its first step
  L = R.layout(env)
  rec = env.recorder_manager.get_term("wm_terminal")
  rec.L = L
  nj, N, dt = L.nj, args.envs, env.step_dt
  robot = env.scene["robot"]
  st = M.ft(env, PALM)
  full, names = R.context_full(env)
  cidx, cnames = R.varying_columns(full, names)
  init_ck = mk = None
  if args.init:  # the world model's context columns are the saved ones (which columns vary depends on the resets)
    init_ck = torch.load(args.init, map_location=dev, weights_only=False)
    ip = Path(args.init)  # its world model: --init-models, else models_N.pt beside actor_N.pt, else <stem>_models.pt
    mpath = (Path(args.init_models) if args.init_models else
             ip.with_name(ip.name.replace("actor_", "models_")) if ip.name.startswith("actor_") else ip.with_name(ip.stem + "_models.pt"))
    mk = torch.load(mpath, map_location=dev,
                    weights_only=False)
    cnames = list(mk["ctx_names"])
    cidx = torch.tensor([names.index(n) for n in cnames], dtype=torch.long, device=full.device)
  n_hand = quat_apply_inverse(L.base_q[:1], st.n_w[None])[0]
  task = FtrotMix(nj, robot.data.default_joint_pos[0, L.term.target_ids].clone(), L.base_q[0].clone(), n_hand, dt,
                  weights, cnames)
  w_drop0 = task.w["dropped"]
  dw_at = lambda n: (args.drop_weight if args.drop_weight_from <= 0 else  # noqa: E731
                     1 + (args.drop_weight - 1) * min(1.0, max(0.0, (n - args.drop_weight_from) / max(1, args.drop_weight_len))))
  task.w["dropped"] = w_drop0 * dw_at(0)
  drift_w_at = lambda n: (args.drift_penalty if args.drift_penalty_from <= 0 else  # noqa: E731
                          args.drift_penalty * min(1.0, max(0.0, (n - args.drift_penalty_from) / max(1, args.drift_penalty_len))))
  rnames = list(env.reward_manager._term_names)
  assert set(TERMS) <= set(rnames), rnames
  ridx = [rnames.index(t) for t in TERMS]
  meta = dict(delta_min=L.term._delta_min.tolist(), delta_max=L.term._delta_max.tolist(),
              max_lead=None if L.term._max_lead is None else L.term._max_lead.tolist(),
              soft_limits=robot.data.soft_joint_pos_limits[0][L.term.target_ids].tolist(),
              clip_to_joint_limits=bool(L.term.cfg.clip_to_joint_limits), step_dt=dt, nj=nj)
  cmap = CommandMap(meta).to(dev)
  from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import LeapGearServoActuator

  servo = next(a for a in robot.actuators if isinstance(a, LeapGearServoActuator))
  plant_now = lambda: torch.cat([servo.current_gain, servo.current_damping, servo.current_limit,  # noqa: E731
                                 servo.torque_constant], -1).clone()

  # --- collection state ---
  cur_s = R.read_state(env, L)
  cur_ctx = full[:, cidx].clone()
  cur_epi = torch.arange(N, device=dev)
  n_epi = N
  cur_info = torch.cat([cur_s[:, :nj], cur_s[:, task.S["p"]], st.extent[:, None]], -1)
  cur_step = env.episode_length_buf.clone().long()
  cur_plant = plant_now()
  aux_now = lambda: torch.cat([(st.qyaw - st.qyaw_max)[:, None], st.idle], -1).clone()  # noqa: E731
  cur_aux = aux_now()
  lag = torch.zeros(N, dtype=torch.bool, device=dev)
  q_prev = cur_s[:, :nj].clone()
  frame0 = task.obs_frame(cur_s[:, :nj], cur_s[:, task.S["cmd"]], cur_ctx)
  hist = frame0[:, None].expand(-1, H_OBS, -1).clone()
  priv_dim = task.priv(cur_s, cur_info, cur_ctx).shape[-1] if args.actor == "priv" else 0
  est = None
  if args.estimator:
    assert args.actor == "proprio", "--estimator is for the proprio actor"
    est = StateEstimator(task.obs_dim, task.priv(cur_s, cur_info, cur_ctx).shape[-1]).to(dev)
    opt_e = torch.optim.Adam(est.parameters(), lr=args.lr)
  actor = Actor(task.obs_dim + priv_dim + (est.out_dim if est is not None else 0), nj).to(dev)
  if init_ck is not None:
    assert (init_ck["obs_dim"], init_ck["act_dim"]) == (actor.obs_dim, actor.act_dim), "--init: actor dims differ"
    actor.load_state_dict(init_ck["state_dict"])
    if est is not None:
      est.load_state_dict(init_ck["est_state"])
  ema = copy.deepcopy(actor) if args.ema > 0 else None

  def ck_extra():
    return {} if est is None else dict(est_state={k: v.detach().cpu() for k, v in est.state_dict().items()},
                                       est_dims=(est.obs_dim, est.out_dim), est_hidden=est.hidden)
  guide = None
  if args.guide:
    gck = torch.load(args.guide, map_location=dev, weights_only=False)
    assert gck.get("actor_kind") == "priv" and list(gck["ctx_names"]) == list(cnames), "guide: needs a priv actor, same ctx"
    guide = Actor.from_checkpoint(gck).to(dev).eval()
    for p_ in guide.parameters():
      p_.requires_grad_(False)
  replay = Replay(args.replay, N, L.dim, nj, len(cidx), task.frame_dim, task.info_dim, len(TERMS), dev, d_plant=4 * nj)
  replay.contact = torch.zeros(args.replay, N, CONTACT_DIM, device=dev)
  replay.aux = torch.zeros(args.replay, N, task.aux_dim, device=dev)
  stats = dict(drops=0)

  def actor_input(hist_, s, info, ctx_raw):
    x = task.flatten(hist_)
    if est is not None:
      x = torch.cat([x, est(x)], -1)
    return torch.cat([x, task.priv(s, info, ctx_raw)], -1) if args.actor == "priv" else x

  noise = args.noise

  def env_step():
    nonlocal cur_s, cur_ctx, n_epi, cur_info, cur_step, cur_plant, cur_aux, q_prev, hist, lag
    with torch.no_grad():
      x = actor_input(hist, cur_s, cur_info, cur_ctx)
      actor.norm.update(x)
      pol = ema if (ema is not None and args.collect_ema) else actor
      a = (pol(x) + noise * torch.randn(N, nj, device=dev)).clamp(-1.0, 1.0)
      rec.clear()
      _, _, term, trunc, _ = env.step(a)
      d = (term | trunc).bool()
      s_post = R.read_state(env, L)
      s_true = s_post.clone()
      con = contact_signals(env, PALM)
      if d.any():
        s_true[rec.ids] = rec.rows
        con[rec.ids] = rec.contacts
      done = torch.where(d, torch.where(trunc.bool(), 2, 1), 0).to(torch.uint8)
      stats["drops"] += int((done == 1).sum())
      # a physics blow-up (the task's NaN termination) leaves non-finite rows: kept out of every fit (episode id -1)
      bad = ~(torch.isfinite(s_true).all(-1) & torch.isfinite(con).all(-1) & torch.isfinite(cur_s).all(-1))
      epi_row = cur_epi.clone()
      if bad.any():
        stats["bad"] = stats.get("bad", 0) + int(bad.sum())
        s_true[bad], con[bad], epi_row[bad] = torch.nan_to_num(cur_s[bad]), 0.0, -1
      replay.add(s=torch.nan_to_num(cur_s), s_next=s_true, a=a, ctx=cur_ctx, frame=hist[:, -1], info=cur_info,
                 info_ok=cur_step >= 0, plant=cur_plant, rew=env.reward_manager._step_reward[:, ridx].clone(), done=done,
                 epi=epi_row, ep_step=cur_step, contact=con, aux=cur_aux)
      # advance to s_post (the next episode's first state where reset)
      q_prev = torch.where(d[:, None], s_post[:, :nj], cur_s[:, :nj])
      cur_s = s_post
      cur_step = env.episode_length_buf.clone().long()
      cur_aux = aux_now()
      if d.any():
        ids = d.nonzero().squeeze(-1)
        cur_epi[ids] = torch.arange(n_epi, n_epi + len(ids), device=dev)
        n_epi += len(ids)
        f2, _ = R.context_full(env)
        cur_ctx[ids] = f2[ids][:, cidx]
        cur_info[ids] = torch.cat([s_post[ids, :nj], s_post[ids][:, task.S["p"]], st.extent[ids, None]], -1)
        cur_plant = plant_now()
      # observation lag (0/1, held, resampled with p 0.1 every 10 steps as the env's delay buffer)
      flip = (torch.rand(N, device=dev) < 0.01)
      lag = torch.where(flip, torch.rand(N, device=dev) < 0.5, lag)
      q_obs = torch.where(lag[:, None], q_prev, cur_s[:, :nj])
      fr = task.obs_frame(q_obs, cur_s[:, task.S["cmd"]], cur_ctx)
      hist = torch.cat([hist[:, 1:], fr[:, None]], 1)
      if d.any():
        hist[ids] = fr[ids, None].expand(-1, H_OBS, -1)

  if args.parity:
    noise = args.warmup_noise
    for _ in range(args.warmup):
      env_step()
    parity(replay, task, ridx, rnames, env, out)
    return

  # --- models ---
  wcfg = WMConfig(nj=nj, dc=len(cidx), hidden=tuple(int(h) for h in args.wm_hidden.split(",")), dt=dt,
                  bf16=bool(args.wm_bf16))
  wm = WorldModel(wcfg).to(dev)
  wm_c = torch.compile(wm) if args.compile else wm  # fitting; diagnostics stay eager

  def wm_frozen(s_, cmd_, ctx_):  # the same model through detached weights (no gradient to the model)
    w = {k: v.detach() for k, v in wm.named_parameters()}
    return torch.func.functional_call(wm, {**w, **dict(wm.named_buffers())}, (s_, cmd_, ctx_))

  # imagination: compiled, the model's weights never toggle requires_grad (a graph traced with frozen weights and
  # reused for fitting gave the model no gradient: WM loss climbed once the policy phase began). Eager is unchanged.
  wm_i = torch.compile(wm_frozen) if args.compile else wm
  head = ContactHead(nj, len(cidx)).to(dev)
  fail = FailHead(nj, len(cidx)).to(dev)
  opt_f = torch.optim.Adam(fail.parameters(), lr=args.lr)
  err = StateError(nj).to(dev)
  c_mu = torch.zeros(len(cidx), device=dev)
  c_sd = torch.ones(len(cidx), device=dev)
  aux0 = torch.zeros(2, task.aux_dim, device=dev)
  critic = Critic(task.critic_features(cur_s[:2], cur_info[:2], aux0, cur_ctx[:2]).shape[-1]).to(dev)
  opt_wm = torch.optim.Adam(list(wm.parameters()) + list(head.parameters()), lr=args.lr)
  opt_pi = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(0.9, 0.95))
  opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
  r_max = 4.0 * dt

  noise = args.warmup_noise
  for _ in range(args.warmup):
    env_step()
  noise = args.noise
  with torch.no_grad():
    rows = torch.arange(replay.G, device=dev) % replay.L
    s, sn = replay.s[rows].reshape(-1, L.dim), replay.s_next[rows].reshape(-1, L.dim)
    cx = replay.ctx[rows].reshape(-1, len(cidx))
    c_mu.copy_(cx.mean(0)); c_sd.copy_(cx.std(0).clamp_min(1e-6))
    if est is not None:
      ge, ee = replay.sample_starts(16384, H_OBS)
      re_ = ge % replay.L
      est.fit_stats(task.flatten(replay.history("frame", ge, ee, H_OBS)),
                    task.priv(replay.s[re_, ee], replay.info[re_, ee], replay.ctx[re_, ee]))
    wm.fit_stats(s, sn[:, wm.S["cmd"]], (cx - c_mu) / c_sd, sn)
    err.fit(s, sn)
    fail.fit_stats(sn, replay.info[rows].reshape(-1, task.info_dim), (cx - c_mu) / c_sd)
    hf = head.features(sn, sn[:, wm.S["cmd"]], (cx - c_mu) / c_sd)
    head.mu.copy_(hf.mean(0)); head.sd.copy_(hf.std(0).clamp_min(1e-4))
  norm_c = lambda c: (c - c_mu) / c_sd  # noqa: E731
  @torch.no_grad()
  def real_targets(n, k, a_mix_):
    """Critic features and TD(lambda) targets on n real k-step windows inside one episode (newest replay rows)."""
    lo, hi = replay.G - min(args.real_critic_window, replay.G, replay.L), replay.G - k
    j = torch.arange(k, device=dev)[:, None]
    gs, es, m = [], [], 0
    for _ in range(20):
      g_ = torch.randint(lo, hi + 1, (2 * n,), device=dev)
      e_ = torch.randint(0, N, (2 * n,), device=dev)
      rr_ = (g_[None] + j) % replay.L
      ep = replay.epi[rr_, e_[None]]
      ok_ = (ep == ep[:1]).all(0) & (ep[0] >= 0) & replay.info_ok[rr_, e_[None]].all(0)
      gs.append(g_[ok_]); es.append(e_[ok_]); m += int(ok_.sum())
      if m >= n:
        break
    g_, e_ = torch.cat(gs)[:n], torch.cat(es)[:n]
    B = len(g_)
    rr, E = (g_[None] + j) % replay.L, e_[None].expand(k, -1)
    fl = lambda x: x.reshape(k * B, *x.shape[2:])  # noqa: E731
    s_, s2_, info_, aux_ = fl(replay.s[rr, E]), fl(replay.s_next[rr, E]), fl(replay.info[rr, E]), fl(replay.aux[rr, E])
    ctx_, dn = fl(norm_c(replay.ctx[rr, E])), replay.done[rr, E]
    terms, aux2 = task.reward_rates(s_, s2_, s2_[:, task.S["cmd"]], info_, aux_, fl(replay.contact[rr, E]),
                                    fl(replay.plant[rr, E]), a_mix_, p_drop=fl(dn == 1).float())
    r = (sum(terms.values()) * dt).reshape(k, B)
    X = task.critic_features(s_, info_, aux_, ctx_).reshape(k, B, -1)
    last = slice((k - 1) * B, k * B)
    XH = task.critic_features(s2_[last], info_[last], aux2[last], ctx_[last])
    V = critic(torch.cat([X, XH[None]]))
    cont = (dn != 1).float()
    Rt, tgt = V[-1], []
    for h in reversed(range(k)):
      Rt = r[h] + args.gamma * cont[h] * ((1 - args.lam) * V[h + 1] + args.lam * Rt)
      tgt.append(Rt)
    T_ = torch.stack(tgt[::-1]).clamp(-r_max * 50 / (1 - args.gamma), r_max / (1 - args.gamma))
    return X.reshape(k * B, -1), T_.reshape(-1)

  if mk is not None:  # after the stats block: the saved normalizers win
    wm.load_state_dict(mk["wm"]); head.load_state_dict(mk["head"]); critic.load_state_dict(mk["critic"])
    fail.load_state_dict(mk["fail"]); c_mu.copy_(mk["c_mu"]); c_sd.copy_(mk["c_sd"])

  log = (out / "log.jsonl").open("a")
  t0, last_ckpt = time.time(), time.time()
  n_up, acc, hist_log = (int(init_ck["update"]) if init_ck is not None else 0), 0.0, []
  term_shift = 0.0
  emp_shift, emp_pos, real_rate, hazard_k, pd_sum, pd_n = None, 0, None, 0.0, 0.0, 0
  last_G, last_drops = replay.G, 0

  @torch.no_grad()
  def empirical_shift():
    """Shift s with mean(sigmoid(logit - s)) == drop rate over the newest real steps (None if < 30 drops)."""
    W = min(args.recal_window, replay.G, replay.L)
    rows = torch.arange(replay.G - W, replay.G, device=dev) % replay.L
    ok = replay.epi[rows] >= 0
    y = (replay.done[rows] == 1)[ok].float()
    if y.sum() < 30:
      return None, int(y.sum()), None
    s_, a_, c_ = replay.s[rows][ok], replay.a[rows][ok], norm_c(replay.ctx[rows][ok])
    i_ = replay.info[rows][ok]
    lgs = []
    for i in range(0, len(s_), 65536):
      sl = slice(i, i + 65536)
      p2, lw = wm(s_[sl], cmap(s_[sl][..., wm.S["cmd"]], a_[sl], s_[sl][..., wm.S["q"]]), c_[sl])
      lgs.append(fail(p2, i_[sl], c_[sl]) if args.fail_head else lw)
    lg = torch.cat(lgs)
    target, lo, hi = float(y.mean()), -30.0, 30.0
    for _ in range(50):  # mean sigmoid(lg - s) decreases in s
      mid = 0.5 * (lo + hi)
      lo, hi = (mid, hi) if float(torch.sigmoid(lg - mid).mean()) > target else (lo, mid)
    return 0.5 * (lo + hi), int(y.sum()), target

  @torch.no_grad()
  def drop_windows(n, k):
    """(g, e) of up to n k-step windows inside one episode whose last step is a drop."""
    rr, ee = (replay.done == 1).nonzero(as_tuple=True)
    if len(rr) == 0:
      return None, None
    pick = torch.randint(len(rr), (2 * n,), device=dev)
    r, e = rr[pick], ee[pick]
    g = replay.G - 1 - ((replay.G - 1 - r) % replay.L)  # newest global step stored in row r
    g0 = g - k + 1
    j = torch.arange(k, device=dev)[:, None]
    ok = g0 >= max(0, replay.G - replay.L)
    ep = replay.epi[(g0.clamp(min=0)[None] + j) % replay.L, e[None]]
    ok &= (ep == ep[:1]).all(0) & (ep[0] >= 0)
    return g0[ok][:n], e[ok][:n]

  @torch.no_grad()
  def drop_diag(shift_now, k=8, W=400, n_max=2048):
    """Does the world model see the real drops coming? Replays recent real k-step windows that end in a drop (and as
    many that do not) open-loop with the executed actions: share of drop windows whose predicted object drift ends
    > 3 cm (k and 1 step ahead), mean predicted / true end drift, the head's calibrated drop probability at the end,
    and the false-alarm share of the non-drop windows."""
    W = min(W, replay.G - k, replay.L - k)
    if W <= 0:
      return {}
    g_end = torch.arange(replay.G - W, replay.G, device=dev)
    r_end = g_end % replay.L
    dn = replay.done[r_end] == 1
    ok_len = replay.ep_step[r_end] >= k
    out = {}
    for name, sel in (("drop", dn & ok_len), ("keep", (replay.done[r_end] == 0) & ok_len & (replay.epi[r_end] >= 0))):
      gi, ei = sel.nonzero(as_tuple=True)
      if len(gi) == 0:
        continue
      pick = torch.randperm(len(gi), device=dev)[:n_max]
      ge, e = g_end[gi[pick]], ei[pick]
      g0 = ge - k + 1
      s0, acts, snx, cx, _ = replay.window(g0, e, k)
      same = (replay.epi[(g0[None] + torch.arange(k, device=dev)[:, None]) % replay.L, e[None]] ==
              replay.epi[g0 % replay.L, e][None]).all(0)
      if same.sum() == 0:
        continue
      s0, acts, snx, cx, e, g0, ge = s0[same], acts[:, same], snx[:, same], cx[same], e[same], g0[same], ge[same]
      p0 = replay.info[g0 % replay.L, e][:, nj:nj + 3]
      c = norm_c(cx)
      sh = s0
      for j in range(k):
        sh, lg = wm(sh, cmap(sh[..., wm.S["cmd"]], acts[j], sh[..., wm.S["q"]]), c)
      s1, _ = wm(snx[-2] if k > 1 else s0, cmap((snx[-2] if k > 1 else s0)[..., wm.S["cmd"]], acts[-1],
                                                 (snx[-2] if k > 1 else s0)[..., wm.S["q"]]), c)
      dk = (sh[..., wm.S["p"]] - p0).norm(dim=-1)
      d1 = (s1[..., wm.S["p"]] - p0).norm(dim=-1)
      dt_ = (snx[-1][..., wm.S["p"]] - p0).norm(dim=-1)
      if args.fail_head:
        lg = fail(sh, replay.info[g0 % replay.L, e], c)
      ph = torch.sigmoid(lg - shift_now)
      out.update({f"dd_{name}_n": int(len(dk)), f"dd_{name}_hit_k": float((dk > DROP_DRIFT).float().mean()),
                  f"dd_{name}_hit_1": float((d1 > DROP_DRIFT).float().mean()),
                  f"dd_{name}_drift_pred_k": float(dk.mean()), f"dd_{name}_drift_true": float(dt_.mean()),
                  f"dd_{name}_head_p_k": float(ph.mean())})
      if task.padfr_idx is not None:  # the same, split by the episode's pad friction (lo: mu < 0.4, hi: mu >= 0.55)
        mu = cx[:, task.padfr_idx]
        for bn, bm in (("lo", mu < 0.4), ("hi", mu >= 0.55)):
          if bm.any():
            out.update({f"dd_{name}_{bn}_n": int(bm.sum()), f"dd_{name}_{bn}_hit_k": float((dk[bm] > DROP_DRIFT).float().mean()),
                        f"dd_{name}_{bn}_head_p_k": float(ph[bm].mean()),
                        f"dd_{name}_{bn}_drift_pred_k": float(dk[bm].mean()), f"dd_{name}_{bn}_drift_true": float(dt_[bm].mean())})
    if task.padfr_idx is not None:  # real per-step drop rate by pad friction over the same rows
      mu_all, live = replay.ctx[r_end][..., task.padfr_idx], replay.epi[r_end] >= 0
      for bn, bm in (("lo", mu_all < 0.4), ("hi", mu_all >= 0.55)):
        if (bm & live).any():
          out[f"dd_rate_{bn}"] = float((replay.done[r_end][bm & live] == 1).float().mean())
    return out

  def set_grad(mods, flag):
    for m in mods:
      for p in m.parameters():
        p.requires_grad_(flag)

  prof = {}
  prof_t = [time.time()]

  def mark(name):
    if args.profile:
      torch.cuda.synchronize()
      now = time.time()
      prof[name] = prof.get(name, 0.0) + now - prof_t[0]
      prof_t[0] = now

  up_start = n_up
  while time.time() - t0 < args.hours * 3600 and not (args.max_updates > 0 and n_up - up_start >= args.max_updates):
    mark("other")
    env_step()
    mark("collect")
    acc += args.utd
    while acc >= 1.0:
      acc -= 1.0
      n_up += 1
      stt = {}
      for _ in range(args.wm_steps):
        nd = int(args.wm_drop_frac * args.wm_batch)
        gd, ed = drop_windows(nd, args.wm_k) if nd else (None, None)
        g, e = replay.sample_windows(args.wm_batch - (0 if gd is None else len(gd)), args.wm_k)
        if gd is not None and len(gd):
          g, e = torch.cat([g, gd]), torch.cat([e, ed])
        s0, acts, snx, cx, dn = replay.window(g, e, args.wm_k)
        c = norm_c(cx)
        sh, ld, preds = s0, 0.0, []
        for j in range(args.wm_k):
          sh, _ = wm_c(sh, cmap(sh[..., wm.S["cmd"]], acts[j], sh[..., wm.S["q"]]), c)
          ld = ld + err(sh, snx[j]).mean()
          preds.append(sh.detach())
        ld = ld / args.wm_k
        if args.fail_head:  # true + predicted next states, importance weights undo the drop-window oversampling
          total = max(1, int((replay.epi >= 0).sum()))
          n_pool = max(1, int((replay.done == 1).sum()))
          q = (0 if gd is None else len(gd)) / len(g)
          in_pool = (dn[-1] == 1).float()
          imp = 1.0 / ((1 - q) + q * in_pool * total / n_pool)
          winfo = replay.info[g % replay.L, e]
          y, st_true, st_pred = (dn == 1).float(), snx, torch.stack(preds)
          if args.fail_last:
            y, st_true, st_pred = y[-1:], st_true[-1:], st_pred[-1:]
          kk = y.shape[0]
          lf_true = F.binary_cross_entropy_with_logits(fail(st_true, winfo[None].expand(kk, -1, -1),
                                                            c[None].expand(kk, -1, -1)), y, reduction="none")
          lf_pred = F.binary_cross_entropy_with_logits(fail(st_pred, winfo[None].expand(kk, -1, -1),
                                                            c[None].expand(kk, -1, -1)), y, reduction="none")
          lf = ((lf_true + lf_pred) * imp[None]).sum() / (2 * imp.numel() * kk)
          opt_f.zero_grad(set_to_none=True)
          lf.backward()
          torch.nn.utils.clip_grad_norm_(fail.parameters(), 1.0)
          opt_f.step()
          stt["fail"] = float(lf.detach())
        term = (replay.done == 1).nonzero()
        nt = min(len(term), args.wm_batch // 4)
        n_rows = int((replay.epi >= 0).sum())
        if nt and n_rows:  # prior shift of the oversampled batch: imagination corrects the head's logit by it
          q_b, pi = nt / args.wm_batch, max(len(term), 1) / n_rows
          term_shift = math.log(q_b / (1 - q_b)) - math.log(pi / (1 - pi))
        rr = torch.randint(0, min(replay.G, replay.L), (2 * (args.wm_batch - nt),), device=dev)
        ee = torch.randint(0, N, (2 * (args.wm_batch - nt),), device=dev)
        okr = replay.epi[rr, ee] >= 0
        rr, ee = rr[okr][:args.wm_batch - nt], ee[okr][:args.wm_batch - nt]
        if nt:
          pick = term[torch.randint(len(term), (nt,), device=dev)]
          rr, ee = torch.cat([rr, pick[:, 0]]), torch.cat([ee, pick[:, 1]])
        s1, a1, n1, c1 = replay.s[rr, ee], replay.a[rr, ee], replay.s_next[rr, ee], norm_c(replay.ctx[rr, ee])
        cmd1 = cmap(s1[..., wm.S["cmd"]], a1, s1[..., wm.S["q"]])
        with torch.no_grad():
          p1, _ = wm(s1, cmd1, c1)
        lt = F.binary_cross_entropy_with_logits(wm.term((wm._term_feat(p1) - wm.t_mu) / wm.t_sd).squeeze(-1),
                                                (replay.done[rr, ee] == 1).float())
        lc = head.loss(head(n1, cmd1, c1), replay.contact[rr, ee])  # trained on true next states
        opt_wm.zero_grad(set_to_none=True)
        (ld + lt + lc).backward()
        torch.nn.utils.clip_grad_norm_(list(wm.parameters()) + list(head.parameters()), 1.0)
        opt_wm.step()
      stt.update(wm=float(ld.detach()), term=float(lt.detach()), contact=float(lc.detach()))
      if est is not None:  # supervised on real replay states
        ge, ee = replay.sample_starts(4096, H_OBS)
        re_ = ge % replay.L
        le = est.loss(task.flatten(replay.history("frame", ge, ee, H_OBS)),
                      task.priv(replay.s[re_, ee], replay.info[re_, ee], replay.ctx[re_, ee]))
        opt_e.zero_grad(set_to_none=True)
        le.backward()
        torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
        opt_e.step()
        stt["est"] = float(le.detach())
      mark("wm")
      if n_up <= args.wm_only:
        hist_log.append(stt)
        continue
      a_mix = min(1.0, max(0.0, (n_up - args.ramp_delay) / max(1, args.ramp)))
      task.w["dropped"] = w_drop0 * dw_at(n_up)
      mix_p[0] = mix_p_at(n_up)
      cal = args.calibrate_term if n_up >= args.calibrate_from else 0
      if cal != args.calibrate_term:  # hazard averages start when calibration does
        pd_sum, pd_n = 0.0, 0
      if cal >= 2 and (emp_shift is None or n_up % args.recal_every == 0):
        new, emp_pos, rate = empirical_shift()
        if new is not None:
          emp_shift = new if emp_shift is None else 0.5 * emp_shift + 0.5 * new
          real_rate = rate
        if cal == 3 and real_rate and pd_n and pd_sum > 0:  # imagined hazard -> the real one
          hazard_k = min(4.0, max(-3.0, hazard_k + args.hazard_lr * (math.log(real_rate) - math.log(pd_sum / pd_n))))
          pd_sum, pd_n = 0.0, 0
      if cal >= 2 and emp_shift is not None:
        shift = emp_shift - (hazard_k if cal == 3 else 0.0)
      else:
        shift = term_shift if cal else 0.0
      for _pi in range(args.pi_steps):  # actor + critic steps per update, each on fresh imagination
        frozen = ([] if args.compile else [wm]) + [head, fail] + ([est] if est is not None else [])
        set_grad(frozen, False)
        g, e = replay.sample_starts(args.starts, H_OBS)
        hard_drift = 0.0
        if args.hard_starts > 0:
          k_h = int(args.hard_starts * len(g))
          g4, e4 = replay.sample_starts(4 * k_h, H_OBS)
          r4 = g4 % replay.L
          d4 = (replay.s[r4, e4][:, wm.S["p"]] - replay.info[r4, e4][:, nj:nj + 3]).norm(dim=-1)
          d4 = torch.where(d4 < 0.9 * DROP_DRIFT, d4, torch.full_like(d4, -1.0))  # not already lost
          top = d4.topk(min(k_h, len(d4))).indices
          g, e = torch.cat([g4[top], g[len(top):]]), torch.cat([e4[top], e[len(top):]])
          hard_drift = float(d4[top].mean())
        r0 = g % replay.L
        s, info, cxr = replay.s[r0, e], replay.info[r0, e], replay.ctx[r0, e]
        ctx, plant, aux = norm_c(cxr), replay.plant[r0, e], replay.aux[r0, e]
        frames = replay.history("frame", g, e, H_OBS)
        qp = replay.history("s", g, e, 2)[:, 0, :nj]
        lag_i = torch.rand(len(g), device=dev) < 0.5
        J, surv, disc = 0.0, torch.ones(len(g), device=dev), 1.0
        feats, rews, conts = [], [], []
        term_acc, pd_acc, pa_acc = {}, 0.0, 0.0
        bc = 0.0
        for h in range(args.horizon):
          feats.append(task.critic_features(s, info, aux, ctx).detach())
          a = actor(actor_input(frames, s, info, cxr))
          if guide is not None:
            with torch.no_grad():
              a_t = guide(torch.cat([task.flatten(frames), task.priv(s, info, cxr)], -1))
            bc = bc + (a - a_t).pow(2).mean() / args.horizon
          cmd = cmap(s[..., wm.S["cmd"]], a, s[..., wm.S["q"]])
          s2, logit = wm_i(s, cmd, ctx)
          if args.fail_head:
            pd = pd_head = torch.sigmoid(fail(s2, info, ctx) - shift)
            if args.fail_clip > 0:
              pd = pd_head = pd.clamp(max=args.fail_clip)
          else:
            pd = pd_head = torch.sigmoid(logit - shift)
          if args.drop_soft_tau > 0:
            drift = (s2[..., wm.S["p"]] - info[:, nj:nj + 3]).norm(dim=-1)
            pa = torch.sigmoid((drift - DROP_DRIFT) / args.drop_soft_tau)
            pd = 1 - (1 - pd_head) * (1 - pa)
            pa_acc += float(pa.detach().mean()) / args.horizon
          hc = head(s2, cmd, ctx)
          terms, aux = task.reward_rates(s, s2, cmd, info, aux, hc, plant, a_mix, p_drop=pd, soft=True)
          if args.drift_penalty > 0:
            drift = (s2[..., wm.S["p"]] - info[:, nj:nj + 3]).norm(dim=-1)
            terms["drift"] = -drift_w_at(n_up) * ((drift - args.drift_free) / (DROP_DRIFT - args.drift_free)).clamp(0.0, 1.0)
          if args.few_pads_penalty > 0:
            n_press = torch.sigmoid((hc["F"] - PRESS_N) / 0.03).sum(-1)
            terms["few_pads"] = -args.few_pads_penalty * (3.0 - n_press).clamp(min=0.0)
          r = sum(terms.values()) * dt
          for k_, v_ in terms.items():
            term_acc[k_] = term_acc.get(k_, 0.0) + float(v_.detach().mean()) / args.horizon
          pd_acc += float(pd_head.detach().mean()) / args.horizon
          J = J + disc * surv * r
          rews.append(r.detach()); conts.append((1 - pd).detach())
          surv = surv * (1 - pd)
          disc *= args.gamma
          q_obs = torch.where(lag_i[:, None], s[..., wm.S["q"]], s2[..., wm.S["q"]])
          frames = torch.cat([frames[:, 1:], task.obs_frame(q_obs, cmd, cxr)[:, None]], 1)
          s = s2
        pd_sum, pd_n = pd_sum + pd_acc, pd_n + 1
        fH = task.critic_features(s, info, aux, ctx)
        J = J + disc * surv * critic(fH)
        loss_pi = -J.mean()
        if guide is not None:
          beta = args.guide_beta * (max(0.0, 1 - n_up / args.guide_decay) if args.guide_decay else 1.0)
          loss_pi = loss_pi + beta * bc
        opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(actor.parameters(), 0.1)
        ok = all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
        if ok:
          if args.lr_decay_start > 0:
            fr = min(1.0, max(0.0, (n_up - args.lr_decay_start) / max(1, args.lr_decay_len)))
            for pg in opt_pi.param_groups:
              pg["lr"] = args.lr * (1 - (1 - args.lr_min_frac) * fr)
          opt_pi.step()
          if ema is not None:
            with torch.no_grad():
              for pe, pa in zip(ema.parameters(), actor.parameters()):
                pe.lerp_(pa, 1 - args.ema)
              for be, ba in zip(ema.buffers(), actor.buffers()):
                be.copy_(ba)
        set_grad(frozen, True)
        mark("policy")
        with torch.no_grad():
          feats.append(fH.detach())
          X = torch.stack(feats)
          critic.norm.update(X)
          V = critic(X)
          Rt, tgt = V[-1], []
          for h in reversed(range(args.horizon)):
            Rt = rews[h] + args.gamma * conts[h] * ((1 - args.lam) * V[h + 1] + args.lam * Rt)
            tgt.append(Rt)
          T = torch.stack(tgt[::-1]).clamp(-r_max * 50 / (1 - args.gamma), r_max / (1 - args.gamma))
        Xf, Tf = X[:-1].reshape(-1, X.shape[-1]), T.reshape(-1)
        if args.real_critic > 0:
          Xr, Tr = real_targets(args.real_critic, args.real_critic_k, a_mix)
          stt.update(v_real=float(Tr.mean()), v_imag=float(Tf.mean()))
          Xf, Tf = torch.cat([Xf, Xr]), torch.cat([Tf, Tr])
        lv = 0.0
        for _ in range(5):
          for mb in torch.randperm(len(Xf), device=dev).chunk(4):
            l = (critic(Xf[mb]) - Tf[mb]).pow(2).mean()
            opt_v.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_v.step()
            lv = float(l)
        stt.update(J=float(J.mean()), imag_r=float(torch.stack(rews).mean() / dt), v=lv, gn=float(gn), hard_drift=hard_drift,
                   surv=float(surv.mean()), a_mix=a_mix, finite=ok, p_drop=pd_acc, p_drop_drift=pa_acc, term_shift=shift, bc=float(bc),
                   term_shift_prior=term_shift, recal_drops=emp_pos, hazard_k=hazard_k,
                   **{"imag_" + k_: v_ for k_, v_ in term_acc.items()})
        hist_log.append(stt)
        mark("critic")
    if n_up and n_up % args.log_every == 0 and hist_log:
      G = replay.G
      k = min(G - last_G, replay.L)
      rows = torch.arange(G - k, G, device=dev) % replay.L
      real = replay.rew[rows].mean((0, 1))
      wall = time.time() - t0
      row = dict(update=n_up, env_steps=G * N, wall_s=wall, s_per_update=wall / max(1, n_up), pad_mix_p=mix_p[0],
                 stage=int(M.mixed_stage(env, args.stage2_steps)), real_reward_per_s=float(real.sum()),
                 real_rotate=float(real[0]),
                 real_drops_per_env_min=(stats["drops"] - last_drops) / N / max(1e-9, (G - last_G) * dt / 60),
                 **{k2: sum(hh[k2] for hh in hist_log if k2 in hh) / max(1, sum(k2 in hh for hh in hist_log))
                    for k2 in ("wm", "term", "contact", "J", "imag_r", "v", "gn", "surv", "a_mix", "p_drop", "term_shift",
                               "term_shift_prior", "recal_drops", "hazard_k", "p_drop_drift", "fail", "bc", "est", "v_real", "v_imag",
                               *("imag_" + t for t in TERMS), "imag_drift", "imag_few_pads", "hard_drift")},
                 **{"real_t_" + t: float(real[i]) for i, t in enumerate(TERMS)})
      row.update(drop_diag(row.get("term_shift", 0.0)))
      if args.profile:
        mark("diag")
        tot = sum(prof.values())
        row.update({f"prof_{k}": v / tot for k, v in prof.items()}, prof_s_per_env_step=tot / max(1, G - last_G))
        print("[prof] " + " ".join(f"{k} {100 * v / tot:.0f}%" for k, v in prof.items())
              + f"  {1e3 * tot / max(1, G - last_G):.0f} ms/env step", flush=True)
        prof.clear()
      log.write(json.dumps(row) + "\n"); log.flush()
      print(f"[ftrot] up {n_up} {G * N / 1e6:.1f}M {wall / 60:.1f} min st{row['stage']}  real r/s {row['real_reward_per_s']:+.3f} "
            f"rot {row['real_rotate']:.3f} drops/min {row['real_drops_per_env_min']:.3f}  imag r/s {row.get('imag_r', 0):+.3f} "
            f"p_drop {row.get('p_drop', 0):.4f} wm {row['wm']:.1f} contact {row['contact']:.3f} surv {row.get('surv', 0):.3f} a {row.get('a_mix', 0):.2f} "
            f"{row['s_per_update']:.2f} s/up" + (f"  V real {row['v_real']:+.3f} imag {row['v_imag']:+.3f}" if args.real_critic > 0 else "")
            + (f"  drops seen k8 {row['dd_drop_hit_k']:.2f} 1-step {row['dd_drop_hit_1']:.2f} "
            f"false {row.get('dd_keep_hit_k', 0):.3f}" if "dd_drop_hit_k" in row else ""), flush=True)
      hist_log, last_G, last_drops = [], G, stats["drops"]
    if time.time() - last_ckpt > args.ckpt_min * 60 and n_up > args.wm_only:
      last_ckpt = time.time()
      torch.save((ema if ema is not None else actor).checkpoint(update=n_up, env_steps=replay.G * N, wall_s=time.time() - t0, actor_kind=args.actor, ctx_names=cnames,
                                  **ck_extra()),
                 out / "ckpt" / f"actor_{n_up:06d}.pt")
      if ema is not None:  # the raw actor too, so the average can be judged against it
        torch.save(actor.checkpoint(update=n_up, env_steps=replay.G * N, wall_s=time.time() - t0, actor_kind=args.actor,
                                    ctx_names=cnames, **ck_extra()), out / "ckpt" / f"actor_raw_{n_up:06d}.pt")
      torch.save(dict(wm=wm.state_dict(), cfg=wcfg.__dict__, head=head.state_dict(), critic=critic.state_dict(),
                      fail=fail.state_dict(), ctx_names=cnames, c_mu=c_mu, c_sd=c_sd, meta=meta, term_shift=shift),
                 out / "ckpt" / f"models_{n_up:06d}.pt")
  torch.save((ema if ema is not None else actor).checkpoint(update=n_up, env_steps=replay.G * N, wall_s=time.time() - t0, actor_kind=args.actor, ctx_names=cnames,
                              **ck_extra()),
             out / "ckpt" / "actor_final.pt")
  if ema is not None:
    torch.save(actor.checkpoint(update=n_up, env_steps=replay.G * N, wall_s=time.time() - t0, actor_kind=args.actor,
                                ctx_names=cnames, **ck_extra()), out / "ckpt" / "actor_raw_final.pt")
  # everything --init needs (fail head + term_shift too), next to the actor so "actor_final" -> "models_final" resolves
  mfin = dict(wm=wm.state_dict(), cfg=wcfg.__dict__, head=head.state_dict(), critic=critic.state_dict(),
              fail=fail.state_dict(), ctx_names=cnames, c_mu=c_mu, c_sd=c_sd, meta=meta, term_shift=shift)
  torch.save(mfin, out / "models_final.pt")
  torch.save(mfin, out / "ckpt" / "models_final.pt")
  print("[ftrot] done")


@torch.no_grad()
def parity(replay, task, ridx, rnames, env, out):
  """The adapter's reward terms on recorded transitions (true contact signals) against the env's own."""
  G = replay.G
  a_env = min(1.0, max(0.0, (env.common_step_counter - env.cfg.rewards["rotate"].params.get("ramp_delay", 0))
                       / max(1, env.cfg.rewards["rotate"].params["ramp_steps"])))
  g = torch.arange(1, G, device=replay.device)
  gg = g[:, None].expand(-1, replay.N).reshape(-1)
  ee = torch.arange(replay.N, device=replay.device)[None].expand(len(g), -1).reshape(-1)
  r = gg % replay.L
  ok = (replay.ep_step[r, ee] >= 1) & (replay.epi[r, ee] >= 0)
  gg, ee, r = gg[ok], ee[ok], r[ok]
  s, s2, info = replay.s[r, ee], replay.s_next[r, ee], replay.info[r, ee]
  cmd = s2[:, task.S["cmd"]]
  terms, _ = task.reward_rates(s, s2, cmd, info, replay.aux[r, ee], replay.contact[r, ee], replay.plant[r, ee], a_env)
  dd = replay.done[r, ee]
  ours_drop = terms["dropped"] != 0
  env_drop = replay.rew[r, ee][:, TERMS.index("dropped")] != 0
  print(f"[parity] drop confusion: both {int((ours_drop & env_drop).sum())} only-ours {int((ours_drop & ~env_drop).sum())} "
        f"only-env {int((~ours_drop & env_drop).sum())}; terminated rows {int((dd == 1).sum())}; ours on non-terminal rows "
        f"{int((ours_drop & (dd != 1)).sum())}")
  res = {}
  for i, name in enumerate(TERMS):
    a, b = terms[name], replay.rew[r, ee][:, i]
    c = float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    res[name] = dict(env=float(b.mean()), ours=float(a.mean()), mean_abs_diff=float((a - b).abs().mean()), corr=c)
    print(f"[parity] {name:17s} env {res[name]['env']:+.4f} ours {res[name]['ours']:+.4f}  "
          f"mean|diff| {res[name]['mean_abs_diff']:.2e}  corr {c:.3f}")
  (out / "parity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
