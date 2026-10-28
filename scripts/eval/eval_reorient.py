"""Evaluate an SO(3) cube-reorientation policy: goals per minute, drops, goal timeouts and time to goal.

Rolls out the deterministic actor (mean, [-1, 1] clip) in the task's PLAY env (uniform SO(3) goals) and reports
goals/min, drops (non-timeout terminations) per hour and per 100 goals, goal timeouts per minute and the median time to
goal, plus contact-force statistics. ``--cond`` picks the plant:

  nominal            domain randomization removed, the servo at its nominal values (gear tasks: the measured LEAP
                     servo's nominal parameters, 10 ms command delay)
  train_dr           the play env's full training randomization kept (in-distribution reference)
  fric_low           pads 0.6, every other hand geom and the cube 0.3 sliding friction
  mass_x1.5          cube mass and inertia x1.5
  obsnoise_train     actor observation noise on at the training levels (the play env has it off)
  On the measured-servo (-Gear) tasks, one servo parameter at an edge of its training range (every env):
  kt_low / kt_high   torque constant 0.75 / 1.0 N m/A
  mu_low / mu_high   gearbox friction 0.45 / 0.65
  gain_low/gain_high current gain x0.9 / x1.1
  lag_short/lag_long command delay 1 / 5 physics steps (5 / 25 ms)

Usage:
  uv run python scripts/eval/eval_reorient.py --cond nominal            # 6 cm cube, the released checkpoint
  uv run python scripts/eval/eval_reorient.py --cond train_dr --num-envs 512 --steps 1200 \
    --task Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube5-S2R3-Gear --ckpt ckpts/so3/ppo_cube5_s2r3_gear.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TASK = "Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube6-S2R3-Gear"
CKPT = ROOT / "ckpts/so3/ppo_cube6_s2r3_gear.pt"
OUT = ROOT / "outputs/eval_reorient"
# the conditions offered on the command line (the rest of CONDS are building blocks of the internal studies)
PUBLIC_CONDS = ("nominal", "train_dr", "fric_low", "mass_x1.5", "obsnoise_train", "kt_low", "kt_high", "mu_low",
                "mu_high", "gain_low", "gain_high", "lag_short", "lag_long")
CONDS = ("nominal", "train_dr", "fric_x0.7", "fric_x0.5", "fric_0.6", "fric_0.4", "no_torsion", "effort_x0.7",
         "gain_x0.7", "mass_x1.5", "obsnoise_train", "obsnoise_x3", "cubelat_100", "cubelat_150", "actdelay_75",
         "actdelay_100", "lead_0.15", "realistic", "lag_10ms", "act_kp4", "act_kp4_eff2", "fricmix_real",
         "act_kp4_fricmix", "ringdip_dead", "stale_real", "real_all", "real_fixed_ring", "kp_x0.75", "kp_x1.5",
         "eff_low", "fric_low", "lag_25ms", "s2r_real", "s2r_hard", "kp_x0.4", "lag_50ms",
         "stale_cam1", "posbias25", "posnoise10", "s2r_cam1_raw", "s2r_cam1_corr", "s2r_real_bias25", "s2r_cam1_ideal",
         "kt_low", "kt_high", "mu_low", "mu_high", "gain_low", "gain_high", "lag_short", "lag_long")
#: Parts that set IdealPd parameters (Kp, Kd, effort limits) or a delay beyond the gear servo's 0-6 step buffer.
IDEAL_PD_PARTS = {"effort_x0.7", "gain_x0.7", "kp_x0.75", "kp_x1.5", "kp_x0.4", "kp_x4", "eff_x2", "eff_low",
                  "actdelay_75", "actdelay_100", "lag_50ms"}
#: The gear servo's parameters, each at an edge of its training DR (LEAP_GEAR_SERVO_2026_10_DR), as
#: ``sampling.env.pin_gear_servo`` overrides.
GEAR_PARTS = {"kt_low": dict(torque_constant=0.75), "kt_high": dict(torque_constant=1.0),
              "mu_low": dict(gear_friction=0.45), "mu_high": dict(gear_friction=0.65),
              "gain_low": dict(current_gain_scale=0.9), "gain_high": dict(current_gain_scale=1.1)}
#: Command delays [physics steps, 5 ms]. Default: 10 (IdealPd), 2 (the S2R tasks and the gear servo).
LAGS = {"actdelay_75": 15, "actdelay_100": 20, "lag_10ms": 2, "lag_25ms": 5, "lag_50ms": 10, "lag_short": 1,
        "lag_long": 5}
#: cam1's line of sight to the cube at inhand_ready, in the training (mj world) frame: real world
#: (0.492, 0.548, -0.676) rotated by palm_quat_mjw * palm_w^-1 (2026-10-06 runs).
CAM1_RAY_MJ = (-0.334, -0.657, -0.676)


class TorchActor:
  """rsl-rl 5.4 actor (normalizer eps 1e-2 + ELU MLP), deterministic mean clipped to [-1, 1]."""

  def __init__(self, ckpt: str, device: str):
    sd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
    self.mean, self.std = sd["obs_normalizer._mean"].reshape(-1), sd["obs_normalizer._std"].reshape(-1)
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    self.layers = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in idx]

  @torch.no_grad()
  def __call__(self, x):
    x = (x - self.mean) / (self.std + 1e-2)
    for i, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if i < len(self.layers) - 1:
        x = torch.nn.functional.elu(x)
    return x.clamp(-1.0, 1.0)


def parts(cond: str, gear: bool = False) -> set[str]:
  """The perturbations a condition applies. ``gear``: the task runs the measured servo, so
  ``s2r_hard`` takes the servo's weak edges instead of the IdealPd ones; parts of the other plant
  are refused."""
  P = _parts(cond, gear)
  wrong = P & (IDEAL_PD_PARTS if gear else set(GEAR_PARTS))
  if wrong:
    raise ValueError(f"condition {cond!r} sets {sorted(wrong)}: "
                     f"{'IdealPd' if gear else 'gear-servo'} parameters, but the task runs the "
                     f"{'gear' if gear else 'IdealPd'} servo")
  return P


def _parts(cond: str, gear: bool) -> set[str]:
  if cond == "realistic":
    return {"fric_x0.7", "lead_0.15", "cubelat_100", "obsnoise_train"}
  if cond == "act_kp4":
    return {"kp_x4", "lag_10ms"}
  if cond == "act_kp4_eff2":
    return {"kp_x4", "eff_x2", "lag_10ms"}
  if cond == "act_kp4_fricmix":
    return {"kp_x4", "lag_10ms", "fricmix_real"}
  if cond == "real_all":
    return {"kp_x4", "lag_10ms", "fricmix_real", "ringdip_dead", "stale_real"}
  if cond == "real_fixed_ring":
    return {"kp_x4", "lag_10ms", "fricmix_real", "stale_real"}
  if cond == "s2r_real":
    return {"fricmix_real", "stale_real", "lag_10ms"}
  if cond == "s2r_hard":
    if gear:
      return {"kt_low", "mu_high", "gain_low", "fric_low", "stale_real", "lag_long"}
    return {"kp_x1.5", "eff_low", "fric_low", "stale_real", "lag_25ms"}
  if cond == "s2r_cam1_raw":
    return {"fricmix_real", "lag_10ms", "stale_cam1", "posbias25"}
  if cond == "s2r_cam1_corr":
    return {"fricmix_real", "lag_10ms", "stale_cam1", "posnoise10"}
  if cond == "s2r_cam1_ideal":
    return {"fricmix_real", "lag_10ms", "stale_cam1"}
  if cond == "s2r_real_bias25":
    return {"fricmix_real", "stale_real", "lag_10ms", "posbias25"}
  return {cond}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--cond", default="nominal", choices=PUBLIC_CONDS)
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=1200)
  ap.add_argument("--warmup", type=int, default=60)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out-dir", default=str(OUT))
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--ckpt", default=str(CKPT))
  ap.add_argument("--tag", default=None, help="output file stem (default: the condition)")
  a = ap.parse_args()

  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import RecomputeLevel
  from mjlab.tasks.registry import load_env_cfg
  from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import LeapGearServoActuator, actuator_effort_limits
  from in_hand_rotation_mjlab.tasks.hand_cube.mdp.commands import GOAL
  from in_hand_rotation_mjlab.tasks.hand_cube.plant_utils import gear_servo_cfg, get_lags, pin_gear_servo, set_lags, strip_plant_dr

  cfg = load_env_cfg(a.task, play=True)
  ckpt = a.ckpt
  gear = gear_servo_cfg(cfg) is not None
  P = parts(a.cond, gear)
  s2r_task = "-S2R" in a.task
  dr = "train_dr" in P
  lag = LAGS.get(next(iter(P & set(LAGS)), ""), 2 if s2r_task or gear else 10)
  if not dr:
    strip_plant_dr(cfg, extra=("dr_split_contact_friction",))
    if gear:  # nominal, then this condition's servo parameters; the reset event keeps them in force
      pin_gear_servo(cfg, lag=lag, **{k: v for p in P & set(GEAR_PARTS) for k, v in GEAR_PARTS[p].items()})
  actor_obs = cfg.observations["actor"]
  notes = []
  if "obsnoise_train" in P or "obsnoise_x3" in P:
    k = 3.0 if "obsnoise_x3" in P else 1.0
    actor_obs.enable_corruption = True
    for name, t in actor_obs.terms.items():
      if t.noise is not None:
        t.noise.n_min *= k
        t.noise.n_max *= k
        notes.append(f"noise {name} U({t.noise.n_min:.4f},{t.noise.n_max:.4f})")
  for tag, obs_lag in (("cubelat_100", 2), ("cubelat_150", 3)):
    if tag in P:
      for name in ("cube_pos", "cube_rot", "rel_rot"):
        t = actor_obs.terms[name]
        t.delay_min_lag = t.delay_max_lag = obs_lag
        t.delay_hold_prob, t.delay_update_period = 0.0, 0
      notes.append(f"cube pose obs lag {obs_lag} steps")
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
  B, dev = a.num_envs, env.device
  robot, cube = env.scene["robot"], env.scene["cube"]
  term = env.action_manager.get_term("joint_pos")
  cmd = env.command_manager.get_term(GOAL)
  sim = env.sim
  mjm = sim.mj_model
  import mujoco

  gnames = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(mjm.ngeom)]
  hand_g = [i for i, n in enumerate(gnames) if n.startswith("robot/")]
  cube_g = [gnames.index("cube/cube_geom")]
  tip_g = [i for i, n in enumerate(gnames) if n.startswith("robot/") and n.endswith("_tip")]
  assert len(hand_g) == 71 and len(tip_g) == 4

  # --- model perturbations (assign from the compiled defaults, after per-world expansion)
  fric_scale = {"fric_x0.7": 0.7, "fric_x0.5": 0.5}
  fric_abs = {"fric_0.6": 0.6, "fric_0.4": 0.4}
  if P & (set(fric_scale) | set(fric_abs) | {"no_torsion", "fricmix_real", "fric_low"}):
    sim.expand_model_fields(("geom_friction",))
    f = sim.get_default_field("geom_friction").clone()  # [ngeom, 3]
    ids = torch.tensor(hand_g + cube_g, device=dev)
    for k, s in fric_scale.items():
      if k in P:
        f[ids, 0] *= s
    for k, v in fric_abs.items():
      if k in P:
        f[ids, 0] = v
    if "no_torsion" in P:
      f[torch.tensor(tip_g, device=dev), 1:] = 0.0
    if "fricmix_real" in P:
      box = torch.tensor([g for g in hand_g if g not in tip_g], device=dev)
      f[box, 0] = 0.4
      f[torch.tensor(cube_g, device=dev), 0] = 0.5
    if "fric_low" in P:
      box = torch.tensor([g for g in hand_g if g not in tip_g], device=dev)
      f[box, 0] = 0.3
      f[torch.tensor(cube_g, device=dev), 0] = 0.3
      f[torch.tensor(tip_g, device=dev), 0] = 0.6
    sim.model.geom_friction[:] = f.unsqueeze(0).expand(B, -1, -1)
    notes.append(f"geom_friction tip {f[tip_g[0]].tolist()} box {f[hand_g[0]].tolist()} cube {f[cube_g[0]].tolist()}")
  if "mass_x1.5" in P:
    sim.expand_model_fields(("body_mass", "body_inertia"))
    bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "cube/cube")
    sim.model.body_mass[:, bid] = 1.5 * sim.get_default_field("body_mass")[bid]
    sim.model.body_inertia[:, bid] = 1.5 * sim.get_default_field("body_inertia")[bid]
    sim.recompute_constants(RecomputeLevel.set_const)
    notes.append(f"cube mass {float(sim.model.body_mass[0, bid]):.4f} kg")
  groups = robot._fused_actuator_group._groups  # IdealPd (none on the gear servo)
  gear_act = next((x for x in robot.actuators if isinstance(x, LeapGearServoActuator)), None)
  if "effort_x0.7" in P:
    for g in groups:
      g.params["force_limit"].mul_(0.7)
    sim.expand_model_fields(("actuator_forcerange",))
    sim.model.actuator_forcerange[:] = (0.7 * sim.get_default_field("actuator_forcerange")).unsqueeze(0).expand(B, -1, -1)
  if "gain_x0.7" in P:
    for g in groups:
      g.params["stiffness"].mul_(0.7)
      g.params["damping"].mul_(0.7)
  for k, v in (("kp_x0.75", 0.75), ("kp_x1.5", 1.5), ("kp_x0.4", 0.4)):
    if k in P:
      for g in groups:
        g.params["stiffness"].mul_(v)
  if "eff_low" in P:
    for g in groups:
      g.params["force_limit"].fill_(0.28)
    sim.expand_model_fields(("actuator_forcerange",))
    fr = sim.get_default_field("actuator_forcerange").clone()
    fr[:, 0], fr[:, 1] = -0.28, 0.28
    sim.model.actuator_forcerange[:] = fr.unsqueeze(0).expand(B, -1, -1)
  if "kp_x4" in P:
    for g in groups:
      g.params["stiffness"].mul_(4.0)
  if "eff_x2" in P:
    for g in groups:
      g.params["force_limit"].mul_(2.0)
    sim.expand_model_fields(("actuator_forcerange",))
    sim.model.actuator_forcerange[:] = (2.0 * sim.get_default_field("actuator_forcerange")).unsqueeze(0).expand(B, -1, -1)
  if "ringdip_dead" in P:
    jid = list(robot.joint_names).index("rf_dip")
    for g in groups:
      hit = (g.target_ids == jid).nonzero().flatten()
      for k in ("stiffness", "damping", "force_limit"):
        g.params[k][:, hit] = 0.0
    if gear_act is not None:  # the reset event scales the defaults
      hit = (gear_act.target_ids == jid).nonzero().flatten()
      for k in ("current_gain", "current_damping"):
        gear_act.defaults[k][:, hit] = 0.0
        getattr(gear_act, k)[:, hit] = 0.0
    notes.append("rf_dip limp")
  if "lead_0.15" in P:
    term._max_lead = torch.full_like(term._max_lead, 0.15)

  def plant(row):
    """The servo parameters in force in env ``row`` (IdealPd: effort and Kp per joint)."""
    if gear_act is None:
      return dict(effort=torch.cat([g.params["force_limit"] for g in groups], -1).reshape(B, -1)[row].tolist(),
                  kp=torch.cat([g.params["stiffness"] for g in groups], -1).reshape(B, -1)[row].tolist())
    return dict(lag=int(get_lags(env)[row]), **{k: getattr(gear_act, k)[row].tolist() for k in (
      "torque_constant", "gear_friction", "current_gain", "current_damping", "dry_friction")})

  def pin_lags(ids=None):
    if not dr:
      n = B if ids is None else len(ids)
      set_lags(env, torch.full((n,), lag, device=dev, dtype=torch.long), ids)

  obs, _ = env.reset()
  pin_lags()
  plant0 = {k: np.round(v, 4).tolist() for k, v in plant(0).items()}
  print(f"[sens] {a.cond}: {'gear' if gear else 'IdealPd'} {plant0} lag {lag} "
        f"lead {float(term._max_lead.flatten()[0]):.3f} notes {notes}", flush=True)

  actor = TorchActor(ckpt, dev)
  om = env.observation_manager
  names, dims = om.active_terms["actor"], om.group_obs_term_dim["actor"]
  off, cube_cols = 0, []
  for n_, d_ in zip(names, dims):
    w = int(np.prod(d_))
    if n_ in ("cube_pos", "cube_rot", "rel_rot"):
      cube_cols += list(range(off, off + w))
    off += w
  pos_cols, off = [], 0
  for n_, d_ in zip(names, dims):
    if n_ == "cube_pos":
      pos_cols = list(range(off, off + int(np.prod(d_))))
    off += int(np.prod(d_))
  cube_cols = torch.tensor(cube_cols, device=dev)
  pos_cols = torch.tensor(pos_cols, device=dev)
  stale = dict(held=torch.zeros(B, dtype=torch.bool, device=dev), val=None)
  gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
  chain = (0.118, 0.839) if "stale_real" in P else (0.02, 0.70) if "stale_cam1" in P else None
  bias = (0.025 * torch.tensor(CAM1_RAY_MJ, device=dev)).repeat(len(pos_cols) // 3)

  def observe(o):
    if "posbias25" in P or "posnoise10" in P:
      o = o.clone()
      if "posbias25" in P:
        o[:, pos_cols] += bias
      if "posnoise10" in P:
        o[:, pos_cols] += (torch.rand(B, len(pos_cols), device=dev, generator=gen) * 2 - 1) * 0.010
    if chain is None:
      return o
    if stale["val"] is None:
      stale["val"] = o[:, cube_cols].clone()
    u = torch.rand(B, device=dev, generator=gen)
    stale["held"] = torch.where(stale["held"], u < chain[1], u < chain[0])
    fresh = ~stale["held"]
    stale["val"][fresh] = o[fresh][:, cube_cols]
    o = o.clone()
    o[:, cube_cols] = stale["val"]
    return o

  # goal-change classification (success vs timeout), as record_contacts.py
  ev = dict(phase=None, succ=torch.zeros(B, dtype=torch.bool, device=dev), tout=torch.zeros(B, dtype=torch.bool, device=dev))
  _orig_resample, _orig_update, _orig_compute = cmd._resample, cmd._update_command, cmd.compute

  def _resample(env_ids):
    if ev["phase"] == "update":
      ev["succ"][env_ids] = True
    elif ev["phase"] == "compute":
      ev["tout"][env_ids] = True
    _orig_resample(env_ids)

  def _update():
    ev["phase"] = "update"
    _orig_update()
    ev["phase"] = "compute"

  def _compute(dt):
    ev["phase"] = "compute"
    _orig_compute(dt)
    ev["phase"] = None

  cmd._resample, cmd._update_command, cmd.compute = _resample, _update, _compute
  tm = env.termination_manager
  tnames = list(tm.active_terms)

  t0 = time.time()
  n_succ = n_tout = 0
  drops = {n: 0 for n in tnames}
  last_change = torch.zeros(B, dtype=torch.long, device=dev)
  ttg = []
  peak_f, tsat, spin, jspeed = [], [], [], []
  with torch.no_grad():
    for t in range(a.warmup + a.steps):
      ev["succ"].zero_()
      ev["tout"].zero_()
      obs, _, term_b, tout_b, _ = env.step(actor(observe(obs["actor"])))
      done = (term_b | tout_b).nonzero().flatten()
      if len(done) and stale["val"] is not None:
        stale["held"][done] = False
        stale["val"][done] = obs["actor"][done][:, cube_cols]
      if len(done):
        pin_lags(done)
      if t < a.warmup:
        last_change[:] = t
        continue
      s, to = ev["succ"].clone(), ev["tout"].clone()
      n_succ += int(s.sum())
      n_tout += int(to.sum())
      for n in tnames:
        if n != "time_out":
          drops[n] += int((tm.get_term(n) & term_b).sum())
      ids = s.nonzero().flatten()
      if len(ids):
        ttg.append(((t - last_change[ids]) * env.step_dt).cpu().numpy())
      changed = s | to | term_b
      last_change[changed] = t
      f = env.scene.sensors["safety_hand_cube"].data.force.norm(dim=-1).amax(-1)
      peak_f.append(f.cpu().numpy())
      eff_now = actuator_effort_limits(robot)
      tsat.append(((robot.data.actuator_force.abs() / eff_now.clamp(min=1e-6)) > 0.98).float().mean().item())
      spin.append(cube.data.root_link_ang_vel_w.norm(dim=-1).cpu().numpy())
      jspeed.append(robot.data.joint_vel.abs().amax(-1).cpu().numpy())
  wall = time.time() - t0
  minutes = B * a.steps * env.step_dt / 60.0
  ttg = np.concatenate(ttg) if ttg else np.zeros(0)
  pf = np.concatenate(peak_f)
  n_drop = sum(drops.values())
  res = dict(
    cond=a.cond, parts=sorted(P), task=a.task, servo="gear" if gear else "ideal_pd", checkpoint=ckpt, num_envs=B,
    steps=a.steps,
    warmup=a.warmup, seed=a.seed, env_minutes=minutes, action_lag_substeps=None if dr else lag,
    max_lead=float(term._max_lead.flatten()[0]), notes=notes,
    goals=n_succ, goals_per_min=n_succ / minutes, goal_timeouts=n_tout, timeouts_per_min=n_tout / minutes,
    drops=n_drop, drops_by_reason=drops, drops_per_env_hour=n_drop / minutes * 60.0,
    drops_per_100_goals=100.0 * n_drop / max(n_succ, 1),
    time_to_goal_median_s=float(np.median(ttg)) if len(ttg) else None, time_to_goal_n=int(len(ttg)),
    hand_cube_force_end_of_step_N=dict(median=float(np.median(pf)), p99=float(np.percentile(pf, 99)),
                                       p999=float(np.percentile(pf, 99.9))),
    torque_saturated_frac=float(np.mean(tsat)),
    cube_spin_rad_s=dict(median=float(np.median(np.concatenate(spin))), p99=float(np.percentile(np.concatenate(spin), 99))),
    max_joint_speed_rad_s=dict(median=float(np.median(np.concatenate(jspeed))), p99=float(np.percentile(np.concatenate(jspeed), 99))),
    wall_s=wall,
    end_check=dict(  # perturbations still in force after all resets
      **plant(-1),
      tip_friction=sim.model.geom_friction[-1, tip_g[0]].tolist(),
      box_friction=sim.model.geom_friction[-1, hand_g[0]].tolist(),
      cube_friction=sim.model.geom_friction[-1, cube_g[0]].tolist(),
      cube_mass=float(sim.model.body_mass[-1, mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "cube/cube")]),
    ),
  )
  print("[sens] RESULT " + json.dumps(res), flush=True)
  out = Path(a.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  (out / f"{a.tag or a.cond}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
