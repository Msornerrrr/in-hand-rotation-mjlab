"""Fit mjwarp hand dynamics to real logs by replaying commanded joint positions.

Finger-focused system ID for LEAP hand Ideal PD actuators.

For the selected finger, this fits 5 parameters per joint (20 params/finger):
  - stiffness
  - damping
  - effort_limit
  - armature
  - frictionloss

All parameters are sampled as multiplicative scales around baseline values from
configured LEAP constants/articulation. A global integer actuator delay
(control steps) is also optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg


@dataclass(frozen=True)
class SysIdFitCliConfig:
  log_file: str
  output_file: str | None = None
  device: str | None = None
  play: bool = True
  command_hz: float = 20.0
  max_shift_steps: int = 4
  random_trials: int = 128
  seed: int = 0
  parallel_envs: int = 8192
  optimizer: Literal["random", "cem"] = "cem"

  # Finger-specific calibration target.
  finger: Literal["index", "middle", "ring", "thumb", "all"] = "index"

  # Global actuator delay (not per-joint), in control steps.
  optimize_delay: bool = True
  delay_min_steps: int = 0
  delay_max_steps: int = 5

  # Per-joint search ranges (multipliers around baseline values).
  # Defaults centered at 1.0 around LEAP constants baseline.
  stiffness_scale_min: float = 0.7
  stiffness_scale_max: float = 1.4
  damping_scale_min: float = 0.7
  damping_scale_max: float = 1.4
  effort_scale_min: float = 0.85
  effort_scale_max: float = 1.15
  armature_scale_min: float = 0.6
  armature_scale_max: float = 1.8
  friction_scale_min: float = 0.3
  friction_scale_max: float = 2.2

  # Joint-order/sign remap from hardware log to sim convention.
  # Swaps are needed for index/middle/ring and thumb negate.
  apply_joint_swap_remap: bool = True
  apply_thumb_cmc_negate: bool = True

  # CEM options.
  cem_population: int | None = None
  cem_generations: int = 12
  cem_elite_frac: float = 0.1
  cem_alpha: float = 0.5
  cem_init_std_frac: float = 0.35
  cem_min_std_frac: float = 0.05


def _load_npz_log(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  in_path = Path(path).expanduser().resolve()
  if not in_path.exists():
    raise FileNotFoundError(f"Log file not found: {in_path}")
  with np.load(str(in_path), allow_pickle=True) as data:
    cmd = np.asarray(data["hand_cmd"], dtype=np.float64)
    state = np.asarray(data["hand_state"], dtype=np.float64)
    if cmd.ndim != 2 or state.ndim != 2:
      raise ValueError(
        f"Expected hand_cmd/hand_state as 2D arrays, got {cmd.shape} and {state.shape}."
      )
    if cmd.shape != state.shape:
      raise ValueError(
        f"hand_cmd and hand_state shape mismatch: {cmd.shape} vs {state.shape}."
      )
    if "time" in data:
      t = np.asarray(data["time"], dtype=np.float64).reshape(-1)
      if t.shape[0] != cmd.shape[0]:
        raise ValueError(
          f"time length ({t.shape[0]}) does not match samples ({cmd.shape[0]})."
        )
    else:
      t = np.arange(cmd.shape[0], dtype=np.float64)
  return t, cmd, state


def _apply_leap_left_remap(
  x: np.ndarray,
  *,
  apply_swap: bool,
  negate_thumb_cmc: bool,
) -> np.ndarray:
  """Map LEAP hardware joint order/sign to sim policy order.

  - swap (0,1), (4,5), (8,9)
  - optionally negate index 12 (thumb CMC)
  """
  if x.ndim != 2 or x.shape[1] != 16:
    raise ValueError(f"LEAP-left remap expects shape (N, 16). Got {x.shape}.")

  if apply_swap:
    idx = np.asarray(
      [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15],
      dtype=np.int64,
    )
    y = x[:, idx].copy()
  else:
    y = x.copy()

  if negate_thumb_cmc:
    y[:, 12] *= -1.0

  return y


def _resample_to_hz(
  time_s: np.ndarray,
  cmd: np.ndarray,
  state: np.ndarray,
  hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  if hz <= 0.0:
    raise ValueError(f"command_hz must be > 0, got {hz}.")

  t0 = float(time_s[0])
  t1 = float(time_s[-1])
  dt = 1.0 / hz
  if t1 <= t0 + 1e-9:
    t_grid = np.arange(cmd.shape[0], dtype=np.float64) * dt
  else:
    n = int(np.floor((t1 - t0) / dt)) + 1
    t_grid = t0 + np.arange(n, dtype=np.float64) * dt

  cmd_rs = np.empty((t_grid.shape[0], cmd.shape[1]), dtype=np.float64)
  state_rs = np.empty((t_grid.shape[0], state.shape[1]), dtype=np.float64)
  for j in range(cmd.shape[1]):
    cmd_rs[:, j] = np.interp(t_grid, time_s, cmd[:, j])
    state_rs[:, j] = np.interp(t_grid, time_s, state[:, j])
  return t_grid, cmd_rs, state_rs


def _best_shift_rmse(
  sim: np.ndarray,
  real: np.ndarray,
  max_shift: int,
  eval_joint_ids: np.ndarray | None = None,
) -> tuple[float, int, np.ndarray]:
  if sim.shape != real.shape:
    raise ValueError(f"Shape mismatch: sim={sim.shape}, real={real.shape}.")

  best_rmse = float("inf")
  best_shift = 0
  ncols = sim.shape[1] if eval_joint_ids is None else len(eval_joint_ids)
  best_per_joint = np.full((ncols,), np.inf, dtype=np.float64)

  for shift in range(-max_shift, max_shift + 1):
    if shift > 0:
      sim_seg = sim[shift:]
      real_seg = real[: sim_seg.shape[0]]
    elif shift < 0:
      real_seg = real[-shift:]
      sim_seg = sim[: real_seg.shape[0]]
    else:
      sim_seg = sim
      real_seg = real

    if sim_seg.shape[0] < 2:
      continue

    if eval_joint_ids is None:
      err = sim_seg - real_seg
    else:
      err = sim_seg[:, eval_joint_ids] - real_seg[:, eval_joint_ids]

    per_joint = np.sqrt(np.mean(err * err, axis=0))
    rmse = float(np.sqrt(np.mean(err * err)))
    if rmse < best_rmse:
      best_rmse = rmse
      best_shift = shift
      best_per_joint = per_joint

  return best_rmse, best_shift, best_per_joint


def _finger_joint_ids(finger: str) -> np.ndarray:
  mapping = {
    "index": np.asarray([0, 1, 2, 3], dtype=np.int64),
    "middle": np.asarray([4, 5, 6, 7], dtype=np.int64),
    "ring": np.asarray([8, 9, 10, 11], dtype=np.int64),
    "thumb": np.asarray([12, 13, 14, 15], dtype=np.int64),
    "all": np.asarray(list(range(16)), dtype=np.int64),
  }
  return mapping[finger]


def _strip_randomization(env_cfg) -> None:
  # Keep deterministic reset behavior only.
  keep = {}
  for name, term in env_cfg.events.items():
    if term is None:
      continue
    if name.startswith("dr_"):
      continue
    if name == "reset_from_grasp_cache":
      continue
    if bool(getattr(term, "domain_randomization", False)):
      continue
    func_name = getattr(term.func, "__name__", "")
    if func_name.startswith("randomize_"):
      continue
    keep[name] = term
  env_cfg.events = keep

  if "reset_robot_joints" in env_cfg.events:
    term = env_cfg.events["reset_robot_joints"]
    term.params["position_range"] = (0.0, 0.0)
    term.params["velocity_range"] = (0.0, 0.0)


def _disable_cfg_actuator_delay(env_cfg) -> int:
  """Disable delayed-actuator wrappers in cfg for sysid.

  SysID explicitly optimizes global delay in command replay. Leaving actuator
  wrappers delayed would double-count latency.
  """
  scene = getattr(env_cfg, "scene", None)
  entities = getattr(scene, "entities", None)
  if not isinstance(entities, dict) or "robot" not in entities:
    return 0
  robot_cfg = entities["robot"]
  articulation = getattr(robot_cfg, "articulation", None)
  actuators = getattr(articulation, "actuators", None)
  if actuators is None:
    return 0

  changed = 0
  for act_cfg in actuators:
    if hasattr(act_cfg, "delay_min_lag") and hasattr(act_cfg, "delay_max_lag"):
      if int(getattr(act_cfg, "delay_min_lag")) != 0 or int(getattr(act_cfg, "delay_max_lag")) != 0:
        changed += 1
      setattr(act_cfg, "delay_min_lag", 0)
      setattr(act_cfg, "delay_max_lag", 0)
      if hasattr(act_cfg, "delay_hold_prob"):
        setattr(act_cfg, "delay_hold_prob", 1.0)
      if hasattr(act_cfg, "delay_update_period"):
        setattr(act_cfg, "delay_update_period", 0)
  return changed


def _build_env(task_id: str, play: bool, device: str, num_envs: int) -> ManagerBasedRlEnv:
  env_cfg = load_env_cfg(task_id, play=play)
  env_cfg.scene.num_envs = int(num_envs)
  _strip_randomization(env_cfg)
  disabled = _disable_cfg_actuator_delay(env_cfg)
  if disabled > 0:
    print(
      f"[INFO] Disabled delayed-actuator wrappers for sysid on {disabled} actuator cfg entries."
    )
  return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)


def _unwrap_actuator(act):
  """Return base actuator if this is a delayed wrapper."""
  return act.base_actuator if hasattr(act, "base_actuator") else act


def _extract_joint_baselines(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor | np.ndarray]:
  """Extract baseline per-joint dynamics values from initialized env."""
  robot = env.scene["robot"]
  nj = len(robot.joint_names)
  dof_ids = robot.indexing.joint_v_adr.to(device=env.device, dtype=torch.long)

  base: dict[str, torch.Tensor | np.ndarray] = {
    "joint_names": np.asarray(robot.joint_names, dtype=np.str_),
    "dof_ids": dof_ids,
    "armature": env.sim.model.dof_armature[0:1, dof_ids].clone(),
    "frictionloss": env.sim.model.dof_frictionloss[0:1, dof_ids].clone(),
    "stiffness": torch.zeros((1, nj), device=env.device, dtype=torch.float32),
    "damping": torch.zeros((1, nj), device=env.device, dtype=torch.float32),
    "effort_limit": torch.zeros((1, nj), device=env.device, dtype=torch.float32),
  }

  # Pull per-joint gains/limits from custom IdealPd actuators.
  for actuator in robot.actuators:
    act = _unwrap_actuator(actuator)
    target_ids = act.target_ids.to(device=env.device, dtype=torch.long)
    if hasattr(act, "stiffness") and getattr(act, "stiffness") is not None:
      base["stiffness"][:, target_ids] = act.stiffness[0:1, : len(target_ids)].clone()
    if hasattr(act, "damping") and getattr(act, "damping") is not None:
      base["damping"][:, target_ids] = act.damping[0:1, : len(target_ids)].clone()
    if hasattr(act, "force_limit") and getattr(act, "force_limit") is not None:
      base["effort_limit"][:, target_ids] = act.force_limit[0:1, : len(target_ids)].clone()

  if torch.any(base["stiffness"] <= 0.0):
    raise RuntimeError("Failed to resolve positive per-joint baseline stiffness.")
  if torch.any(base["damping"] <= 0.0):
    raise RuntimeError("Failed to resolve positive per-joint baseline damping.")
  if torch.any(base["effort_limit"] <= 0.0):
    raise RuntimeError("Failed to resolve positive per-joint baseline effort_limit.")

  return base


def _identity_scales(nj: int) -> dict[str, np.ndarray]:
  one = np.ones((nj,), dtype=np.float64)
  return {
    "stiffness_scale": one.copy(),
    "damping_scale": one.copy(),
    "effort_scale": one.copy(),
    "armature_scale": one.copy(),
    "friction_scale": one.copy(),
  }


def _sample_scales(
  rng: np.random.Generator,
  cfg: SysIdFitCliConfig,
  nj: int,
  train_joint_ids: np.ndarray,
) -> dict[str, np.ndarray]:
  # Only sample selected finger joints; keep others fixed at 1.0.
  scales = _identity_scales(nj)
  ids = train_joint_ids
  scales["stiffness_scale"][ids] = rng.uniform(
    cfg.stiffness_scale_min, cfg.stiffness_scale_max, size=(len(ids),)
  )
  scales["damping_scale"][ids] = rng.uniform(
    cfg.damping_scale_min, cfg.damping_scale_max, size=(len(ids),)
  )
  scales["effort_scale"][ids] = rng.uniform(
    cfg.effort_scale_min, cfg.effort_scale_max, size=(len(ids),)
  )
  scales["armature_scale"][ids] = rng.uniform(
    cfg.armature_scale_min, cfg.armature_scale_max, size=(len(ids),)
  )
  scales["friction_scale"][ids] = rng.uniform(
    cfg.friction_scale_min, cfg.friction_scale_max, size=(len(ids),)
  )
  return scales


def _sample_delay(
  rng: np.random.Generator,
  *,
  optimize_delay: bool,
  min_steps: int,
  max_steps: int,
) -> int:
  if not optimize_delay:
    return 0
  if min_steps > max_steps:
    raise ValueError(f"Invalid delay range: ({min_steps}, {max_steps})")
  return int(rng.integers(min_steps, max_steps + 1))


def _continuous_bounds(
  cfg: SysIdFitCliConfig, train_joint_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  n = len(train_joint_ids)
  lows = np.concatenate(
    [
      np.full((n,), cfg.stiffness_scale_min, dtype=np.float64),
      np.full((n,), cfg.damping_scale_min, dtype=np.float64),
      np.full((n,), cfg.effort_scale_min, dtype=np.float64),
      np.full((n,), cfg.armature_scale_min, dtype=np.float64),
      np.full((n,), cfg.friction_scale_min, dtype=np.float64),
    ]
  )
  highs = np.concatenate(
    [
      np.full((n,), cfg.stiffness_scale_max, dtype=np.float64),
      np.full((n,), cfg.damping_scale_max, dtype=np.float64),
      np.full((n,), cfg.effort_scale_max, dtype=np.float64),
      np.full((n,), cfg.armature_scale_max, dtype=np.float64),
      np.full((n,), cfg.friction_scale_max, dtype=np.float64),
    ]
  )
  return lows, highs


def _vector_to_scales(
  vector: np.ndarray, *, nj: int, train_joint_ids: np.ndarray
) -> dict[str, np.ndarray]:
  n = len(train_joint_ids)
  if vector.shape != (5 * n,):
    raise ValueError(f"Expected vector shape ({5 * n},), got {vector.shape}.")
  scales = _identity_scales(nj)
  cursor = 0
  for key in (
    "stiffness_scale",
    "damping_scale",
    "effort_scale",
    "armature_scale",
    "friction_scale",
  ):
    scales[key][train_joint_ids] = vector[cursor : cursor + n]
    cursor += n
  return scales


def _sample_cem_population(
  rng: np.random.Generator,
  mean: np.ndarray,
  std: np.ndarray,
  low: np.ndarray,
  high: np.ndarray,
  population: int,
) -> np.ndarray:
  z = rng.standard_normal((population, mean.shape[0]))
  samples = mean.reshape(1, -1) + z * std.reshape(1, -1)
  np.clip(samples, low.reshape(1, -1), high.reshape(1, -1), out=samples)
  return samples


def _delay_values_and_probs(cfg: SysIdFitCliConfig) -> tuple[np.ndarray, np.ndarray]:
  if not cfg.optimize_delay:
    return np.asarray([0], dtype=np.int64), np.asarray([1.0], dtype=np.float64)
  if cfg.delay_min_steps > cfg.delay_max_steps:
    raise ValueError(f"Invalid delay range: ({cfg.delay_min_steps}, {cfg.delay_max_steps})")
  values = np.arange(cfg.delay_min_steps, cfg.delay_max_steps + 1, dtype=np.int64)
  probs = np.full((values.shape[0],), 1.0 / float(values.shape[0]), dtype=np.float64)
  return values, probs


def _sample_delay_batch(
  rng: np.random.Generator,
  delay_values: np.ndarray,
  delay_probs: np.ndarray,
  population: int,
) -> np.ndarray:
  if delay_values.shape[0] == 1:
    return np.full((population,), int(delay_values[0]), dtype=np.int64)
  return rng.choice(delay_values, size=(population,), p=delay_probs).astype(np.int64)


def _update_delay_probs(
  current_probs: np.ndarray,
  delay_values: np.ndarray,
  elite_delays: np.ndarray,
  alpha: float,
) -> np.ndarray:
  if delay_values.shape[0] == 1:
    return current_probs.copy()
  counts = np.zeros_like(current_probs)
  delay_to_idx = {int(v): i for i, v in enumerate(delay_values.tolist())}
  for d in elite_delays.tolist():
    counts[delay_to_idx[int(d)]] += 1.0
  elite_probs = counts + 1e-3
  elite_probs /= np.sum(elite_probs)
  updated = (1.0 - alpha) * current_probs + alpha * elite_probs
  updated = np.clip(updated, 1e-6, None)
  updated /= np.sum(updated)
  return updated


def _summarize_scale(scales: np.ndarray) -> str:
  return (
    f"mean={float(np.mean(scales)):.3f}, "
    f"min={float(np.min(scales)):.3f}, "
    f"max={float(np.max(scales)):.3f}"
  )


def _candidate_summary(scales: dict[str, np.ndarray], train_joint_ids: np.ndarray) -> str:
  ids = train_joint_ids
  return (
    f"stiff({_summarize_scale(scales['stiffness_scale'][ids])}) "
    f"damp({_summarize_scale(scales['damping_scale'][ids])}) "
    f"eff({_summarize_scale(scales['effort_scale'][ids])}) "
    f"arm({_summarize_scale(scales['armature_scale'][ids])}) "
    f"fric({_summarize_scale(scales['friction_scale'][ids])})"
  )


def _build_scales_batch(
  candidates: list[dict[str, object]],
  start: int,
  batch_envs: int,
  nj: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
  scales_batch = {
    "stiffness_scale": np.ones((batch_envs, nj), dtype=np.float64),
    "damping_scale": np.ones((batch_envs, nj), dtype=np.float64),
    "effort_scale": np.ones((batch_envs, nj), dtype=np.float64),
    "armature_scale": np.ones((batch_envs, nj), dtype=np.float64),
    "friction_scale": np.ones((batch_envs, nj), dtype=np.float64),
  }
  delay_batch = np.zeros((batch_envs,), dtype=np.int64)

  active = min(batch_envs, len(candidates) - start)
  for env_id in range(active):
    candidate = candidates[start + env_id]
    scales = candidate["scales"]
    scales_batch["stiffness_scale"][env_id] = scales["stiffness_scale"]
    scales_batch["damping_scale"][env_id] = scales["damping_scale"]
    scales_batch["effort_scale"][env_id] = scales["effort_scale"]
    scales_batch["armature_scale"][env_id] = scales["armature_scale"]
    scales_batch["friction_scale"][env_id] = scales["friction_scale"]
    delay_batch[env_id] = int(candidate["delay_steps"])

  # Fill unused env slots with the first active candidate to keep sim stable.
  if active > 0 and active < batch_envs:
    for env_id in range(active, batch_envs):
      scales_batch["stiffness_scale"][env_id] = scales_batch["stiffness_scale"][0]
      scales_batch["damping_scale"][env_id] = scales_batch["damping_scale"][0]
      scales_batch["effort_scale"][env_id] = scales_batch["effort_scale"][0]
      scales_batch["armature_scale"][env_id] = scales_batch["armature_scale"][0]
      scales_batch["friction_scale"][env_id] = scales_batch["friction_scale"][0]
      delay_batch[env_id] = int(delay_batch[0])

  return scales_batch, delay_batch, active


def _apply_joint_params(
  env: ManagerBasedRlEnv,
  *,
  base: dict[str, torch.Tensor | np.ndarray],
  scales_batch: dict[str, np.ndarray],
) -> None:
  robot = env.scene["robot"]
  dof_ids = base["dof_ids"]
  num_envs = int(env.num_envs)
  nj = int(dof_ids.shape[0])

  stiffness = base["stiffness"] * torch.as_tensor(
    scales_batch["stiffness_scale"], device=env.device, dtype=torch.float32
  ).reshape(num_envs, nj)
  damping = base["damping"] * torch.as_tensor(
    scales_batch["damping_scale"], device=env.device, dtype=torch.float32
  ).reshape(num_envs, nj)
  effort_limit = base["effort_limit"] * torch.as_tensor(
    scales_batch["effort_scale"], device=env.device, dtype=torch.float32
  ).reshape(num_envs, nj)
  armature = base["armature"] * torch.as_tensor(
    scales_batch["armature_scale"], device=env.device, dtype=torch.float32
  ).reshape(num_envs, nj)
  frictionloss = base["frictionloss"] * torch.as_tensor(
    scales_batch["friction_scale"], device=env.device, dtype=torch.float32
  ).reshape(num_envs, nj)

  env.sim.model.dof_armature[:, dof_ids] = armature
  env.sim.model.dof_frictionloss[:, dof_ids] = frictionloss

  for actuator in robot.actuators:
    act = _unwrap_actuator(actuator)
    target_ids = act.target_ids.to(device=env.device, dtype=torch.long)
    if hasattr(act, "set_gains"):
      act.set_gains(
        env_ids=slice(None),
        kp=stiffness[:, target_ids],
        kd=damping[:, target_ids],
      )
    if hasattr(act, "set_effort_limit"):
      act.set_effort_limit(
        env_ids=slice(None),
        effort_limit=effort_limit[:, target_ids],
      )


def _rollout_and_score_batch(
  env: ManagerBasedRlEnv,
  cmd: np.ndarray,
  real: np.ndarray,
  init_state: np.ndarray,
  delay_steps: np.ndarray,
  eval_joint_ids: np.ndarray,
  max_shift_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Run batched rollouts and compute best-shift RMSE per env without storing trajectories."""
  robot = env.scene["robot"]
  num_envs = int(env.num_envs)
  q_ids = robot.indexing.joint_q_adr.to(device=env.device, dtype=torch.long)
  v_ids = robot.indexing.joint_v_adr.to(device=env.device, dtype=torch.long)
  n = int(cmd.shape[0])
  nj = int(cmd.shape[1])
  eval_joint_ids_t = torch.as_tensor(
    eval_joint_ids, device=env.device, dtype=torch.long
  )
  n_eval = int(eval_joint_ids_t.shape[0])

  q0 = torch.as_tensor(init_state, device=env.device, dtype=torch.float32).reshape(1, nj)
  q0 = q0.repeat(num_envs, 1)
  v0 = torch.zeros((num_envs, nj), device=env.device, dtype=torch.float32)
  env.sim.data.qpos[:, q_ids] = q0
  env.sim.data.qvel[:, v_ids] = v0
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.scene.update(dt=0.0)

  delay_steps_t = torch.as_tensor(
    delay_steps, device=env.device, dtype=torch.long
  ).reshape(num_envs)
  if torch.any(delay_steps_t < 0):
    raise ValueError("delay_steps must be non-negative.")
  cmd_t = torch.as_tensor(cmd, device=env.device, dtype=torch.float32)
  real_t = torch.as_tensor(real, device=env.device, dtype=torch.float32)

  shifts_np = np.arange(-int(max_shift_steps), int(max_shift_steps) + 1, dtype=np.int64)
  shifts_t = torch.as_tensor(shifts_np, device=env.device, dtype=torch.long)
  n_shifts = int(shifts_np.shape[0])
  counts = torch.zeros((n_shifts,), device=env.device, dtype=torch.float32)
  sumsq_eval = torch.zeros((num_envs, n_shifts, n_eval), device=env.device, dtype=torch.float32)
  sumsq_full = torch.zeros((num_envs, n_shifts, nj), device=env.device, dtype=torch.float32)

  for i in range(n):
    cmd_idx = torch.clamp(i - delay_steps_t, min=0, max=n - 1)
    target = cmd_t[cmd_idx]
    encoder_bias = robot.data.encoder_bias[:, :]
    robot.set_joint_position_target(target - encoder_bias, joint_ids=slice(None))

    for _ in range(env.cfg.decimation):
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)

    env.sim.forward()
    env.scene.update(dt=0.0)
    sim_now = robot.data.joint_pos

    for shift_idx, shift in enumerate(shifts_np.tolist()):
      real_idx = i - int(shift)
      if real_idx < 0 or real_idx >= n:
        continue
      diff = sim_now - real_t[real_idx].reshape(1, nj)
      sq = diff * diff
      sumsq_full[:, shift_idx, :] += sq
      sumsq_eval[:, shift_idx, :] += sq[:, eval_joint_ids_t]
      counts[shift_idx] += 1.0

  counts = torch.clamp(counts, min=1.0)
  rmse_eval_by_shift = torch.sqrt(
    sumsq_eval.sum(dim=-1) / (counts.reshape(1, n_shifts) * float(max(1, n_eval)))
  )
  best_shift_idx = torch.argmin(rmse_eval_by_shift, dim=1)
  best_rmse = rmse_eval_by_shift.gather(1, best_shift_idx.reshape(num_envs, 1)).squeeze(1)
  best_shift = shifts_t[best_shift_idx]

  mse_eval_by_joint = sumsq_eval / counts.reshape(1, n_shifts, 1)
  mse_full_by_joint = sumsq_full / counts.reshape(1, n_shifts, 1)

  gather_eval_idx = best_shift_idx.reshape(num_envs, 1, 1).expand(num_envs, 1, n_eval)
  gather_full_idx = best_shift_idx.reshape(num_envs, 1, 1).expand(num_envs, 1, nj)
  best_eval_joint_rmse = torch.sqrt(
    torch.gather(mse_eval_by_joint, 1, gather_eval_idx).squeeze(1)
  )
  best_full_joint_rmse = torch.sqrt(
    torch.gather(mse_full_by_joint, 1, gather_full_idx).squeeze(1)
  )

  return (
    best_rmse.detach().cpu().numpy().astype(np.float64),
    best_shift.detach().cpu().numpy().astype(np.int64),
    best_eval_joint_rmse.detach().cpu().numpy().astype(np.float64),
    best_full_joint_rmse.detach().cpu().numpy().astype(np.float64),
  )


def _simulate_open_loop_single(
  env: ManagerBasedRlEnv,
  cmd: np.ndarray,
  init_state: np.ndarray,
  delay_steps: int,
) -> np.ndarray:
  if int(env.num_envs) != 1:
    raise ValueError("Single rollout requires env.num_envs == 1.")

  robot = env.scene["robot"]
  q_ids = robot.indexing.joint_q_adr.to(device=env.device, dtype=torch.long)
  v_ids = robot.indexing.joint_v_adr.to(device=env.device, dtype=torch.long)
  n = int(cmd.shape[0])
  nj = int(cmd.shape[1])

  q0 = torch.as_tensor(init_state, device=env.device, dtype=torch.float32).reshape(1, nj)
  v0 = torch.zeros((1, nj), device=env.device, dtype=torch.float32)
  env.sim.data.qpos[:, q_ids] = q0
  env.sim.data.qvel[:, v_ids] = v0
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.scene.update(dt=0.0)

  delay_steps = max(0, int(delay_steps))
  cmd_t = torch.as_tensor(cmd, device=env.device, dtype=torch.float32)
  sim = np.zeros((n, nj), dtype=np.float64)

  for i in range(n):
    cmd_idx = max(0, min(n - 1, i - delay_steps))
    target = cmd_t[cmd_idx].reshape(1, nj)
    encoder_bias = robot.data.encoder_bias[:, :]
    robot.set_joint_position_target(target - encoder_bias, joint_ids=slice(None))

    for _ in range(env.cfg.decimation):
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)

    env.sim.forward()
    env.scene.update(dt=0.0)
    sim[i] = robot.data.joint_pos[0].detach().cpu().numpy()

  return sim


def _evaluate_candidates(
  env: ManagerBasedRlEnv,
  *,
  base: dict[str, torch.Tensor | np.ndarray],
  cmd: np.ndarray,
  real: np.ndarray,
  init_state: np.ndarray,
  eval_joint_ids: np.ndarray,
  max_shift_steps: int,
  candidates: list[dict[str, object]],
  trial_offset: int = 0,
  log_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
  n_candidates = len(candidates)
  batch_envs = int(env.num_envs)
  nj = int(cmd.shape[1])

  rmses = np.full((n_candidates,), np.inf, dtype=np.float64)
  shifts = np.zeros((n_candidates,), dtype=np.int64)

  local_best: dict[str, object] = {
    "rmse": float("inf"),
    "shift": 0,
    "delay_steps": 0,
    "scales": _identity_scales(nj),
    "per_joint_eval": np.full((len(eval_joint_ids),), np.inf, dtype=np.float64),
    "per_joint_full": np.full((nj,), np.inf, dtype=np.float64),
    "candidate_id": -1,
  }

  for start in range(0, n_candidates, batch_envs):
    scales_batch, delay_batch, active = _build_scales_batch(
      candidates=candidates,
      start=start,
      batch_envs=batch_envs,
      nj=nj,
    )
    _apply_joint_params(env, base=base, scales_batch=scales_batch)
    rmse_batch, shift_batch, per_eval_batch, per_full_batch = _rollout_and_score_batch(
      env=env,
      cmd=cmd,
      real=real,
      init_state=init_state,
      delay_steps=delay_batch,
      eval_joint_ids=eval_joint_ids,
      max_shift_steps=max_shift_steps,
    )

    for env_id in range(active):
      candidate_id = start + env_id
      candidate = candidates[candidate_id]
      scales = candidate["scales"]
      delay_steps = int(candidate["delay_steps"])
      rmse = float(rmse_batch[env_id])
      shift = int(shift_batch[env_id])
      per_eval = per_eval_batch[env_id]
      per_full = per_full_batch[env_id]

      rmses[candidate_id] = rmse
      shifts[candidate_id] = shift
      if rmse < float(local_best["rmse"]):
        local_best = {
          "rmse": rmse,
          "shift": shift,
          "delay_steps": delay_steps,
          "scales": {k: v.copy() for k, v in scales.items()},
          "per_joint_eval": per_eval.copy(),
          "per_joint_full": per_full.copy(),
          "candidate_id": candidate_id,
        }

      global_trial_idx = trial_offset + candidate_id
      if global_trial_idx == 0 or ((global_trial_idx + 1) % log_every == 0):
        print(
          f"[INFO] trial={global_trial_idx + 1} rmse={rmse:.6f} "
          f"delay={delay_steps} shift={shift} {_candidate_summary(scales, eval_joint_ids)}"
        )

  return rmses, shifts, local_best


def _default_output_path(log_file: str) -> Path:
  src = Path(log_file).expanduser().resolve()
  out_dir = src.parent
  return out_dir / f"{src.stem}_sysid_fit.npz"


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    SysIdFitCliConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  del remaining_args

  rng = np.random.default_rng(args.seed)
  t_raw, cmd_raw, state_raw = _load_npz_log(args.log_file)
  if args.apply_joint_swap_remap or args.apply_thumb_cmc_negate:
    cmd_raw = _apply_leap_left_remap(
      cmd_raw,
      apply_swap=args.apply_joint_swap_remap,
      negate_thumb_cmc=args.apply_thumb_cmc_negate,
    )
    state_raw = _apply_leap_left_remap(
      state_raw,
      apply_swap=args.apply_joint_swap_remap,
      negate_thumb_cmc=args.apply_thumb_cmc_negate,
    )
    print(
      "[INFO] Applied LEAP-left joint remap to log data: "
      f"swap={args.apply_joint_swap_remap}, negate_thumb_cmc={args.apply_thumb_cmc_negate}."
    )

  t_cmd, cmd, real = _resample_to_hz(t_raw, cmd_raw, state_raw, args.command_hz)

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script, but no GPU is available.")
  device = args.device or "cuda:0"
  if not str(device).startswith("cuda"):
    raise ValueError(f"This script requires a CUDA device. Got device='{device}'.")

  if args.parallel_envs <= 0:
    raise ValueError(f"parallel_envs must be > 0, got {args.parallel_envs}.")

  # Build temporary single-env instance to resolve joint count and candidates.
  env_probe = _build_env(chosen_task, play=args.play, device=device, num_envs=1)
  try:
    base_probe = _extract_joint_baselines(env_probe)
    nj = int(base_probe["dof_ids"].shape[0])
  finally:
    env_probe.close()

  if cmd.shape[1] != nj:
    raise ValueError(
      f"Joint dimension mismatch between log ({cmd.shape[1]}) and sim ({nj})."
    )

  train_joint_ids = _finger_joint_ids(args.finger)
  if np.any(train_joint_ids < 0) or np.any(train_joint_ids >= nj):
    raise ValueError(
      f"Finger '{args.finger}' joint ids out of range for nj={nj}: {train_joint_ids.tolist()}"
    )

  if args.optimizer == "cem":
    if args.cem_generations <= 0:
      raise ValueError(f"cem_generations must be > 0, got {args.cem_generations}.")
    if not (0.0 < args.cem_elite_frac <= 1.0):
      raise ValueError(f"cem_elite_frac must be in (0,1], got {args.cem_elite_frac}.")
    if not (0.0 < args.cem_alpha <= 1.0):
      raise ValueError(f"cem_alpha must be in (0,1], got {args.cem_alpha}.")
    if not (0.0 <= args.cem_min_std_frac <= args.cem_init_std_frac):
      raise ValueError(
        "cem_min_std_frac must satisfy 0 <= cem_min_std_frac <= cem_init_std_frac."
      )

  if args.optimizer == "cem":
    pop_size = int(args.cem_population) if args.cem_population is not None else int(args.parallel_envs)
    if pop_size <= 0:
      raise ValueError(f"cem_population must be > 0, got {pop_size}.")
    max_candidates_per_round = pop_size
  else:
    max_candidates_per_round = 1 + max(0, int(args.random_trials))

  batch_envs = min(int(args.parallel_envs), int(max_candidates_per_round))
  print(
    f"[INFO] Finger calibration mode: finger='{args.finger}', "
    f"train_joint_ids={train_joint_ids.tolist()} (params={5 * len(train_joint_ids)}), "
    f"delay_range=[{args.delay_min_steps}, {args.delay_max_steps}], optimize_delay={args.optimize_delay}"
  )
  print(
    f"[INFO] Running batched sysid on GPU: device={device}, "
    f"optimizer={args.optimizer}, parallel_envs={batch_envs}"
  )

  env = _build_env(chosen_task, play=args.play, device=device, num_envs=batch_envs)
  try:
    step_dt = float(env.step_dt)
    expected_hz = 1.0 / step_dt
    if abs(expected_hz - args.command_hz) > 1e-5:
      print(
        "[WARN] Command Hz and env step Hz mismatch: "
        f"command_hz={args.command_hz:.4f}, env_hz={expected_hz:.4f}"
      )

    base = _extract_joint_baselines(env)

    best_rmse = float("inf")
    best_shift = 0
    best_delay_steps = 0
    best_scales = _identity_scales(nj)
    best_per_joint_eval = np.full((len(train_joint_ids),), np.inf, dtype=np.float64)
    best_per_joint_full = np.full((cmd.shape[1],), np.inf, dtype=np.float64)
    total_trials = 0

    if args.optimizer == "random":
      candidates: list[dict[str, object]] = [
        {
          "scales": _identity_scales(nj),
          "delay_steps": 0,
        }
      ]
      for _ in range(max(0, int(args.random_trials))):
        candidates.append(
          {
            "scales": _sample_scales(rng, args, nj, train_joint_ids=train_joint_ids),
            "delay_steps": _sample_delay(
              rng,
              optimize_delay=args.optimize_delay,
              min_steps=int(args.delay_min_steps),
              max_steps=int(args.delay_max_steps),
            ),
          }
        )
      rmses, _, local_best = _evaluate_candidates(
        env,
        base=base,
        cmd=cmd,
        real=real,
        init_state=real[0],
        eval_joint_ids=train_joint_ids,
        max_shift_steps=int(args.max_shift_steps),
        candidates=candidates,
        trial_offset=0,
      )
      total_trials = len(candidates)
      if float(local_best["rmse"]) < best_rmse:
        best_rmse = float(local_best["rmse"])
        best_shift = int(local_best["shift"])
        best_delay_steps = int(local_best["delay_steps"])
        best_scales = {k: v.copy() for k, v in local_best["scales"].items()}
        best_per_joint_eval = local_best["per_joint_eval"].copy()
        best_per_joint_full = local_best["per_joint_full"].copy()
      del rmses
    else:
      pop_size = int(args.cem_population) if args.cem_population is not None else int(args.parallel_envs)
      low, high = _continuous_bounds(args, train_joint_ids=train_joint_ids)
      mean = np.ones_like(low, dtype=np.float64)
      std = np.maximum(
        (high - low) * float(args.cem_init_std_frac),
        (high - low) * float(args.cem_min_std_frac),
      )
      std_floor = np.maximum((high - low) * float(args.cem_min_std_frac), 1e-6)
      delay_values, delay_probs = _delay_values_and_probs(args)
      elite_count = max(1, int(round(float(args.cem_elite_frac) * float(pop_size))))

      for gen in range(int(args.cem_generations)):
        vecs = _sample_cem_population(
          rng,
          mean=mean,
          std=std,
          low=low,
          high=high,
          population=pop_size,
        )
        delays = _sample_delay_batch(
          rng, delay_values=delay_values, delay_probs=delay_probs, population=pop_size
        )
        # Keep baseline in population for numerical robustness.
        vecs[0, :] = 1.0
        delays[0] = 0

        candidates = [
          {
            "scales": _vector_to_scales(v, nj=nj, train_joint_ids=train_joint_ids),
            "delay_steps": int(d),
          }
          for v, d in zip(vecs, delays, strict=True)
        ]

        rmses, _, local_best = _evaluate_candidates(
          env,
          base=base,
          cmd=cmd,
          real=real,
          init_state=real[0],
          eval_joint_ids=train_joint_ids,
          max_shift_steps=int(args.max_shift_steps),
          candidates=candidates,
          trial_offset=total_trials,
        )
        total_trials += len(candidates)

        if float(local_best["rmse"]) < best_rmse:
          best_rmse = float(local_best["rmse"])
          best_shift = int(local_best["shift"])
          best_delay_steps = int(local_best["delay_steps"])
          best_scales = {k: v.copy() for k, v in local_best["scales"].items()}
          best_per_joint_eval = local_best["per_joint_eval"].copy()
          best_per_joint_full = local_best["per_joint_full"].copy()

        elite_idx = np.argpartition(rmses, elite_count - 1)[:elite_count]
        elite_vecs = vecs[elite_idx]
        elite_delays = delays[elite_idx]
        elite_mean = np.mean(elite_vecs, axis=0)
        elite_std = np.std(elite_vecs, axis=0)
        mean = (1.0 - float(args.cem_alpha)) * mean + float(args.cem_alpha) * elite_mean
        std = (1.0 - float(args.cem_alpha)) * std + float(args.cem_alpha) * elite_std
        mean = np.clip(mean, low, high)
        std = np.maximum(std, std_floor)
        delay_probs = _update_delay_probs(
          delay_probs,
          delay_values=delay_values,
          elite_delays=elite_delays,
          alpha=float(args.cem_alpha),
        )

        print(
          f"[INFO] CEM gen={gen + 1}/{args.cem_generations} "
          f"best_gen_rmse={float(np.min(rmses)):.6f} global_best_rmse={best_rmse:.6f} "
          f"delay_probs={delay_probs.tolist()}"
        )

    # Re-run best candidate in a single env to record the final simulated trajectory.
    env_final = _build_env(chosen_task, play=args.play, device=device, num_envs=1)
    try:
      base_final = _extract_joint_baselines(env_final)
      single_scales_batch = {k: v.reshape(1, -1) for k, v in best_scales.items()}
      _apply_joint_params(env_final, base=base_final, scales_batch=single_scales_batch)
      best_sim = _simulate_open_loop_single(
        env_final,
        cmd=cmd,
        init_state=real[0],
        delay_steps=best_delay_steps,
      )
    finally:
      env_final.close()

    print("[INFO] Best sysid result:")
    print(f"  rmse={best_rmse:.6f}, best_shift={best_shift}, best_delay_steps={best_delay_steps}")
    print(f"  per_joint_rmse({args.finger})={best_per_joint_eval.tolist()}")

    out_path = (
      Path(args.output_file).expanduser().resolve()
      if args.output_file is not None
      else _default_output_path(args.log_file)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
      str(out_path),
      time=t_cmd.astype(np.float64),
      hand_cmd=cmd.astype(np.float32),
      hand_state_real=real.astype(np.float32),
      hand_state_sim=best_sim.astype(np.float32),
      best_shift=np.asarray([best_shift], dtype=np.int32),
      best_delay_steps=np.asarray([best_delay_steps], dtype=np.int32),
      best_rmse=np.asarray([best_rmse], dtype=np.float64),
      eval_finger=np.asarray(args.finger),
      eval_joint_ids=train_joint_ids.astype(np.int32),
      per_joint_rmse_eval=best_per_joint_eval.astype(np.float64),
      per_joint_rmse_full=best_per_joint_full.astype(np.float64),
      joint_names=base["joint_names"],
      best_stiffness_scale=best_scales["stiffness_scale"].astype(np.float64),
      best_damping_scale=best_scales["damping_scale"].astype(np.float64),
      best_effort_scale=best_scales["effort_scale"].astype(np.float64),
      best_armature_scale=best_scales["armature_scale"].astype(np.float64),
      best_friction_scale=best_scales["friction_scale"].astype(np.float64),
      baseline_stiffness=base["stiffness"][0].detach().cpu().numpy().astype(np.float64),
      baseline_damping=base["damping"][0].detach().cpu().numpy().astype(np.float64),
      baseline_effort_limit=base["effort_limit"][0].detach().cpu().numpy().astype(np.float64),
      baseline_armature=base["armature"][0].detach().cpu().numpy().astype(np.float64),
      baseline_frictionloss=base["frictionloss"][0].detach().cpu().numpy().astype(np.float64),
      fitted_stiffness=(
        base["stiffness"][0].detach().cpu().numpy() * best_scales["stiffness_scale"]
      ).astype(np.float64),
      fitted_damping=(
        base["damping"][0].detach().cpu().numpy() * best_scales["damping_scale"]
      ).astype(np.float64),
      fitted_effort_limit=(
        base["effort_limit"][0].detach().cpu().numpy() * best_scales["effort_scale"]
      ).astype(np.float64),
      fitted_armature=(
        base["armature"][0].detach().cpu().numpy() * best_scales["armature_scale"]
      ).astype(np.float64),
      fitted_frictionloss=(
        base["frictionloss"][0].detach().cpu().numpy() * best_scales["friction_scale"]
      ).astype(np.float64),
      trials=np.asarray([total_trials], dtype=np.int32),
      optimizer=np.asarray(args.optimizer),
      task_id=np.asarray(chosen_task),
      command_hz=np.asarray([args.command_hz], dtype=np.float64),
      env_step_dt=np.asarray([step_dt], dtype=np.float64),
      remap_apply_swap=np.asarray([args.apply_joint_swap_remap], dtype=np.bool_),
      remap_negate_thumb_cmc=np.asarray([args.apply_thumb_cmc_negate], dtype=np.bool_),
    )
    print(f"[INFO] Saved sysid fit artifact: {out_path}")

  finally:
    env.close()


if __name__ == "__main__":
  main()
