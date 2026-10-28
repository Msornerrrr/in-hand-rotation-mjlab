"""RL task: fingertip-only yaw rotation of a sphere (4.5-6 cm) on the LEAP left custom hand, palm up
or palm down, on the measured plant (gear servo + its randomization, the 0.15 rad driver lead clamp).

Design:
- resets: full holds from the planner pools (``resets_rl/<sphere>_<palm>.npz``: settled 4-pad grasps
  plus sampling-MPC trajectory states over 0-120 deg of yaw), each with the object radius it was
  recorded with (set per env at reset);
- action: joint-target deltas of 1/24 rad per 50 ms step (0.83 rad/s) through the real driver's
  lead clamp, one setpoint per control step;
- reward: yaw rate about the palm normal clipped at a slow 0.3 rad/s, gated by drift; stability
  (object speed, drift, off-axis rotation, spin above 0.5 rad/s); keep all four PaXini pads in
  contact and nothing else; no crushing (> 8 N per pad); torque / work / pose / action-rate costs;
- actor observations (deployable): measured joints, the commanded targets, and the four pads'
  tactile force (normal + shear in the pad frame, sensor gain / offset randomization, dead band)
  plus contact flags, 8-step history; the critic sees the object state and plant parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hc

from . import rl_mdp as M
from .env_cfg import fingertip_rotate_env_cfg
from .objects import OBJECTS

_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_POOL = str(_DATA / "resets_rl/sphere*mm_*.npz")  # single-sphere pools (not shipped; the mixed tasks do not use them)
MIXED_POOL = str(_DATA / "palmdown_canon4.npz")  # four-pad template starts of the 22 mixed objects
PAD_FRICTION_2026_10 = ((0.25, 1.2), (0.2, 0.8))  # sliding friction: pads, other contacts
ROBOT = SceneEntityCfg("robot", joint_names=(".*",))
CUBE_GEOM = SceneEntityCfg("cube", geom_names=("cube_geom",))
_DELAY = dict(delay_min_lag=0, delay_max_lag=1, delay_hold_prob=0.9, delay_update_period=10)


def fingertip_rotate_rl_env_cfg(palm: str = "up", *, pool_glob: str | None = None, delta: float = 1.0 / 24.0,
                                clip_rate: float = 0.3, clip_min: float | None = None, pads_bonus: float = 0.0,
                                privileged_actor: bool = False, gravity_curriculum: tuple[float, int] | None = None,
                                few_pads_penalty: float = 0.0, recipe: str = "base", pose_weight: float = 0.6,
                                push: tuple[float, float] = (2.0, 0.25), drop_penalty: float = 50.0,
                                max_drift: float = 0.04, rotate_reward: str = "rate", rotate_window: float = 5.0, episode_s: float = 20.0,
                                mix_ramp: int = 2000 * 32, critic_deficit: bool = False,
                                actor_obs: str = "tactile", actor_history: int = 8, support_weight: float = 0.0,
                                nonpad_weight: float = 2.0, palm_weight: float = 0.0, mix_delay: int = 0,
                                pad_friction: tuple[float, float] | None = None,
                                mixed_pool: str | None = None, stage2_iter: int = 2000, pad_model: bool = False,
                                contact_friction: tuple | None = None, idle_weight: float = 0.0, idle_s: float = 1.0,
                                link_weight: float = 0.0, pad_friction_obs: bool = False,
                                play: bool = False):
  """``pads_bonus`` > 0 adds ``pads_beyond_two`` (a third / fourth pad on the object) with that weight.
  ``privileged_actor``: a teacher, whose actor also sees the object's position (vs its reset) and velocity in
  the hand frame and the sphere radius (to be distilled into the proprioceptive + tactile student).
  ``clip_min``: the most negative yaw rate counted (default -clip_rate; a smaller magnitude forgives the slight
  back-rotation of a regrasp, a large one makes a fast snap-back cost its full angle). ``gravity_curriculum`` (g0, ramp control steps): object weight ramping g0 -> 1.
  ``recipe="hora"``: HORA's reward (github.com/haozhiqi/hora, AllegroHandHora.yaml) on top of the pad-only
  constraints: clipped rotation, object linear velocity (L1), joint pose drift from the reset grasp (L2, no dead
  zone), torque and work, plus random object pushes (``push`` = (scale x mass, probability per step)). The
  penalties keep HORA's ratios to the maximum rotation reward (``pose_weight`` = 0.6 per rad^2 is HORA's): a
  one-way stroke that parks the fingers at their limits then costs more than it earns, so the policy has to keep
  cycling the fingers back to the grasp. A privileged actor also sees HORA's mass and friction.
  ``drop_penalty`` (one-off, x dt) and ``max_drift`` (drop termination, m): with the HORA penalties the policy
  learned to spin the object fast and let it go (a fresh grasp after each reset is a free stroke); a penalty
  larger than a spin-and-drop earns (~7.5) and a tighter drift limit close that (HORA itself ends the episode
  once the object sinks 5 mm). ``rotate_reward``: "rate" (clipped instantaneous yaw rate), "ratchet" (new yaw
  beyond the episode's high-water mark, ``rot_progress_ratchet``) or "window" (net yaw over ``rotate_window`` s,
  ``rot_progress_window``); the last two pay only net rotation, not a wind-forward / snap-back cycle. "mix"
  ramps from "rate" to "ratchet" over ``mix_ramp`` control steps (``rot_progress_mix``), for warm starts.
  ``critic_deficit``: the critic also sees the gap below the high-water mark (``yaw_deficit``, appended last), which
  the ratchet's reward depends on.
  ``episode_s``: a longer episode makes one stroke from the fresh reset grasp worth less next to a gait.
  ``actor_obs="proprio"``: HORA's asymmetric actor-critic split - the actor sees only what the real LEAP driver
  reports (measured joints, the commanded targets, and their difference, the tracking error that carries the
  contact loads), ``actor_history`` steps of it; no tactile, no object state. (``privileged_actor`` adds the
  object state on top for a teacher.) The critic keeps the privileged state.
  ``support_weight``: reward for the share of the object's weight carried by contact normal forces
  (``normal_support``, ~0 for an equator pinch, ~1 for a cradle from below; also shown to the critic).
  ``nonpad_weight``: penalty while links / palm touch the object (0 allows link contact); ``palm_weight``: penalty
  while the palm touches it. ``mix_delay``: control steps of the clipped rate alone before the "mix" ramp starts.
  ``pad_friction``: override of the pad sliding-friction randomization range.
  ``mixed_pool``: train over several primitives at once (``objects.MixedObject``: one body with a sphere, a box and a
  cylinder geom, the episode's one active) from a mixed-shape pool (``export_mixed_resets.py``); spheres and cylinders
  until ``stage2_iter`` iterations (32 control steps each; env ``FTROT_STAGE2_ITER`` overrides), then all four shapes. The object's shape and size replace
  the radius in the privileged inputs, and ``support_weight`` pays ``pads_low`` (pads pressing the object's lower
  part, shape-agnostic) instead of the sphere-only ``normal_support``. ``pad_model``: the calibrated PaXini pad contact
  model (``LEAP_PAD_TWIN_2026_10``); ``contact_friction``: (pad, other) sliding-friction DR ranges.
  ``idle_weight``: penalty per finger whose pad has not pressed the object for over ``idle_s`` (``finger_idle``):
  every finger must take part in the gait (fingers may lift to step, none may sit out). ``link_weight``: penalty
  while a finger link other than the fingertip presses the object (> 0.2 N; ``link_contact``). ``pad_friction_obs``:
  the privileged actor and the critic also see the pads' sliding friction (the one that sets the pad-object contact)."""
  pool_glob = pool_glob or os.environ.get("FTROT_RESETS", DEFAULT_POOL)
  from in_hand_rotation_mjlab.robots.leap_hand.pad_model import LEAP_PAD_TWIN_2026_10

  from .objects import MIXED_SET_2026_10, MixedObject

  mixed = mixed_pool is not None
  # the largest sphere: the model's collision bounds cover every radius the resets shrink it to (the multi-shape
  # object: each geom compiled at its largest size in the set)
  obj = MixedObject(tuple(OBJECTS[n] for n in MIXED_SET_2026_10)) if mixed else OBJECTS["sphere60mm"]
  cfg = fingertip_rotate_env_cfg(obj, palm, delta=delta, play=play, pad_model=LEAP_PAD_TWIN_2026_10 if pad_model else None,
                                 contact_friction=contact_friction)
  p = dict(palm=palm)

  # --- events: the plant / contact randomization of the base task, then the pool reset --------
  for k in ("reset_robot_joints", "reset_cube_pose"):
    cfg.events.pop(k, None)
  cfg.events["dr_encoder_bias"] = EventTermCfg(func=envs_mdp.dr.encoder_bias, mode="reset",
                                               params={"bias_range": (-0.015, 0.015), "asset_cfg": ROBOT})
  tac = {"tactile_gain": (1.0, 1.0) if play else (0.7, 1.3), "tactile_bias": 0.0 if play else 0.05}
  if mixed:
    cfg.events["reset_from_pool"] = EventTermCfg(
      func=M.reset_from_mixed_pool, mode="reset",
      params=p | tac | {"pool_path": os.environ.get("FTROT_MIXED_RESETS", mixed_pool),
                        "stage2_step": int(os.environ.get("FTROT_STAGE2_ITER", stage2_iter)) * 32})
  else:
    cfg.events["reset_from_pool"] = EventTermCfg(
      func=M.reset_from_pool, mode="reset", params=p | tac | {"pool_glob": pool_glob, "cube_cfg": CUBE_GEOM})
  if recipe == "hora" and not play and push[0] > 0:
    cfg.events["random_push"] = EventTermCfg(func=M.random_object_push, mode="step",
                                             params=p | {"scale": push[0], "prob": push[1]})
  if gravity_curriculum is not None and not play:
    cfg.events["gravity_curriculum"] = EventTermCfg(
      func=M.reset_gravity_curriculum, mode="reset",
      params=p | {"g0": gravity_curriculum[0], "ramp_steps": gravity_curriculum[1]})

  if pad_friction is not None and "dr_split_contact_friction" in cfg.events:
    cfg.events["dr_split_contact_friction"].params["pad_range"] = tuple(pad_friction)

  # --- observations ---------------------------------------------------------------------------
  actor = {
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                                    params={"asset_cfg": ROBOT, "biased": True}, **_DELAY),
    "prev_commanded_joint_pos": ObservationTermCfg(func=hc.joint_pos_commanded,
                                                   params={"action_name": "joint_pos", "asset_cfg": ROBOT}),
  }
  if actor_obs == "tactile":
    actor |= {
      "pad_tactile": ObservationTermCfg(func=M.pad_tactile, noise=Unoise(n_min=-0.03, n_max=0.03),
                                        params=p | {"randomize": not play}, **_DELAY),
      "pad_contact": ObservationTermCfg(func=M.pad_contact_flags, params={"f_min": 0.1}, **_DELAY),
    }
  elif actor_obs == "proprio":
    # commanded - measured (encoder: offset, noise, delay): the servos' tracking error, i.e. the load on each joint
    actor["joint_pos_error"] = ObservationTermCfg(func=hc.joint_pos_command_error, noise=Unoise(n_min=-0.01, n_max=0.01),
                                                  params={"action_name": "joint_pos", "biased": True, "asset_cfg": ROBOT},
                                                  **_DELAY)
  else:
    raise ValueError(actor_obs)
  if privileged_actor:
    actor |= {
      "obj_pos": ObservationTermCfg(func=M.obj_pos_rel_hand, params=p),
      "obj_vel": ObservationTermCfg(func=M.obj_vel_hand, params=p),
      "obj_radius": ObservationTermCfg(func=M.obj_shape_dims if mixed else M.obj_radius, params=p),
    }
    if recipe == "hora":
      actor |= {
        "obj_mass": ObservationTermCfg(func=hc.cube_mass, params={"object_name": "cube", "body_name": "cube"}),
        "obj_friction": ObservationTermCfg(func=hc.cube_friction_coeff,
                                           params={"object_name": "cube", "geom_name": "cube_geom", "axis": 0}),
      }
  if privileged_actor and pad_friction_obs:
    actor["pad_friction"] = ObservationTermCfg(func=M.pad_friction)
  critic = {
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params={"asset_cfg": ROBOT}),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params={"asset_cfg": ROBOT}),
    "joint_pos_error": ObservationTermCfg(func=hc.joint_pos_command_error,
                                          params={"action_name": "joint_pos", "biased": True, "asset_cfg": ROBOT}),
    "prev_commanded_joint_pos": ObservationTermCfg(func=hc.joint_pos_commanded,
                                                   params={"action_name": "joint_pos", "asset_cfg": ROBOT}),
    "obj_pos": ObservationTermCfg(func=M.obj_pos_rel_hand, params=p),
    "obj_vel": ObservationTermCfg(func=M.obj_vel_hand, params=p),
    "obj_radius": ObservationTermCfg(func=M.obj_shape_dims if mixed else M.obj_radius, params=p),
    "obj_mass": ObservationTermCfg(func=hc.cube_mass, params={"object_name": "cube", "body_name": "cube"}),
    "obj_friction": ObservationTermCfg(func=hc.cube_friction_coeff,
                                       params={"object_name": "cube", "geom_name": "cube_geom", "axis": 0}),
    "pad_forces": ObservationTermCfg(func=M.pad_forces_true, params=p),
    "pad_tactile": ObservationTermCfg(func=M.pad_tactile, params=p | {"randomize": not play}),
    "nonpad_force": ObservationTermCfg(func=M.nonpad_force_obs),
    "episode_yaw": ObservationTermCfg(func=M.episode_yaw, params=p),
  }
  if pad_friction_obs:
    critic["pad_friction"] = ObservationTermCfg(func=M.pad_friction)
  if critic_deficit:
    critic["yaw_deficit"] = ObservationTermCfg(func=M.yaw_deficit, params=p)
  if support_weight > 0:
    critic["support"] = ObservationTermCfg(func=M.pads_low_obs if mixed else M.support_obs, params=p)
  cfg.observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=not play, history_length=actor_history, flatten_history_dim=True),
    "critic": ObservationGroupCfg(critic, enable_corruption=False, history_length=1, flatten_history_dim=True),
  }

  # --- rewards (x dt: per-second rates) -------------------------------------------------------
  cfg.rewards = {
    # weight 10 at a 0.3 rad/s cap: rotating must pay more than holding still with every pad (the palm-up
    # pilot settled into a 4-pad hold at weight 5 with a 0.5 pad bonus)
    "rotate": RewardTermCfg(func={"rate": M.rot_vel_clipped, "ratchet": M.rot_progress_ratchet,
                                  "window": M.rot_progress_window, "mix": M.rot_progress_mix}[rotate_reward], weight=10.0,
                            params=p | {"clip_min": -clip_rate if clip_min is None else clip_min, "clip_max": clip_rate,
                                        "drift_threshold": 0.02,
                                        "outside_factor": 0.1} | ({"window_s": rotate_window} if rotate_reward == "window" else {})
                                        | ({"ramp_steps": mix_ramp, "ramp_delay": mix_delay} if rotate_reward == "mix" else {})),
    "pad_contact": RewardTermCfg(func=M.pad_contact_frac, weight=0.1, params={"f_min": 0.1}),
    "nonpad_contact": RewardTermCfg(func=M.nonpad_contact, weight=-nonpad_weight, params={"f_min": 0.05}),
    "obj_linvel": RewardTermCfg(func=M.obj_linvel_l2, weight=-20.0),
    "obj_drift": RewardTermCfg(func=M.obj_drift_l2, weight=-500.0, params=p),
    "angvel_perp": RewardTermCfg(func=M.angvel_perp_l2, weight=-1.0, params=p),
    "spin_excess": RewardTermCfg(func=M.yaw_rate_excess_sq, weight=-2.0, params=p | {"free": 0.5}),
    # the settled holds press ~5-6 N per pad: only guard against crushing (sensor-safe), do not fight the squeeze
    "pad_force_excess": RewardTermCfg(func=M.pad_force_excess, weight=-0.1, params={"free": 8.0, "cap": 5.0}),
    "pose_diff": RewardTermCfg(func=hc.pose_diff_l2_from_reset, weight=-0.1,
                               params={"asset_cfg": ROBOT, "average_per_joint": False, "joint_tolerance": 0.4}),
    "torque": RewardTermCfg(func=hc.joint_torque_l2, weight=-0.1, params={"asset_cfg": ROBOT}),
    "work": RewardTermCfg(func=hc.actuator_work_l2_penalty, weight=-0.05, params={"asset_cfg": ROBOT}),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    # iteration-800 stalls: the fingers' abduction joints pinned at their limits after one stroke
    "abduction_limit": RewardTermCfg(func=M.joint_limit_margin_sq, weight=-0.5,
                                     params={"asset_cfg": SceneEntityCfg("robot", joint_names=("if_rot", "mf_rot", "rf_rot")),
                                             "margin": 0.1}),
    "dropped": RewardTermCfg(func=M.obj_dropped_penalty, weight=-drop_penalty, params=p | {"max_drift": max_drift}),
  }
  if recipe == "hora":
    rmax = 10.0 * clip_rate  # the most the rotation term pays per second
    cfg.rewards = {
      "rotate": cfg.rewards["rotate"],
      "obj_linvel": RewardTermCfg(func=M.obj_linvel_l1, weight=-0.6 * rmax),
      "pose_diff": RewardTermCfg(func=hc.pose_diff_l2_from_reset, weight=-pose_weight * rmax,
                                 params={"asset_cfg": ROBOT, "average_per_joint": False, "joint_tolerance": 0.0}),
      "torque": RewardTermCfg(func=hc.joint_torque_l2, weight=-0.2 * rmax, params={"asset_cfg": ROBOT}),
      "work": RewardTermCfg(func=hc.actuator_work_l2_penalty, weight=-1.0 * rmax, params={"asset_cfg": ROBOT}),
      # the task's own constraints: pads only (PaXini), no crushing, keep the object
      "nonpad_contact": cfg.rewards["nonpad_contact"],
      "pad_force_excess": cfg.rewards["pad_force_excess"],
      "dropped": cfg.rewards["dropped"],
    }
  if nonpad_weight <= 0:
    cfg.rewards.pop("nonpad_contact", None)
  if support_weight > 0:
    cfg.rewards["support"] = RewardTermCfg(func=M.pads_low if mixed else M.normal_support_capped, weight=support_weight,
                                           params=p)
  if link_weight > 0:
    cfg.rewards["link_contact"] = RewardTermCfg(func=M.link_contact, weight=-link_weight, params={"f_min": 0.2})
  if idle_weight > 0:
    cfg.rewards["finger_idle"] = RewardTermCfg(func=M.finger_idle, weight=-idle_weight, params=p | {"idle_s": idle_s})
  if palm_weight > 0:
    cfg.rewards["palm_contact"] = RewardTermCfg(func=M.palm_contact, weight=-palm_weight, params={"f_min": 0.05})
  if few_pads_penalty > 0:
    cfg.rewards["few_pads"] = RewardTermCfg(func=M.few_pads, weight=-few_pads_penalty, params={"f_min": 0.1, "n_min": 3})
  if pads_bonus > 0:
    cfg.rewards["pads_beyond_two"] = RewardTermCfg(func=M.pads_beyond_two, weight=pads_bonus, params={"f_min": 0.1})
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "dropped": TerminationTermCfg(func=M.object_dropped, params=p | {"max_drift": max_drift}),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }
  cfg.metrics = {
    "yaw_rate": MetricsTermCfg(func=M.m_yaw_rate, params=p),
    "episode_yaw": MetricsTermCfg(func=M.m_episode_yaw, params=p),
    "drift": MetricsTermCfg(func=M.m_drift, params=p),
    "pads_pressing": MetricsTermCfg(func=M.m_pads, params={"f_min": 0.1}),
    "nonpad_touching": MetricsTermCfg(func=M.m_nonpad),
    "normal_support": MetricsTermCfg(func=M.m_support, params=p),
    "palm_touching": MetricsTermCfg(func=M.m_palm),
    "gravity_scale": MetricsTermCfg(func=M.m_gravity_scale, params=p),
    "push_force": MetricsTermCfg(func=M.m_push, params=p),
  }
  if link_weight > 0:
    cfg.metrics["link_touching"] = MetricsTermCfg(func=M.m_link_touching, params={"f_min": 0.2})
  if idle_weight > 0:
    cfg.metrics["fingers_active"] = MetricsTermCfg(func=M.m_fingers_active, params=p | {"idle_s": idle_s})
    cfg.metrics["index_active"] = MetricsTermCfg(func=M.m_index_active, params=p | {"idle_s": idle_s})
  if mixed:
    cfg.metrics["pads_low"] = MetricsTermCfg(func=M.m_pads_low, params=p)
    cfg.metrics["stage"] = MetricsTermCfg(func=M.m_stage, params=p)
  cfg.curriculum = {}
  cfg.commands = {}
  cfg.episode_length_s = 1e9 if play else episode_s
  return cfg


def fingertip_rotate_ppo_cfg(palm: str, suffix: str = "", init_std: float = 0.5):
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_left_custom.rl_cfg import leap_left_custom_hand_cube_rotate_gear_ppo_cfg

  cfg = leap_left_custom_hand_cube_rotate_gear_ppo_cfg()
  cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "init_std": init_std}
  cfg.experiment_name = f"leap_left_custom_fingertip_rotate_palm{palm}{suffix}"
  cfg.max_iterations = 5000
  cfg.save_interval = 200
  return cfg
