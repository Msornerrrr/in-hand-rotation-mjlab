"""Goal-conditioned SO(3) cube reorientation (RL), with the hardware-safe LEAP settings: the env cfg builder.

Same hand/cube/DR as the yaw task, plus what a sampling-MPC study established as
sim2real-relevant:
  * sysid-corrected servo gains (Kp 1.5 / Kd 0.05, 0.75x on rot joints),
  * a |command - measured| clamp (``max_command_lead``; the real stack's is 0.15 rad),
  * a 0.2 rad/step target slew cap,
  * contact-force penalties (hand-cube above 10 N, finger-finger above 2 N).
Goals are uniformly random orientations; a goal counts as reached within 0.4 rad
and is immediately replaced. Actor observations equal the BC student's features
(``sampling/bc.py``), so demonstration-trained students can initialize or be compared.
Training starts from the sampling-MPC expert's states (``reset_reverse_curriculum``).

The MDP terms (goal command, observations, rewards, events) are in ``hand_cube/mdp``; the
registered variants in ``config/leap_left_custom_reorient``.
"""

from __future__ import annotations

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import LEAP_SERVO_KP15_PERJOINT_2026_07
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import Sim2RealCfg, apply_sim2real
from in_hand_rotation_mjlab.tasks.hand_cube.mdp.commands import GOAL


# ---------------------------------------------------------------------------
# Nominal resting cube position (env frame) per variant: grasp-cache mean xy, median z.
CUBE_HOME = {"5": (-0.085, 0.055, 0.230), "6": (-0.086, 0.056, 0.236), "7": (-0.085, 0.058, 0.241)}

# Cube variants: side length (cm) -> (half size, grasp cache relative to the in_hand_rotation_mjlab package).
CUBE_VARIANTS = {
  "5": (0.025, "tasks/hand_cube/cache/leap_left_custom_grasp_cache_cube5cm.npz"),
  "6": (0.030, "tasks/hand_cube/cache/leap_left_custom_grasp_cache_cube6cm.npz"),
  "7": (0.035, "tasks/hand_cube/cache/leap_left_custom_grasp_cache_cube7cm.npz"),
}


def make_reorient_env_cfg(play: bool = False, *, cube: str, max_lead: float = 0.25, rc_glob: str | None = None,
                          rc_prob: float = 0.75, home_anchor: bool = False, cube_obs_delay: bool = False,
                          peak_force: tuple[float, float, float] | None = None,
                          home_wall: tuple[float, float] | None = None, s2r: Sim2RealCfg | None = None):
  """``cube``: a ``CUBE_VARIANTS`` key (side length in cm). ``max_lead``: the |command -
  measured| clamp (rad).

  ``rc_glob``: expert demo shards (``;``-separated globs) for the reverse curriculum: a
  fraction ``rc_prob`` of episodes starts from an expert state whose goal the expert reached
  within ``rc_steps`` steps (growing with the success rate); the rest start from the grasp
  cache with a uniform SO(3) goal. Follow-up goals use the goal-angle curriculum.
  ``home_anchor``: cube-position penalty from the palm home instead of the episode start;
  ``home_wall`` (weight, margin m) adds a steep penalty beyond the margin.
  ``cube_obs_delay``: 0-100 ms randomized cube-pose latency (a vision tracker's).
  ``peak_force`` (weight, free cube N, free self N): per-substep peak contact-force penalty.
  ``s2r``: the sim-to-real recipe (``Sim2RealCfg``); its ``peak_force`` replaces the argument.
  """
  if s2r is not None and s2r.peak_force is not None:
    peak_force = s2r.peak_force
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_left_custom.env_cfgs import leap_left_custom_hand_cube_rotate_env_cfg
  from in_hand_rotation_mjlab.tasks.hand_cube.safety import add_safety_sensors

  half, cache = CUBE_VARIANTS[cube]
  servo = s2r.servo if s2r is not None and s2r.servo is not None else LEAP_SERVO_KP15_PERJOINT_2026_07
  cfg = leap_left_custom_hand_cube_rotate_env_cfg(play=play, cube_half_size=half, grasp_cache_file=cache, servo=servo)
  add_safety_sensors(cfg)
  act = cfg.actions["joint_pos"]
  act.delta_min, act.delta_max = -0.2, 0.2
  act.max_command_lead = max_lead
  cfg.curriculum = {}
  cfg.metrics = {}
  rc = rc_glob is not None and not play
  cfg.commands = {GOAL: hand_cube_mdp.CubeGoalCommandCfg(resampling_time_range=(10.0, 10.0), debug_vis=False,
                                           init_max_angle=torch.pi if play else 0.8,
                                           reverse_curriculum=rc)}

  robot = SceneEntityCfg("robot", joint_names=(".*",))
  actor = {
    "joint_pos": ObservationTermCfg(func=hand_cube_mdp.joint_pos_abs, params={"asset_cfg": robot}, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "cmd": ObservationTermCfg(func=hand_cube_mdp.joint_pos_commanded, params={"action_name": "joint_pos", "asset_cfg": robot}),
    "cube_pos": ObservationTermCfg(func=hand_cube_mdp.cube_pos_rel, noise=Unoise(n_min=-0.002, n_max=0.002)),
    "cube_rot": ObservationTermCfg(func=hand_cube_mdp.cube_rot6d, noise=Unoise(n_min=-0.02, n_max=0.02)),
    "goal_rot": ObservationTermCfg(func=hand_cube_mdp.goal_rot6d),
    "rel_rot": ObservationTermCfg(func=hand_cube_mdp.goal_rel_rot6d, noise=Unoise(n_min=-0.02, n_max=0.02)),
  }
  critic = {k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()}
  critic |= {
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params={"asset_cfg": robot}),
    "cube_lin_vel": ObservationTermCfg(func=hand_cube_mdp.cube_lin_vel),
    "cube_ang_vel": ObservationTermCfg(func=hand_cube_mdp.cube_ang_vel),
    "cube_mass": ObservationTermCfg(func=hand_cube_mdp.cube_mass, params={"object_name": "cube", "body_name": "cube"}),
    "cube_friction": ObservationTermCfg(func=hand_cube_mdp.cube_friction_coeff,
                                        params={"object_name": "cube", "geom_name": "cube_geom", "axis": 0}),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=not play, history_length=3, flatten_history_dim=True),
    "critic": ObservationGroupCfg(critic, enable_corruption=False, history_length=1, flatten_history_dim=True),
  }
  cfg.scale_rewards_by_dt = False
  cfg.rewards = {
    # progress clipped at 0.05 rad/step: a flung, tumbling cube must not farm rotation reward
    "rot_progress": RewardTermCfg(func=hand_cube_mdp.goal_rot_progress, weight=5.0, params={"clip": 0.05}),
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.02),
    "cube_linvel": RewardTermCfg(func=hand_cube_mdp.object_linvel_l1, weight=-0.5, params={"object_name": "cube"}),
    "success": RewardTermCfg(func=hand_cube_mdp.goal_success, weight=5.0),
    "rot_dist": RewardTermCfg(func=hand_cube_mdp.goal_rot_dist, weight=-0.02),
    "cube_pos": RewardTermCfg(func=hand_cube_mdp.object_position_l2_from_reset, weight=-20.0,
                              params={"object_name": "cube"}),
    "fallen": RewardTermCfg(func=envs_mdp.is_terminated, weight=-30.0),  # any non-timeout termination
    "cube_force": RewardTermCfg(func=hand_cube_mdp.contact_force_excess_sq, weight=-2e-3,
                                params={"sensor_name": "safety_hand_cube", "free": 10.0}),
    "self_force": RewardTermCfg(func=hand_cube_mdp.contact_force_excess_sq, weight=-2e-3,
                                params={"sensor_name": "safety_hand_self", "free": 2.0}),
    "torque": RewardTermCfg(func=hand_cube_mdp.joint_torque_l2, weight=-0.02, params={"asset_cfg": robot}),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.002),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(func=envs_mdp.root_height_below_minimum,
                                    params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube")}),
    "cube_too_fast": TerminationTermCfg(func=hand_cube_mdp.object_linear_speed_above,
                                        params={"max_linear_speed": 1.0, "asset_cfg": SceneEntityCfg("cube")}),
    "cube_far": TerminationTermCfg(func=hand_cube_mdp.object_pose_rp_position_deviation_from_reset,
                                   params={"max_position_error": 0.08, "max_tilt_error": 1e9,
                                           "asset_cfg": SceneEntityCfg("cube")}),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }
  cfg.episode_length_s = 1e9 if play else 30.0
  if rc_glob is not None:
    cfg.rewards["cube_pos"] = RewardTermCfg(func=hand_cube_mdp.cube_pos_from_start, weight=-20.0)
  if cube_obs_delay and not play:  # 0-2 step (0-100 ms) latency on the cube pose, as from a vision tracker
    for name in ("cube_pos", "cube_rot", "rel_rot"):
      t = cfg.observations["actor"].terms[name]
      t.delay_min_lag, t.delay_max_lag, t.delay_hold_prob, t.delay_update_period = 0, 2, 0.9, 10
  if peak_force is not None and not play:
    # (weight, free cube N, free self N): penalize the peak contact force over all physics
    # substeps of the control step (impacts last a few ms); gentle, aimed at rare spikes
    from mjlab.managers.metrics_manager import MetricsTermCfg

    w, free_cube, free_self = peak_force
    cfg.metrics["peak_cube_force"] = MetricsTermCfg(func=hand_cube_mdp.track_peak_forces, per_substep=True, reduce="max")
    cfg.rewards["cube_force"] = RewardTermCfg(func=hand_cube_mdp.peak_force_excess_sq, weight=w,
                                              params={"key": "_peak_cube_force", "free": free_cube})
    cfg.rewards["self_force"] = RewardTermCfg(func=hand_cube_mdp.peak_force_excess_sq, weight=w,
                                              params={"key": "_peak_self_force", "free": free_self})
  if home_anchor:  # keep the cube near the palm centre across many goals (anti-drift)
    cfg.rewards["cube_pos"] = RewardTermCfg(func=hand_cube_mdp.cube_pos_from_home, weight=-20.0,
                                            params={"home": CUBE_HOME[cube]})
    if home_wall is not None:  # (weight, margin m): steep penalty for excursions beyond the margin
      cfg.rewards["cube_wall"] = RewardTermCfg(func=hand_cube_mdp.cube_pos_from_home, weight=home_wall[0],
                                               params={"home": CUBE_HOME[cube], "margin": home_wall[1]})
  if s2r is not None:
    apply_sim2real(cfg, s2r, play)
  if rc:
    from mjlab.managers.event_manager import EventTermCfg

    # applied after the grasp-cache reset (events run in insertion order)
    cfg.events["reset_reverse_curriculum"] = EventTermCfg(
      func=hand_cube_mdp.reset_reverse_curriculum, mode="reset",
      params={"shard_glob": rc_glob, "prob": rc_prob,
              "cube_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",))})
  return cfg
