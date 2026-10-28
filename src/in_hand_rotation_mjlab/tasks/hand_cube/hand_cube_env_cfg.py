import os
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import (
  LEAP_REFLECTED_ARMATURE,
  LEAP_S2R_2026_10_CONTACT_DR,
  LEAP_S2R_2026_10_DR,
  LEAP_SERVO_S2R_2026_10,
  LeapServo,
)
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.mdp.actions import JointPositionDeltaActionCfg


# Actor observation delay randomization (in env/control steps).
# Light randomization between 0 and 1 step.
_ACTOR_OBS_DELAY_MIN_LAG = 0
_ACTOR_OBS_DELAY_MAX_LAG = 1
_ACTOR_OBS_DELAY_HOLD_PROB = 0.9
_ACTOR_OBS_DELAY_UPDATE_PERIOD = 10


def _cube_half_size_from_env(default: float) -> float:
  """Read an optional play/train cube-size override from the environment."""
  side_length = os.environ.get("HAND_CUBE_SIDE_LENGTH")
  half_size = os.environ.get("HAND_CUBE_HALF_SIZE")

  if side_length is not None:
    default = 0.5 * float(side_length)
  if half_size is not None:
    default = float(half_size)
  if default <= 0.0:
    raise ValueError("Cube size must be positive.")
  return default


def make_hand_cube_inhand_rotate_env_cfg() -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_min_lag=_ACTOR_OBS_DELAY_MIN_LAG,
      delay_max_lag=_ACTOR_OBS_DELAY_MAX_LAG,
      delay_hold_prob=_ACTOR_OBS_DELAY_HOLD_PROB,
      delay_update_period=_ACTOR_OBS_DELAY_UPDATE_PERIOD,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        # Include encoder offset effects in the measured proprioception stream.
        "biased": True,
      },
    ),
    "prev_commanded_joint_pos": ObservationTermCfg(
      func=hand_cube_mdp.joint_pos_commanded,
      params={
        "action_name": "joint_pos",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  critic_terms = {
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_pos_error": ObservationTermCfg(
      func=hand_cube_mdp.joint_pos_command_error,
      params={
        "action_name": "joint_pos",
        "biased": True,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "prev_commanded_joint_pos": ObservationTermCfg(
      func=hand_cube_mdp.joint_pos_commanded,
      params={
        "action_name": "joint_pos",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "cube_pose_palm": ObservationTermCfg(
      func=hand_cube_mdp.cube_pose_in_palm_frame,
      params={
        "object_name": "cube",
        "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
        "palm_center_geom_expr": "palm_collision_.*",
      },
    ),
    "cube_lin_vel_palm": ObservationTermCfg(
      func=hand_cube_mdp.cube_lin_vel_in_palm_frame,
      params={
        "object_name": "cube",
        "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
        "palm_center_geom_expr": "palm_collision_.*",
      },
    ),
    "cube_ang_vel_palm": ObservationTermCfg(
      func=hand_cube_mdp.cube_ang_vel_in_palm_frame,
      params={
        "object_name": "cube",
        "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
        "palm_center_geom_expr": "palm_collision_.*",
      },
    ),
    "cube_size": ObservationTermCfg(
      func=hand_cube_mdp.cube_size,
      params={"object_name": "cube", "geom_name": "cube_geom"},
    ),
    "cube_mass": ObservationTermCfg(
      func=hand_cube_mdp.cube_mass,
      params={"object_name": "cube", "body_name": "cube"},
    ),
    "cube_com_offset_b": ObservationTermCfg(
      func=hand_cube_mdp.cube_com_offset_b,
      params={"object_name": "cube", "body_name": "cube"},
    ),
    "cube_friction": ObservationTermCfg(
      func=hand_cube_mdp.cube_friction_coeff,
      params={"object_name": "cube", "geom_name": "cube_geom", "axis": 0},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      actor_terms,
      enable_corruption=True,
      history_length=10,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      critic_terms,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionDeltaActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,
      offset=0.0,
      use_default_offset=False,
      clip_to_joint_limits=True,
      use_soft_joint_pos_limits=True,
      delta_min=-(1/24),
      delta_max=(1/24),
      interpolate_decimation=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "frame_viz": hand_cube_mdp.HandCubeFrameVizCommandCfg(
      hand_name="robot",
      object_name="cube",
      palm_body_name="palm",
      palm_center_geom_expr="palm_collision_.*",
      resampling_time_range=(1e9, 1e9),
      debug_vis=True,
    ),
  }

  events = {
    # Required for fixed-base mocap robots in vectorized envs:
    # place each hand at its own env origin on every reset.
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_cube_pose": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube"),
        "pose_range": {
          "x": (-0.015, 0.015),
          "y": (-0.015, 0.015),
          "z": (-0.01, 0.01),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
  }

  rewards = {
    # LEAPSim-inspired reward with an additional drift-gated rotation term:
    # R_rotate = clip(ω_fd_z) * drift_factor(xyz, roll/pitch error from reset)
    "rotate_finite_diff": RewardTermCfg(
      func=hand_cube_mdp.object_yaw_finite_diff_clipped,
      weight=1.25,
      params={
        "object_name": "cube",
        "clip_min": -0.25,
        "clip_max": 0.25,
        # Smooth finite-difference yaw over recent steps for robustness.
        "history_steps": 4,
        # Multiplicative drift gate on non-yaw motion from reset pose.
        # If position drift > 2 cm or roll/pitch tilt drift > threshold,
        # reduce this reward to 10% (step mode).
        "drift_position_threshold": 0.02,
        "drift_tilt_threshold": 0.35,
        "drift_mode": "step",
        "drift_inside_factor": 1.0,
        "drift_outside_factor": 0.1,
      },
    ),
    # "fingertip_contact": RewardTermCfg(
    #   func=hand_cube_mdp.fingertip_cube_contact_fraction,
    #   weight=0.1,
    #   params={"sensor_name": "fingertip_cube_contact"},
    # ),
    "object_linvel_penalty": RewardTermCfg(
      func=hand_cube_mdp.object_linvel_l1,
      weight=-0.3,
      params={"object_name": "cube"},
    ),
    "pose_diff_penalty": RewardTermCfg(
      func=hand_cube_mdp.pose_diff_l2_from_reset,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "average_per_joint": False,
        "joint_tolerance": 0.4,
      },
    ),
    "torque_penalty": RewardTermCfg(
      func=hand_cube_mdp.joint_torque_l2,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "work_penalty": RewardTermCfg(
      func=hand_cube_mdp.actuator_work_l2_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "object_fallen": RewardTermCfg(
      func=hand_cube_mdp.object_fallen,
      weight=-10.0,
      params={"object_name": "cube", "minimum_height": 0.2},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube")},
    ),
    "cube_too_fast": TerminationTermCfg(
      func=hand_cube_mdp.object_linear_speed_above,
      params={"max_linear_speed": 1.0, "asset_cfg": SceneEntityCfg("cube")},
    ),
    "cube_pose_deviation": TerminationTermCfg(
      func=hand_cube_mdp.object_pose_rp_position_deviation_from_reset,
      params={
        "max_position_error": 0.08,
        "max_tilt_error": 0.8,
        "asset_cfg": SceneEntityCfg("cube"),
      },
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  metrics = {
    "rotation_progress": MetricsTermCfg(
      func=hand_cube_mdp.object_rotation_progress,
      params={
        "asset_cfg": SceneEntityCfg("cube"),
        "target_yaw_rate": 0.20,
        "position_threshold": 0.02,
        "tilt_threshold": 0.35,
      },
    ),
    "linear_speed": MetricsTermCfg(
      func=hand_cube_mdp.object_linear_speed,
      params={"asset_cfg": SceneEntityCfg("cube")},
    ),
    "position_error": MetricsTermCfg(
      func=hand_cube_mdp.object_pose_rp_error_from_reset,
      params={
        "component": "position",
        "asset_cfg": SceneEntityCfg("cube"),
      },
    ),
    "tilt_error": MetricsTermCfg(
      func=hand_cube_mdp.object_pose_rp_error_from_reset,
      params={
        "component": "tilt",
        "asset_cfg": SceneEntityCfg("cube"),
      },
    ),
  }

  # Smooth progress-based curriculum from episode metric `rotation_progress`.
  # As progress improves, penalties are increased toward their final strengths.
  curriculum = {
    "object_linvel_penalty_weight": CurriculumTermCfg(
      func=hand_cube_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "object_linvel_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.03,
        "weight_max": -0.3,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
    "pose_diff_penalty_weight": CurriculumTermCfg(
      func=hand_cube_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "pose_diff_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.01,
        "weight_max": -0.1,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
    "torque_penalty_weight": CurriculumTermCfg(
      func=hand_cube_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "torque_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.1,
        "weight_max": -1.0,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
    "work_penalty_weight": CurriculumTermCfg(
      func=hand_cube_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "work_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.01,
        "weight_max": -0.1,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(),
      entities={},
      num_envs=1,
      env_spacing=0.6,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="palm",
      distance=0.45,
      elevation=-25,
      azimuth=110,
    ),
    sim=SimulationCfg(
      nconmax=55,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=10,
    # Match LEAP: 400 control steps at 20 Hz.
    episode_length_s=20.0,
    scale_rewards_by_dt=True,
  )

# Helper function to create embodiment-specific configs
# ---------------------------------------------------------------------------
# Sim-to-real recipe for the hand_cube tasks, from the first hardware runs of the SO(3)
# reorientation policy (2026-10-05). On the real LEAP
# the Lead20 policy dropped the cube every ~10 s: the real servos are ~2x faster/stiffer than
# the training plant (replay fit Kp 4-8 N m/rad, 0-10 ms delay vs Kp 1.5 + 25-75 ms), the
# printed links and cube are slippery, the vision pose freezes on ~40% of steps, and the
# policy flicks the cube (p99 5 rad/s) with ~2.5x the yaw policy's contact force, running
# motors at their limit (one shut down on overload). Every field is a switch, for ablations.
@dataclass(frozen=True)
class Sim2RealCfg:
  delta: float | None = 1.0 / 12.0  # rad/step target slew (1.67 rad/s; Lead20: 0.2 = 4 rad/s)
  # the servo model (robots/leap_hand/leap_plant.py): Kp 4 N m/rad (Lead20: KP15_PERJOINT, Kp 1.5)
  servo: LeapServo | None = LEAP_SERVO_S2R_2026_10
  kp_dr: tuple[float, float] | None = LEAP_S2R_2026_10_DR.kp_scale  # startup scale -> 3..6 N m/rad
  kd_dr: tuple[float, float] | None = LEAP_S2R_2026_10_DR.kd_scale
  effort: tuple[float, float] | None = LEAP_S2R_2026_10_DR.effort  # N m, absolute per joint per reset
  armature_dr: tuple[float, float] | None = LEAP_S2R_2026_10_DR.armature_scale  # x the reflected rotor inertia
  lag: tuple[int, int] | None = LEAP_S2R_2026_10_DR.lag  # command delay, physics substeps (0-25 ms; Lead20 5-15)
  pad_friction: tuple[float, float] | None = LEAP_S2R_2026_10_CONTACT_DR.pad_friction  # pads (they win their contacts)
  other_friction: tuple[float, float] | None = LEAP_S2R_2026_10_CONTACT_DR.other_friction  # links, palm, cube
  peak_force: tuple[float, float, float] | None = (-3e-3, 6.0, 2.0)  # (w, free cube N, free self N)
  torque_sat: tuple[float, float] | None = (-1.0, 0.8)  # (w, fraction of the effort limit)
  joint_limit: float | None = -2.0  # rad beyond the soft limits (do not stall a joint on its stop)
  cube_spin: tuple[float, float] | None = (-0.05, 2.0)  # (w, free rad/s)
  # vision-pose freezes of the actor's cube_pos / cube_rot / rel_rot terms: per-episode Markov
  # chain, P(fresh->held), P(held->held) ranges
  freeze: tuple[tuple[float, float], tuple[float, float]] | None = ((0.0, 0.2), (0.5, 0.9))


def apply_sim2real(cfg, s: Sim2RealCfg, play: bool) -> None:
  """Apply ``s`` to a hand_cube env cfg (the servo version itself is chosen with the robot)."""
  from mjlab.envs import mdp as mjlab_mdp
  from mjlab.managers.event_manager import EventTermCfg

  act = cfg.actions["joint_pos"]
  if s.delta is not None:
    act.delta_min, act.delta_max = -s.delta, s.delta
  ev = cfg.events
  if s.kp_dr is not None and "dr_robot_pd_gains" in ev:
    ev["dr_robot_pd_gains"].params.update(kp_range=s.kp_dr, kd_range=s.kd_dr or (0.9, 1.1))
  if s.effort is not None and "dr_effort_limits" in ev:
    ev["dr_effort_limits"].params["effort_limit"] = s.effort
  if s.armature_dr is not None and "dr_reflected_inertia" in ev:
    ev["dr_reflected_inertia"].params["ranges"] = tuple(k * LEAP_REFLECTED_ARMATURE for k in s.armature_dr)
  if s.lag is not None and "dr_action_delay" in ev:
    ev["dr_action_delay"].params["lag_range"] = s.lag
  if s.pad_friction is not None and "dr_shared_contact_friction" in ev:
    ev.pop("dr_shared_contact_friction")
    ev["dr_split_contact_friction"] = EventTermCfg(
      func=hand_cube_mdp.randomize_split_contact_friction, mode="reset",
      params={"pad_range": s.pad_friction, "other_range": s.other_friction})
  if play:
    return
  robot = SceneEntityCfg("robot", joint_names=(".*",))
  if s.torque_sat is not None:
    cfg.rewards["torque_sat"] = RewardTermCfg(func=hand_cube_mdp.torque_near_limit, weight=s.torque_sat[0],
                                              params={"frac": s.torque_sat[1]})
  if s.joint_limit is not None:
    cfg.rewards["joint_limit"] = RewardTermCfg(func=mjlab_mdp.joint_pos_limits, weight=s.joint_limit,
                                               params={"asset_cfg": robot})
  if s.cube_spin is not None:
    cfg.rewards["cube_spin"] = RewardTermCfg(func=hand_cube_mdp.cube_angvel_excess_sq, weight=s.cube_spin[0],
                                             params={"free": s.cube_spin[1]})
  if s.freeze is not None:
    terms = cfg.observations["actor"].terms
    terms["cube_pos"].func = hand_cube_mdp.cube_pos_rel_frozen
    terms["cube_rot"].func = hand_cube_mdp.cube_rot6d_frozen
    terms["rel_rot"].func = hand_cube_mdp.goal_rel_rot6d_frozen
    ev["reset_cube_freeze"] = EventTermCfg(func=hand_cube_mdp.reset_cube_freeze, mode="reset",
                                           params={"p_fh": s.freeze[0], "p_hh": s.freeze[1]})


# ---------------------------------------------------------------------------


# The measured hand only (servo randomization, command delay, contact friction), without the
# reorientation policy's action and reward shaping: for hand_cube tasks moved to the 2026-10
# plant that keep their own action scale and rewards (the yaw rotation tasks).
MEASURED_PLANT_2026_10 = Sim2RealCfg(delta=None, peak_force=None, torque_sat=None, joint_limit=None,
                                     cube_spin=None, freeze=None)


def measured_gear_plant(cfg, dr=None):
  """A hand_cube env cfg (yaw or reorientation) on the measured LEAP servo, gear_2026_10.

  ``use_leap_gear_servo`` (robots/leap_hand/leap_plant.py: the gear servo, its randomization,
  the real hand's link masses), applied last, after the task's own sim-to-real recipe. The setpoint
  is not interpolated across the 50 ms control step: the real dispatcher publishes one target per
  step, and the servo samples it every 12.5 ms. ``dr``: the servo randomization (default the
  version's measured uncertainty; ``LEAP_GEAR_SERVO_2026_10_NOMINAL`` for nominal evaluation).
  """
  from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import LEAP_GEAR_SERVO_2026_10_DR, use_leap_gear_servo

  use_leap_gear_servo(cfg, dr=dr or LEAP_GEAR_SERVO_2026_10_DR)
  cfg.actions["joint_pos"].interpolate_decimation = False
  return cfg


def make_hand_cube_embodiment_env_cfg(
    *,
    robot_cfg_fn,
    grasp_init_joint_pos: dict[str, float],
    cube_spawn_pos: tuple[float, float, float],
    grasp_cache_file: str | None,
    negate_yaw_rate: bool = True,
    play: bool = False,
    palm_body_name: str = "palm",
    palm_center_geom_expr: str = "palm_collision_.*",
    fingertip_geom_expr: str = ".*_tip",
    cube_half_size: float | None = None,
    cube_mass: float = 0.1,
    use_grasp_cache: bool = True,
    enable_domain_randomization: bool = True,
    sim2real: Sim2RealCfg | None = None,
    max_command_lead: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create hand_cube env config for a specific LEAP hand embodiment.

    Args:
        robot_cfg_fn: Function that returns robot EntityCfg (e.g., get_leap_left_hand_cfg)
        grasp_init_joint_pos: Initial joint positions for grasping pose
        cube_spawn_pos: (x, y, z) spawn position for cube
        grasp_cache_file: Path to .npz grasp cache file
        negate_yaw_rate: Whether to negate yaw rate for rewards (True for left hand, False for right hand)
        play: Whether this is for deployment (nominal cube size, no observation noise or
            curriculum, infinite episodes; domain randomization stays on,
            ``enable_domain_randomization=False`` removes it)
        sim2real: plant randomization / shaping recipe applied last (``Sim2RealCfg``)
        max_command_lead: the |command - measured| clamp [rad] (the real LEAP driver's)
    """
    import mujoco
    from in_hand_rotation_mjlab import PACKAGE_PATH
    from mjlab.entity import EntityCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import (
        LEAP_ACTUATOR_EFFORT_LIMIT_NM,
        LEAP_REFLECTED_ARMATURE,
    )
    
    if cube_half_size is None:
        cube_half_size = _cube_half_size_from_env(default=0.0375)

    def get_cube_spec(cube_size: float = cube_half_size, mass: float = cube_mass) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="cube")
        body.add_freejoint(name="cube_joint")
        body.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(cube_size,) * 3,
            mass=mass,
            rgba=(0.85, 0.3, 0.15, 1.0),
        )
        return spec
    
    cfg = make_hand_cube_inhand_rotate_env_cfg()

    # Embodiment-specific palm geometry used by observations and frame debug viz.
    for term_name in ("cube_pose_palm", "cube_lin_vel_palm", "cube_ang_vel_palm"):
        cfg.observations["critic"].terms[term_name].params["hand_cfg"] = SceneEntityCfg(
            "robot", body_names=(palm_body_name,)
        )
        cfg.observations["critic"].terms[term_name].params["palm_center_geom_expr"] = palm_center_geom_expr
    cfg.commands["frame_viz"].palm_body_name = palm_body_name
    cfg.commands["frame_viz"].palm_center_geom_expr = palm_center_geom_expr

    # Set embodiment-specific yaw reward sign
    cfg.rewards["rotate_finite_diff"].params["negate_yaw_rate"] = negate_yaw_rate

    robot_cfg = robot_cfg_fn()
    robot_cfg.init_state.joint_pos = grasp_init_joint_pos
    
    cfg.scene.entities = {
        "robot": robot_cfg,
        "cube": EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=cube_spawn_pos,
                rot=(1.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
            spec_fn=get_cube_spec,
        ),
    }
    
    # Add contact sensor
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="fingertip_cube_contact",
            primary=ContactMatch(mode="geom", pattern=fingertip_geom_expr, entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    cfg.sim.contact_sensor_maxmatch = 256
    
    # Contact metrics
    cfg.metrics["fingertip_contact_count"] = MetricsTermCfg(
        func=hand_cube_mdp.contact_count,
        params={"sensor_name": "fingertip_cube_contact"},
    )
    cfg.metrics["fingertip_contact_fraction"] = MetricsTermCfg(
        func=hand_cube_mdp.contact_fraction,
        params={"sensor_name": "fingertip_cube_contact"},
    )
    cfg.metrics["fingertip_found_mean"] = MetricsTermCfg(
        func=hand_cube_mdp.contact_found_mean,
        params={"sensor_name": "fingertip_cube_contact"},
    )
    cfg.metrics["fingertip_found_max"] = MetricsTermCfg(
        func=hand_cube_mdp.contact_found_max,
        params={"sensor_name": "fingertip_cube_contact"},
    )
    
    # Keep reset joint init fixed
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    
    # Action config
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionDeltaActionCfg)
    joint_pos_action.scale = 1.0
    joint_pos_action.offset = 0.0
    joint_pos_action.use_default_offset = False
    
    # Cube reset pose range
    cfg.events["reset_cube_pose"].params["pose_range"] = {
        "x": (-0.006, 0.006),
        "y": (-0.006, 0.006),
        "z": (-0.005, 0.005),
        "yaw": (-3.14, 3.14),
    }
    
    # Grasp cache reset
    if use_grasp_cache and grasp_cache_file is not None:
        cfg.events["reset_from_grasp_cache"] = EventTermCfg(
            func=hand_cube_mdp.reset_from_grasp_cache,
            mode="reset",
            params={
                "cache_file": str(PACKAGE_PATH / grasp_cache_file),
                "scale_list": (1.0,) if play else (0.95, 0.9, 1.0, 1.05, 1.1),
                "scale_jitter": 0.0 if play else 0.025,
                "cube_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
                "pose_range": {
                    "x": (-0.003, 0.003),
                    "y": (-0.003, 0.003),
                    "z": (-0.002, 0.002),
                    "yaw": (-3.14, 3.14),
                },
            },
        )
    
    # Domain randomization events (shared across all embodiments)
    cfg.events["dr_cube_com"] = EventTermCfg(
        func=hand_cube_mdp.randomize_body_ipos,
        mode="reset",
        params={
            "ranges": {0: (-0.002, 0.002), 1: (-0.002, 0.002), 2: (-0.002, 0.002)},
            "distribution": "uniform",
            "operation": "add",
            "asset_cfg": SceneEntityCfg("cube", body_names=("cube",)),
        },
    )
    cfg.events["dr_shared_contact_friction"] = EventTermCfg(
        func=hand_cube_mdp.randomize_shared_contact_friction,
        mode="reset",
        params={
            "friction_range": (0.6, 1.4),
            "hand_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
            "cube_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
            "axes": (0,),
        },
    )
    cfg.events["dr_cube_mass"] = EventTermCfg(
        func=hand_cube_mdp.randomize_body_mass,
        mode="reset",
        params={
            "mass_range": (0.7, 1.4),
            "distribution": "uniform",
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("cube", body_names=("cube",)),
        },
    )
    cfg.events["dr_robot_link_masses"] = EventTermCfg(
        func=hand_cube_mdp.randomize_body_mass,
        mode="reset",
        params={
            "mass_range": (0.8, 1.2),
            "distribution": "uniform",
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        },
    )
    cfg.events["dr_motor_friction"] = EventTermCfg(
        func=hand_cube_mdp.randomize_dof_frictionloss,
        mode="reset",
        params={
            "ranges": (0.5, 1.8),
            "distribution": "uniform",
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
    cfg.events["dr_motor_damping"] = EventTermCfg(
        func=hand_cube_mdp.randomize_dof_damping,
        mode="reset",
        params={
            "ranges": (0.6, 1.6),
            "distribution": "uniform",
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
    cfg.events["dr_reflected_inertia"] = EventTermCfg(
        func=hand_cube_mdp.randomize_dof_armature,
        mode="startup",
        params={
            "ranges": (0.7 * LEAP_REFLECTED_ARMATURE, 1.3 * LEAP_REFLECTED_ARMATURE),
            "distribution": "uniform",
            "operation": "abs",
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
    cfg.events["dr_robot_pd_gains"] = EventTermCfg(
        func=hand_cube_mdp.randomize_pd_gains,
        mode="startup",
        params={
            "kp_range": (0.9, 1.1),
            "kd_range": (0.9, 1.1),
            "distribution": "uniform",
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events["dr_action_delay"] = EventTermCfg(
        func=hand_cube_mdp.sync_actuator_delays,
        mode="reset",
        params={
            "lag_range": (
                int(round(0.5 * cfg.decimation)),
                int(round(1.5 * cfg.decimation)),
            ),
            "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        },
    )
    cfg.events["dr_effort_limits"] = EventTermCfg(
        func=hand_cube_mdp.set_actuator_effort_limits,
        mode="reset",
        params={
            "effort_limit": (
                0.9 * LEAP_ACTUATOR_EFFORT_LIMIT_NM,
                1.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM,
            ),
            "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        },
    )

    if not enable_domain_randomization:
        for event_name in (
            "dr_cube_com",
            "dr_shared_contact_friction",
            "dr_cube_mass",
            "dr_robot_link_masses",
            "dr_motor_friction",
            "dr_motor_damping",
            "dr_reflected_inertia",
            "dr_robot_pd_gains",
            "dr_action_delay",
            "dr_effort_limits",
        ):
            cfg.events.pop(event_name, None)
    
    cfg.viewer.body_name = palm_body_name
    
    if play:
        cfg.episode_length_s = 1e9
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    if max_command_lead is not None:
        cfg.actions["joint_pos"].max_command_lead = max_command_lead
    if sim2real is not None:
        apply_sim2real(cfg, sim2real, play)
    
    return cfg
