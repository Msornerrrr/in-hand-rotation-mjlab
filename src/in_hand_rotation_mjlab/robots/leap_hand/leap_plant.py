"""The LEAP hand's servo model (Dynamixel XC330-M288 under the LEAP driver), versioned.

Every LEAP-hand robot config builds its finger actuators from one ``LeapServo`` version with
``leap_actuators``; the hands (``leap_*_constants.py``) take the version as an argument, and each task config names the
version it was trained on. A new system identification adds a version here and moves
``LEAP_SERVO_LATEST``; a task changes its pin only when it is retrained on the new version,
so no trained policy's plant changes silently. Every training run records its plant (``utils/plant.py``), and loading
a checkpoint into a task on a different plant prints the differences.

Two model families:

- ``LeapServo``: mjlab's IdealPd actuator, tau = clip(kp (q* - q) - kd qdot, +-effort), with a
  reflected rotor inertia (armature), dry joint friction (frictionloss) and a command delay.
  The index/middle/ring abduction ("rot") joints run at ``rot_scale`` x kp and kd, as the
  driver's side gains do. ``per_joint`` names the quantities that also take the per-joint
  CEM sysid scales below. Every version before 2026-10-07.
- ``LeapGearServo``: the measured servo (gear_servo.py): the
  driver's 80 Hz goal clamp, the current-based position law with its current limit, the motor
  constant, and gearbox friction proportional to the load the gear carries, so the hand is
  ~5x harder to push open than it pushes. It also scales the hand's link masses to the real
  hand's 540 g. From gear_2026_10b, optional low-speed stiction on chosen joints.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, replace

import mujoco

from mjlab.actuator import IdealPdActuatorCfg

from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import LeapGearServoActuatorCfg

# -----------------------------------------------------------------------------
# Motor and driver constants (https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/)
# -----------------------------------------------------------------------------
# The driver runs current-based position control, i = Kp_i (q* - q) - Kd_i qdot, with
# Kp_i = 800 mA/rad, Kd_i = 200 mA s/rad and a 550 mA current limit; torque = K_t i.
LEAP_MOTOR_TORQUE_PER_AMP_NM = 0.53
LEAP_MOTOR_CURRENT_LIMIT_MA = 550.0
LEAP_MOTOR_CURRENT_LIMIT_A = LEAP_MOTOR_CURRENT_LIMIT_MA / 1000.0
LEAP_ACTUATOR_EFFORT_LIMIT_NM = LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_MOTOR_CURRENT_LIMIT_A

LEAP_PD_CURRENT_KP_MA_PER_RAD = 800.0
LEAP_PD_CURRENT_KD_MA_PER_RAD_S = 200.0
LEAP_PD_CURRENT_KP_A_PER_RAD = LEAP_PD_CURRENT_KP_MA_PER_RAD / 1000.0
LEAP_PD_CURRENT_KD_A_PER_RAD_S = LEAP_PD_CURRENT_KD_MA_PER_RAD_S / 1000.0
LEAP_PD_STIFFNESS_NM_PER_RAD = LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KP_A_PER_RAD
LEAP_PD_DAMPING_NM_PER_RAD_S = LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KD_A_PER_RAD_S

# The driver's reduced gains on the index/middle/ring abduction ("rot") joints.
LEAP_MCP_SIDE_GAIN_SCALE = 0.75
LEAP_MCP_SIDE_PD_STIFFNESS_NM_PER_RAD = LEAP_PD_STIFFNESS_NM_PER_RAD * LEAP_MCP_SIDE_GAIN_SCALE
LEAP_MCP_SIDE_PD_DAMPING_NM_PER_RAD_S = LEAP_PD_DAMPING_NM_PER_RAD_S * LEAP_MCP_SIDE_GAIN_SCALE

# Reflected rotor inertia: Ia = kg^2 * Ir, with kg = 288.35, Ir = 1.7e-8 kg m^2.
LEAP_GEAR_RATIO = 288.35
LEAP_ROTOR_INERTIA = 1.7e-8
LEAP_REFLECTED_ARMATURE = (LEAP_GEAR_RATIO**2) * LEAP_ROTOR_INERTIA

# Nominal dry friction: 10% of the effort limit (the LEAP paper's estimate).
LEAP_NOMINAL_FRICTIONLOSS_NM = 0.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM

# Command-delay buffer bounds in physics steps; a reset event picks each episode's lag.
LEAP_ACTION_DELAY_MIN_LAG = 0
LEAP_ACTION_DELAY_MAX_LAG = 20
LEAP_ACTION_DELAY_HOLD_PROB = 1.0
LEAP_ACTION_DELAY_UPDATE_PERIOD = 0
LEAP_ACTION_DELAY_PER_ENV_PHASE = True

# -----------------------------------------------------------------------------
# Joint naming
# -----------------------------------------------------------------------------
# Semantic names (the standalone hand XMLs) in training order.
LEAP_JOINT_ORDER = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
LEAP_MCP_SIDE_ROT_JOINTS = ("if_rot", "mf_rot", "rf_rot")
# Hardware names (URDF / Dynamixel ids, the xArm7 composition's grasp XML): joint_k. The
# driver lists each finger rot first; the thumb CMC axis is flipped (q_semantic = -q_hw).
LEAP_HW_JOINT_NAMES = tuple(f"joint_{i}" for i in range(16))
LEAP_HW_NAME = {
    "if_rot": "joint_0", "if_mcp": "joint_1", "if_pip": "joint_2", "if_dip": "joint_3",
    "mf_rot": "joint_4", "mf_mcp": "joint_5", "mf_pip": "joint_6", "mf_dip": "joint_7",
    "rf_rot": "joint_8", "rf_mcp": "joint_9", "rf_pip": "joint_10", "rf_dip": "joint_11",
    "th_cmc": "joint_12", "th_axl": "joint_13", "th_mcp": "joint_14", "th_ipl": "joint_15",
}
LEAP_SEMANTIC_NAME = {hw: sem for sem, hw in LEAP_HW_NAME.items()}
LEAP_HW_SIGN = {name: (-1.0 if name == "th_cmc" else 1.0) for name in LEAP_JOINT_ORDER}

# -----------------------------------------------------------------------------
# Per-joint CEM sysid scales (index_175818 / middle_180246 sysid fits, joints 0..7; ring and
# thumb from the same search). Stiffness sat at the search ceiling (1.4x) and damping at its
# floor (0.7x): the fit was clipped, which is why later versions replace kp and kd outright.
# -----------------------------------------------------------------------------
LEAP_STIFFNESS_SCALE_BY_JOINT = {
    "if_mcp": 1.3821, "if_rot": 1.3903, "if_pip": 1.3601, "if_dip": 1.4000,
    "mf_mcp": 1.3860, "mf_rot": 1.4000, "mf_pip": 1.3462, "mf_dip": 1.4000,
    "rf_mcp": 1.3983, "rf_rot": 1.4000, "rf_pip": 1.4000, "rf_dip": 1.3490,
    "th_cmc": 1.3544, "th_axl": 1.2862, "th_mcp": 1.3870, "th_ipl": 1.3084,
}
LEAP_DAMPING_SCALE_BY_JOINT = {
    "if_mcp": 0.7264, "if_rot": 0.7000, "if_pip": 0.7000, "if_dip": 0.7699,
    "mf_mcp": 0.8130, "mf_rot": 0.7281, "mf_pip": 0.7918, "mf_dip": 0.7000,
    "rf_mcp": 0.7887, "rf_rot": 0.7251, "rf_pip": 0.7000, "rf_dip": 0.8001,
    "th_cmc": 0.7000, "th_axl": 0.7064, "th_mcp": 0.9578, "th_ipl": 0.7178,
}
LEAP_EFFORT_SCALE_BY_JOINT = {
    "if_mcp": 0.8500, "if_rot": 1.0684, "if_pip": 0.8500, "if_dip": 0.8500,
    "mf_mcp": 0.8621, "mf_rot": 1.0606, "mf_pip": 0.8500, "mf_dip": 0.8500,
    "rf_mcp": 0.8892, "rf_rot": 0.8500, "rf_pip": 0.8500, "rf_dip": 1.1062,
    "th_cmc": 0.8500, "th_axl": 0.8633, "th_mcp": 0.8612, "th_ipl": 1.0408,
}
LEAP_ARMATURE_SCALE_BY_JOINT = {
    "if_mcp": 1.2629, "if_rot": 1.4800, "if_pip": 1.1925, "if_dip": 0.6543,
    "mf_mcp": 0.8186, "mf_rot": 1.0668, "mf_pip": 1.3619, "mf_dip": 1.1253,
    "rf_mcp": 0.6000, "rf_rot": 1.2074, "rf_pip": 1.1534, "rf_dip": 1.3067,
    "th_cmc": 0.9726, "th_axl": 0.8835, "th_mcp": 1.0873, "th_ipl": 1.1179,
}
LEAP_FRICTION_SCALE_BY_JOINT = {
    "if_mcp": 1.3455, "if_rot": 0.7058, "if_pip": 1.1626, "if_dip": 0.3881,
    "mf_mcp": 0.9242, "mf_rot": 0.3264, "mf_pip": 0.9817, "mf_dip": 1.5468,
    "rf_mcp": 1.8238, "rf_rot": 1.0831, "rf_pip": 0.9794, "rf_dip": 0.8705,
    "th_cmc": 0.3610, "th_axl": 2.2000, "th_mcp": 0.5970, "th_ipl": 1.8449,
}
_SCALES = {
    "kp": LEAP_STIFFNESS_SCALE_BY_JOINT, "kd": LEAP_DAMPING_SCALE_BY_JOINT, "effort": LEAP_EFFORT_SCALE_BY_JOINT,
    "armature": LEAP_ARMATURE_SCALE_BY_JOINT, "frictionloss": LEAP_FRICTION_SCALE_BY_JOINT,
}


@dataclass(frozen=True)
class LeapServo:
    """One version of the LEAP servo model (see the module docstring)."""

    name: str
    kp: float  # N m/rad, flexion joints
    kd: float  # N m s/rad
    effort: float  # N m, the torque clip
    armature: float  # kg m^2
    frictionloss: float  # N m
    per_joint: frozenset[str] = frozenset()  # of "kp", "kd", "effort", "armature", "frictionloss"
    rot_scale: float = LEAP_MCP_SIDE_GAIN_SCALE
    delay_min_lag: int = LEAP_ACTION_DELAY_MIN_LAG
    delay_max_lag: int = LEAP_ACTION_DELAY_MAX_LAG
    delay_hold_prob: float = LEAP_ACTION_DELAY_HOLD_PROB
    delay_update_period: int = LEAP_ACTION_DELAY_UPDATE_PERIOD
    delay_per_env_phase: bool = LEAP_ACTION_DELAY_PER_ENV_PHASE

    def value(self, quantity: str, joint: str) -> float:
        """``quantity`` ("kp", "kd", ...) for one joint (semantic name)."""
        v = getattr(self, quantity)
        if quantity in ("kp", "kd") and joint in LEAP_MCP_SIDE_ROT_JOINTS:
            v = v * self.rot_scale
        if quantity in self.per_joint:
            v = v * _SCALES[quantity].get(joint, 1.0)
        return v


@dataclass(frozen=True)
class LeapServoDR:
    """Training randomization around a version: its measured uncertainty."""

    kp_scale: tuple[float, float]  # x kp at startup
    kd_scale: tuple[float, float]
    effort: tuple[float, float]  # N m, absolute, per joint per reset
    armature_scale: tuple[float, float]  # x LEAP_REFLECTED_ARMATURE
    lag: tuple[int, int]  # command delay in physics substeps (5 ms)
    frictionloss_scale: tuple[float, float] = (0.5, 1.8)  # x the joint's dry friction, per reset
    damping_scale: tuple[float, float] = (0.6, 1.6)  # x the joint's passive damping, per reset


@dataclass(frozen=True)
class LeapContactDR:
    """Sliding-friction randomization of the hand's surfaces (one sample each per reset)."""

    pad_friction: tuple[float, float]  # the fingertip pads (robot/*_tip)
    other_friction: tuple[float, float]  # every other hand geom (printed links, palm) and the object


# The real LEAP driver's |command - measured| clamp per 80 Hz tick (its max_delta_q).
LEAP_DRIVER_LEAD_CLAMP = 0.15

_ALL = frozenset(_SCALES)

# 2025: the driver's current-domain gains mapped to torque, with all five per-joint CEM scales
# (Kp about 0.59 N m/rad). Too soft: the fit was clipped at its search bounds.
LEAP_SERVO_CURRENT_PD = LeapServo(
    name="current_pd_2025", kp=LEAP_PD_STIFFNESS_NM_PER_RAD, kd=LEAP_PD_DAMPING_NM_PER_RAD_S,
    effort=LEAP_ACTUATOR_EFFORT_LIMIT_NM, armature=LEAP_REFLECTED_ARMATURE, frictionloss=LEAP_NOMINAL_FRICTIONLOSS_NM,
    per_joint=_ALL,
)
# 2026-07: kp and kd set past the clipped fit (Kp 1.5, Kd 0.05), keeping the per-joint effort,
# armature and friction scales (the in-hand rotation / reorientation pipeline).
LEAP_SERVO_KP15_PERJOINT_2026_07 = replace(
    LEAP_SERVO_CURRENT_PD, name="kp1.5_perjoint_2026_07", kp=1.5, kd=0.05,
    per_joint=frozenset({"effort", "armature", "frictionloss"}),
)
# 2026-07: the same Kp 1.5 / Kd 0.05 with uniform effort, armature and friction (the xArm7
# composition; two actuator groups, flexion and rot).
LEAP_SERVO_KP15_UNIFORM_2026_07 = replace(
    LEAP_SERVO_CURRENT_PD, name="kp1.5_uniform_2026_07", kp=1.5, kd=0.05, per_joint=frozenset(),
)
# 2026-10: fitted to real LEAP commands replayed open-loop in simulation (2026-10-05): the
# real servos are faster and stiffer (Kp 4-8 N m/rad, 0-10 ms delay); Kp 4 / Kd 0.05. The
# sim-to-real (S2R) reorientation policies train on it with LEAP_S2R_2026_10_DR.
LEAP_SERVO_S2R_2026_10 = replace(LEAP_SERVO_KP15_PERJOINT_2026_07, name="s2r_2026_10", kp=4.0)
LEAP_S2R_2026_10_DR = LeapServoDR(
    kp_scale=(0.75, 1.5), kd_scale=(0.6, 2.0), effort=(0.28, 0.45), armature_scale=(0.6, 1.3), lag=(0, 5),
)
# The printed links, palm and the taped cube are slippery; the pads grip (the 2026-10-05 real runs).
LEAP_S2R_2026_10_CONTACT_DR = LeapContactDR(pad_friction=(0.6, 1.2), other_friction=(0.3, 0.8))



@dataclass(frozen=True)
class LeapGearServo:
    """One version of the measured LEAP servo model (gear_servo.py; units A, rad, s, N m)."""

    name: str
    current_gain: float  # G: servo current per rad of goal error, A/rad
    current_damping: float  # Gd: A s/rad
    current_limit: float  # the driver's Goal Current, A
    torque_constant: float  # K_t, N m/A, motor side (before the gear's friction)
    gear_friction: float  # mu: gear friction per N m of link-side load
    dry_friction: float  # f0, N m
    armature: float  # reflected rotor inertia, kg m^2
    viscous_damping: float  # N m s/rad
    hand_mass_scale: float  # x the XML hand's link masses and inertias
    rot_scale: float = LEAP_MCP_SIDE_GAIN_SCALE
    host_rate_hz: float = 80.0
    lead_clamp: float = LEAP_DRIVER_LEAD_CLAMP
    delay_min_lag: int = 0  # buffer bounds in physics steps; the DR event sets each env's lag
    delay_max_lag: int = 6
    # Low-speed stiction (a Karnopp band): + stiction_friction N m of dry friction on stiction_joints
    # (semantic names) while |qdot| < stiction_speed rad/s. Off before gear_2026_10b.
    stiction_friction: float = 0.0
    stiction_speed: float = 0.05
    stiction_joints: tuple[str, ...] = ()

    @property
    def has_stiction(self) -> bool:
        return bool(self.stiction_joints)


_STICTION_FIELDS = ("stiction_friction", "stiction_speed", "stiction_joints")


def gear_servo_fields(servo: LeapGearServo) -> dict:
    """A ``LeapGearServo`` version's fields (the deploy card's ``gear_servo`` block). The stiction
    fields are listed only for a version that has stiction, so the earlier versions' cards are
    unchanged."""
    d = dataclasses.asdict(servo)
    if not servo.has_stiction:
        for k in _STICTION_FIELDS:
            d.pop(k)
    return d


@dataclass(frozen=True)
class LeapGearServoDR:
    """Training randomization around a ``LeapGearServo`` version: its measured uncertainty.

    Sampled per env per reset (``in_hand_rotation_mjlab.mdp.randomize_leap_gear_servo``); the ``*_scale``
    ranges multiply the version's value, the others are absolute.
    """

    torque_constant: tuple[float, float]  # N m/A
    gear_friction: tuple[float, float]
    current_gain_scale: tuple[float, float]
    current_damping_scale: tuple[float, float]
    dry_friction_scale: tuple[float, float]
    armature_scale: tuple[float, float]
    damping_scale: tuple[float, float]
    lag: tuple[int, int]  # physics steps (5 ms)
    # N m, absolute, per stiction joint per reset (each joint its own sample); None: the version's value
    stiction_friction: tuple[float, float] | None = None


# 2026-10-07: identified on the real hand (rigid driver, current read with the joint state).
# In mjlab against the
# recordings: free-air sweeps of all 8 index and thumb joints track at 13.8 mrad rms (s2r_2026_10:
# 24.2); gauge pushes on the index pad at the same ~0.3 A take 4.7 N to push the finger in and
# 0.7 N while it follows back (real ~5 / <=1 N; s2r_2026_10: 2.3 / 1.6). K_t / (1 + mu) = 0.58
# N m/A is the datasheet's driving torque per amp. Known gap: stiction (a released finger can
# stop 0.076 rad short of its goal; this model stops 0.038 short).
LEAP_GEAR_SERVO_2026_10 = LeapGearServo(
    name="gear_2026_10", current_gain=2.0, current_damping=0.02, current_limit=LEAP_MOTOR_CURRENT_LIMIT_A,
    torque_constant=0.9, gear_friction=0.55, dry_friction=0.005, armature=0.001, viscous_damping=0.048,
    hand_mass_scale=0.724, delay_min_lag=0, delay_max_lag=6,
)
LEAP_GEAR_SERVO_2026_10_DR = LeapGearServoDR(
    torque_constant=(0.75, 1.0), gear_friction=(0.45, 0.65), current_gain_scale=(0.9, 1.1),
    current_damping_scale=(0.5, 1.5), dry_friction_scale=(0.5, 2.0), armature_scale=(0.8, 1.25),
    # lag: the fitted 10 ms plus the deployed policy's round trip (served over the network)
    damping_scale=(0.7, 1.4), lag=(1, 5),
)
# The version's nominal values, nothing randomized (play, evaluation against recordings).
LEAP_GEAR_SERVO_2026_10_NOMINAL = LeapGearServoDR(
    torque_constant=(0.9, 0.9), gear_friction=(0.55, 0.55), current_gain_scale=(1.0, 1.0),
    current_damping_scale=(1.0, 1.0), dry_friction_scale=(1.0, 1.0), armature_scale=(1.0, 1.0),
    damping_scale=(1.0, 1.0), lag=(2, 2),
)

# 2026-10-11: gear_2026_10 checked against the real in-hand deployments (6 runs, 20.9k policy
# steps). Two changes:
# - nominal gear friction mu 0.55 -> 0.50 (free MCP response and the PaXini backdrive ratio fit
#   0.45-0.5); the DR keeps 0.45-0.65;
# - the abduction joints stick at low speed: + 0.01 N m of dry friction while |qdot| < 0.05 rad/s
#   (DR 0-0.02 N m per joint). Real abduction joints do not move on 0.005-0.02 rad leads in 72% of
#   steps (gear_2026_10: 53%) and hold 1.3-1.7x larger offsets.
# With both, short-horizon prediction of the real free fingers: no-motion at 0.005-0.02 rad leads
# 58 / 48 / 85% (free / all / at rest; real 57 / 50 / 86%), abduction 74% (real 72%), MCP 58% (real
# 52%), rms 4.92 / 6.72 mrad at 50 / 100 ms (gear_2026_10: 4.97 / 6.83). Contact is not validated.
LEAP_GEAR_SERVO_2026_10B = replace(
    LEAP_GEAR_SERVO_2026_10, name="gear_2026_10b", gear_friction=0.50,
    stiction_friction=0.01, stiction_speed=0.05, stiction_joints=LEAP_MCP_SIDE_ROT_JOINTS,
)
LEAP_GEAR_SERVO_2026_10B_DR = replace(LEAP_GEAR_SERVO_2026_10_DR, stiction_friction=(0.0, 0.02))
LEAP_GEAR_SERVO_2026_10B_NOMINAL = replace(
    LEAP_GEAR_SERVO_2026_10_NOMINAL, gear_friction=(0.50, 0.50), stiction_friction=(0.01, 0.01),
)

# The current best model of the real hand: what new and retrained tasks should use.
LEAP_SERVO_LATEST = LEAP_GEAR_SERVO_2026_10B
LEAP_SERVOS = {
    s.name: s
    for s in (LEAP_SERVO_CURRENT_PD, LEAP_SERVO_KP15_PERJOINT_2026_07, LEAP_SERVO_KP15_UNIFORM_2026_07,
              LEAP_SERVO_S2R_2026_10, LEAP_GEAR_SERVO_2026_10, LEAP_GEAR_SERVO_2026_10B)
}
# Each gear version's training randomization and nominal plant (use_leap_gear_servo's default DR).
LEAP_GEAR_SERVO_DR = {"gear_2026_10": LEAP_GEAR_SERVO_2026_10_DR, "gear_2026_10b": LEAP_GEAR_SERVO_2026_10B_DR}
LEAP_GEAR_SERVO_NOMINAL = {"gear_2026_10": LEAP_GEAR_SERVO_2026_10_NOMINAL,
                           "gear_2026_10b": LEAP_GEAR_SERVO_2026_10B_NOMINAL}


def leap_actuators(servo: LeapServo | LeapGearServo,
                   joint_names: str = "semantic") -> tuple[IdealPdActuatorCfg | LeapGearServoActuatorCfg, ...]:
    """The 16 LEAP joints' actuators for one servo version.

    ``joint_names``: "semantic" (``if_mcp``..., the standalone hand XMLs) or "hardware"
    (``joint_0``..., the xArm7 composition). A ``LeapGearServo`` is one actuator over all 16
    joints. A ``LeapServo`` is IdealPd + command delay: one cfg per joint when any quantity is
    scaled per joint, otherwise two groups, flexion and rot.
    """
    assert joint_names in ("semantic", "hardware"), joint_names
    hw = joint_names == "hardware"
    if isinstance(servo, LeapGearServo):
        names = LEAP_HW_JOINT_NAMES if hw else LEAP_JOINT_ORDER
        rot = tuple(n for n in names if (LEAP_SEMANTIC_NAME[n] if hw else n) in LEAP_MCP_SIDE_ROT_JOINTS)
        # stiction fields only on a version with stiction: the earlier versions' cfgs stay as they were
        assert set(servo.stiction_joints) <= set(LEAP_JOINT_ORDER), servo.stiction_joints
        stiction = dict(stiction_friction=servo.stiction_friction, stiction_speed=servo.stiction_speed,
                        stiction_joints=tuple(n for n in names if (LEAP_SEMANTIC_NAME[n] if hw else n)
                                              in servo.stiction_joints)) if servo.has_stiction else {}
        return (LeapGearServoActuatorCfg(
            target_names_expr=names, current_gain=servo.current_gain, current_damping=servo.current_damping,
            current_limit=servo.current_limit, torque_constant=servo.torque_constant,
            gear_friction=servo.gear_friction, dry_friction=servo.dry_friction, armature=servo.armature,
            viscous_damping=servo.viscous_damping, side_joints=rot, side_gain_scale=servo.rot_scale,
            host_period=1.0 / servo.host_rate_hz, lead_clamp=servo.lead_clamp,
            delay_min_lag=servo.delay_min_lag, delay_max_lag=servo.delay_max_lag, delay_hold_prob=1.0,
            **stiction,
        ),)
    delay = dict(delay_min_lag=servo.delay_min_lag, delay_max_lag=servo.delay_max_lag,
                 delay_hold_prob=servo.delay_hold_prob, delay_update_period=servo.delay_update_period,
                 delay_per_env_phase=servo.delay_per_env_phase)
    if servo.per_joint:
        order = LEAP_HW_JOINT_NAMES if hw else LEAP_JOINT_ORDER
        return tuple(
            IdealPdActuatorCfg(
                target_names_expr=(name,),
                stiffness=servo.value("kp", sem), damping=servo.value("kd", sem), effort_limit=servo.value("effort", sem),
                armature=servo.value("armature", sem), frictionloss=servo.value("frictionloss", sem), **delay,
            )
            for name, sem in ((n, LEAP_SEMANTIC_NAME[n] if hw else n) for n in order)
        )
    names = LEAP_HW_JOINT_NAMES if hw else LEAP_JOINT_ORDER
    rot = tuple(n for n in names if (LEAP_SEMANTIC_NAME[n] if hw else n) in LEAP_MCP_SIDE_ROT_JOINTS)
    flex = tuple(n for n in names if n not in rot)
    common = dict(effort_limit=servo.effort, armature=servo.armature, frictionloss=servo.frictionloss, **delay)
    return (
        IdealPdActuatorCfg(target_names_expr=flex, stiffness=servo.kp, damping=servo.kd, **common),
        IdealPdActuatorCfg(target_names_expr=rot, stiffness=servo.kp * servo.rot_scale,
                           damping=servo.kd * servo.rot_scale, **common),
    )


def scale_hand_mass(spec, scale: float) -> mujoco.MjSpec:
    """Scale the mass and inertia of the LEAP hand's bodies: the palm and everything below it."""
    def in_hand(body):
        while body is not None and body.name != "world":
            if body.name.rsplit("/", 1)[-1] == "palm":
                return True
            body = body.parent
        return False
    for body in spec.bodies:
        if body.name != "world" and body.mass > 0.0 and in_hand(body):
            body.mass *= scale
            body.inertia = [v * scale for v in body.inertia]
            if any(body.fullinertia):
                body.fullinertia = [v * scale for v in body.fullinertia]
    return spec


def leap_hand_spec_fn(spec_fn, servo: LeapServo | LeapGearServo):
    """``spec_fn`` with the hand's link masses scaled to the version's (the identity for LeapServo).

    Idempotent: a spec_fn already scaled for some version is unwrapped first.
    """
    while isinstance(spec_fn, functools.partial) and spec_fn.func is _scaled_spec:
        spec_fn = spec_fn.args[0]
    scale = getattr(servo, "hand_mass_scale", 1.0)
    if scale == 1.0:
        return spec_fn
    return functools.partial(_scaled_spec, spec_fn, scale)


def _scaled_spec(spec_fn, scale: float) -> mujoco.MjSpec:
    return scale_hand_mass(spec_fn(), scale)


def leap_gear_servo_events(dr: LeapGearServoDR, servo: LeapGearServo = LEAP_GEAR_SERVO_2026_10,
                           entity: str = "robot", joint_names: tuple[str, ...] = (".*",)) -> dict:
    """The event terms a task on a ``LeapGearServo`` needs: the servo's own parameters and lag
    (``dr_leap_gear_servo``, required even when nothing is randomized), and the hand joints'
    armature (at startup) and viscous damping (per reset). ``joint_names`` selects the hand's
    joints when the entity has others (the xArm7 composition: ("joint_.*",))."""
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from in_hand_rotation_mjlab import mdp

    joints = SceneEntityCfg(entity, joint_names=joint_names)
    # the stiction range only where the DR has one (a version without it keeps its event params)
    assert dr.stiction_friction is None or servo.has_stiction, f"{servo.name} has no stiction joints to randomize"
    stiction = {} if dr.stiction_friction is None else dict(stiction_friction=dr.stiction_friction)
    return dict(
        dr_leap_gear_servo=EventTermCfg(func=mdp.randomize_leap_gear_servo, mode="reset", params=dict(
            torque_constant=dr.torque_constant, gear_friction=dr.gear_friction,
            current_gain_scale=dr.current_gain_scale, current_damping_scale=dr.current_damping_scale,
            dry_friction_scale=dr.dry_friction_scale, lag=dr.lag, asset_cfg=SceneEntityCfg(entity), **stiction)),
        dr_reflected_inertia=EventTermCfg(func=mdp.randomize_dof_armature, mode="startup", params=dict(
            ranges=tuple(k * servo.armature for k in dr.armature_scale), distribution="uniform", operation="abs",
            asset_cfg=joints)),
        dr_motor_damping=EventTermCfg(func=mdp.randomize_dof_damping, mode="reset", params=dict(
            ranges=dr.damping_scale, distribution="uniform", operation="scale", asset_cfg=joints)),
    )


# Randomization events that assume IdealPd hand actuators (gains, effort clip) or that the gear
# servo overrides (it rewrites the hand's dry friction every step, and samples its own lag).
_IDEAL_PD_ONLY_EVENTS = ("randomize_pd_gains", "set_actuator_effort_limits", "randomize_dof_frictionloss",
                         "sync_actuator_delays")


def use_leap_gear_servo(cfg, servo: LeapGearServo = LEAP_GEAR_SERVO_2026_10,
                        dr: LeapGearServoDR | None = None, entity: str = "robot"):
    """Move an env cfg's LEAP hand onto a ``LeapGearServo`` version (the measured hand).

    Works for the standalone hands and the xArm7 composition: the hand's IdealPd actuators
    become the one gear-servo actuator (other actuators, e.g. the arm's, stay), the hand's link
    masses take the version's scale, the IdealPd-only randomization events are dropped and the
    version's own events added (``leap_gear_servo_events``). The hand action's command-lead
    clamp, the observations and the rewards are the task's own. ``dr``: the randomization,
    default the version's own (``LEAP_GEAR_SERVO_DR``). On a cfg already on a gear version it
    swaps the version (evaluating a policy on another version). Returns ``cfg``.
    """
    if dr is None:
        assert servo.name in LEAP_GEAR_SERVO_DR, f"no training randomization for {servo.name}: pass dr="
        dr = LEAP_GEAR_SERVO_DR[servo.name]
    entity_cfg = cfg.scene.entities[entity]
    hand_names = set(LEAP_JOINT_ORDER) | set(LEAP_HW_JOINT_NAMES)
    is_hand = lambda a: isinstance(a, (IdealPdActuatorCfg, LeapGearServoActuatorCfg)) and set(a.target_names_expr) <= hand_names
    hand = [a for a in entity_cfg.articulation.actuators if is_hand(a)]
    assert hand, f"no LEAP hand actuators on {entity!r}"
    hw = any(n in LEAP_HW_JOINT_NAMES for a in hand for n in a.target_names_expr)
    others = tuple(a for a in entity_cfg.articulation.actuators if not is_hand(a))
    articulation = replace(entity_cfg.articulation,
                           actuators=(*others, *leap_actuators(servo, "hardware" if hw else "semantic")))
    cfg.scene.entities[entity] = replace(entity_cfg, articulation=articulation,
                                         spec_fn=leap_hand_spec_fn(entity_cfg.spec_fn, servo))
    for name in [k for k, ev in cfg.events.items() if getattr(ev.func, "__name__", "") in _IDEAL_PD_ONLY_EVENTS]:
        cfg.events.pop(name)
    cfg.events.update(leap_gear_servo_events(dr, servo, entity, (r"joint_\d+",) if hw else (".*",)))
    return cfg
