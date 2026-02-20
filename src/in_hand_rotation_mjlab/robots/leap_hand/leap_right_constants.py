"""
LEAP Hand (Right) constants and EntityCfg for mjlab.

Goals:
- Spawn LEAP RH stably
- Use mjlab IdealPd actuators with current-mapped torque limits
- Embed mesh assets for MuJoCo Warp + Viser
- No task / policy logic
"""

from __future__ import annotations

from pathlib import Path
from math import pi

import mujoco

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from mjlab.actuator import DelayedActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


# -----------------------------------------------------------------------------
# Actuator SysID constants (DYNAMIXEL XC330-M288)
# -----------------------------------------------------------------------------
#
# We use an ideal PD actuator model with current-domain interpretation:
#   i_des = Kp_i * (q_des - q) - Kd_i * qdot
#   tau_des = K_t * i_des
#
# Datasheet source:
# https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/
#
# Note:
# - We treat Current Limit value as mA (e.g., 550 -> 0.55 A).
# - MuJoCo actuator gains are configured in torque domain:
#     Kp_tau = K_t * Kp_i
#     Kd_tau = K_t * Kd_i
#
LEAP_MOTOR_TORQUE_PER_AMP_NM = 0.53
LEAP_MOTOR_CURRENT_LIMIT_MA = 550.0
LEAP_MOTOR_CURRENT_LIMIT_A = LEAP_MOTOR_CURRENT_LIMIT_MA / 1000.0
LEAP_ACTUATOR_EFFORT_LIMIT_NM = (
    LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_MOTOR_CURRENT_LIMIT_A
)

# Current-domain PD gains requested:
#   mI = 800 * joint_error + 200 * d_joint_error
# where mI is in mA, joint_error is in rad, d_joint_error is in rad/s.
LEAP_PD_CURRENT_KP_MA_PER_RAD = 800.0
LEAP_PD_CURRENT_KD_MA_PER_RAD_S = 200.0
LEAP_PD_CURRENT_KP_A_PER_RAD = LEAP_PD_CURRENT_KP_MA_PER_RAD / 1000.0
LEAP_PD_CURRENT_KD_A_PER_RAD_S = LEAP_PD_CURRENT_KD_MA_PER_RAD_S / 1000.0

LEAP_PD_STIFFNESS_NM_PER_RAD = (
    LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KP_A_PER_RAD
)
LEAP_PD_DAMPING_NM_PER_RAD_S = (
    LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KD_A_PER_RAD_S
)

# Match hardware MCP-side behavior in this sim joint convention:
# apply reduced gains on rotational MCP joints {if_rot, mf_rot, rf_rot}.
LEAP_MCP_SIDE_GAIN_SCALE = 0.75
LEAP_MCP_SIDE_PD_STIFFNESS_NM_PER_RAD = (
    LEAP_PD_STIFFNESS_NM_PER_RAD * LEAP_MCP_SIDE_GAIN_SCALE
)
LEAP_MCP_SIDE_PD_DAMPING_NM_PER_RAD_S = (
    LEAP_PD_DAMPING_NM_PER_RAD_S * LEAP_MCP_SIDE_GAIN_SCALE
)

# Reflected armature estimate from LEAP hand SysID note:
# Ia = kg^2 * Ir, with kg=288.35, Ir=1.7e-8 kg m^2.
LEAP_GEAR_RATIO = 288.35
LEAP_ROTOR_INERTIA = 1.7e-8
LEAP_REFLECTED_ARMATURE = (LEAP_GEAR_RATIO**2) * LEAP_ROTOR_INERTIA

# Nominal dry-friction estimate: 10% of max torque (paper suggestion).
LEAP_NOMINAL_FRICTIONLOSS_NM = 0.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM

# Delay wrapper bounds (in physics steps).
# The env reset event sets the actual lag each episode (e.g., 5..15 for
# decimation=10); keep bounds wide enough to avoid clamping.
LEAP_ACTION_DELAY_MIN_LAG = 0
LEAP_ACTION_DELAY_MAX_LAG = 20
LEAP_ACTION_DELAY_HOLD_PROB = 1.0
LEAP_ACTION_DELAY_UPDATE_PERIOD = 0
LEAP_ACTION_DELAY_PER_ENV_PHASE = True


def configure_leap_spec_for_ideal_pd(spec: mujoco.MjSpec) -> None:
    """Remove XML actuators so mjlab IdealPd actuators are the only actuators."""
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)


# -----------------------------------------------------------------------------
# MJCF + assets
# -----------------------------------------------------------------------------

leap_hand_XML: Path = (
    MYMJLAB_SRC_PATH / "robots" / "leap_hand" / "xmls" / "right_hand.xml"
)
assert leap_hand_XML.exists(), f"Missing MJCF: {leap_hand_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """
    Embed mesh assets into MjSpec.assets.

    Required for:
    - mujoco-warp
    - viser
    - distributed viewers
    """
    assets: dict[str, bytes] = {}
    update_assets(
        assets,
        leap_hand_XML.parent.parent / "assets",
        meshdir,
    )
    return assets


def get_spec() -> mujoco.MjSpec:
    """Create MjSpec and attach embedded assets."""
    spec = mujoco.MjSpec.from_file(str(leap_hand_XML))
    configure_leap_spec_for_ideal_pd(spec)
    spec.assets = get_assets(spec.meshdir)
    return spec


# -----------------------------------------------------------------------------
# Joint naming (16 DoF)
# -----------------------------------------------------------------------------
# Index / Middle / Ring / Thumb
LEAP_JOINTS_EXPR = (
    "if_(mcp|rot|pip|dip)",
    "mf_(mcp|rot|pip|dip)",
    "rf_(mcp|rot|pip|dip)",
    "th_(cmc|axl|mcp|ipl)",
)


# -----------------------------------------------------------------------------
# Actuators (ideal PD, current-mapped)
# Per-joint scales below come from current index+middle SysID artifacts:
# - index_175818_sysid_fit.npz (joints 0..3)
# - middle_180246_sysid_fit.npz (joints 4..7)
# Ring/thumb remain at baseline (scale=1.0) until calibrated.
# -----------------------------------------------------------------------------
LEAP_JOINT_ORDER = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)

LEAP_MCP_SIDE_ROT_JOINTS = ("if_rot", "mf_rot", "rf_rot")

LEAP_STIFFNESS_SCALE_BY_JOINT = {
    "if_mcp": 1.3821,
    "if_rot": 1.3903,
    "if_pip": 1.3601,
    "if_dip": 1.4000,
    "mf_mcp": 1.3860,
    "mf_rot": 1.4000,
    "mf_pip": 1.3462,
    "mf_dip": 1.4000,
    "rf_mcp": 1.3983,
    "rf_rot": 1.4000,
    "rf_pip": 1.4000,
    "rf_dip": 1.3490,
    "th_cmc": 1.3544,
    "th_axl": 1.2862,
    "th_mcp": 1.3870,
    "th_ipl": 1.3084,
}
LEAP_DAMPING_SCALE_BY_JOINT = {
    "if_mcp": 0.7264,
    "if_rot": 0.7000,
    "if_pip": 0.7000,
    "if_dip": 0.7699,
    "mf_mcp": 0.8130,
    "mf_rot": 0.7281,
    "mf_pip": 0.7918,
    "mf_dip": 0.7000,
    "rf_mcp": 0.7887,
    "rf_rot": 0.7251,
    "rf_pip": 0.7000,
    "rf_dip": 0.8001,
    "th_cmc": 0.7000,
    "th_axl": 0.7064,
    "th_mcp": 0.9578,
    "th_ipl": 0.7178,
}
LEAP_EFFORT_SCALE_BY_JOINT = {
    "if_mcp": 0.8500,
    "if_rot": 1.0684,
    "if_pip": 0.8500,
    "if_dip": 0.8500,
    "mf_mcp": 0.8621,
    "mf_rot": 1.0606,
    "mf_pip": 0.8500,
    "mf_dip": 0.8500,
    "rf_mcp": 0.8892,
    "rf_rot": 0.8500,
    "rf_pip": 0.8500,
    "rf_dip": 1.1062,
    "th_cmc": 0.8500,
    "th_axl": 0.8633,
    "th_mcp": 0.8612,
    "th_ipl": 1.0408,
}
LEAP_ARMATURE_SCALE_BY_JOINT = {
    "if_mcp": 1.2629,
    "if_rot": 1.4800,
    "if_pip": 1.1925,
    "if_dip": 0.6543,
    "mf_mcp": 0.8186,
    "mf_rot": 1.0668,
    "mf_pip": 1.3619,
    "mf_dip": 1.1253,
    "rf_mcp": 0.6000,
    "rf_rot": 1.2074,
    "rf_pip": 1.1534,
    "rf_dip": 1.3067,
    "th_cmc": 0.9726,
    "th_axl": 0.8835,
    "th_mcp": 1.0873,
    "th_ipl": 1.1179,
}
LEAP_FRICTION_SCALE_BY_JOINT = {
    "if_mcp": 1.3455,
    "if_rot": 0.7058,
    "if_pip": 1.1626,
    "if_dip": 0.3881,
    "mf_mcp": 0.9242,
    "mf_rot": 0.3264,
    "mf_pip": 0.9817,
    "mf_dip": 1.5468,
    "rf_mcp": 1.8238,
    "rf_rot": 1.0831,
    "rf_pip": 0.9794,
    "rf_dip": 0.8705,
    "th_cmc": 0.3610,
    "th_axl": 2.2000,
    "th_mcp": 0.5970,
    "th_ipl": 1.8449,
}


def _base_stiffness_for_joint(joint_name: str) -> float:
    if joint_name in LEAP_MCP_SIDE_ROT_JOINTS:
        return LEAP_MCP_SIDE_PD_STIFFNESS_NM_PER_RAD
    return LEAP_PD_STIFFNESS_NM_PER_RAD


def _base_damping_for_joint(joint_name: str) -> float:
    if joint_name in LEAP_MCP_SIDE_ROT_JOINTS:
        return LEAP_MCP_SIDE_PD_DAMPING_NM_PER_RAD_S
    return LEAP_PD_DAMPING_NM_PER_RAD_S


def _scaled(value: float, scale_map: dict[str, float], joint_name: str) -> float:
    return value * scale_map.get(joint_name, 1.0)


def _make_joint_actuator_cfg(joint_name: str) -> DelayedActuatorCfg:
    base_cfg = IdealPdActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=_scaled(
            _base_stiffness_for_joint(joint_name),
            LEAP_STIFFNESS_SCALE_BY_JOINT,
            joint_name,
        ),
        damping=_scaled(
            _base_damping_for_joint(joint_name),
            LEAP_DAMPING_SCALE_BY_JOINT,
            joint_name,
        ),
        effort_limit=_scaled(
            LEAP_ACTUATOR_EFFORT_LIMIT_NM,
            LEAP_EFFORT_SCALE_BY_JOINT,
            joint_name,
        ),
        armature=_scaled(
            LEAP_REFLECTED_ARMATURE,
            LEAP_ARMATURE_SCALE_BY_JOINT,
            joint_name,
        ),
        frictionloss=_scaled(
            LEAP_NOMINAL_FRICTIONLOSS_NM,
            LEAP_FRICTION_SCALE_BY_JOINT,
            joint_name,
        ),
    )
    return DelayedActuatorCfg(
        base_cfg=base_cfg,
        delay_target="position",
        delay_min_lag=LEAP_ACTION_DELAY_MIN_LAG,
        delay_max_lag=LEAP_ACTION_DELAY_MAX_LAG,
        delay_hold_prob=LEAP_ACTION_DELAY_HOLD_PROB,
        delay_update_period=LEAP_ACTION_DELAY_UPDATE_PERIOD,
        delay_per_env_phase=LEAP_ACTION_DELAY_PER_ENV_PHASE,
    )


LEAP_ACTUATORS = tuple(_make_joint_actuator_cfg(name) for name in LEAP_JOINT_ORDER)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=LEAP_ACTUATORS,
    soft_joint_pos_limit_factor=0.95,
)


# -----------------------------------------------------------------------------
# Initial pose (open, non-singular)
# -----------------------------------------------------------------------------
# Conservative "open hand" pose:
# - MCP slightly flexed
# - DIP/PIP lightly bent
# - Thumb in neutral opposition
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.1),
    joint_pos={
        # Index
        "if_mcp": 0.3,
        "if_rot": 0.0,
        "if_pip": 0.3,
        "if_dip": 0.2,
        # Middle
        "mf_mcp": 0.3,
        "mf_rot": 0.0,
        "mf_pip": 0.3,
        "mf_dip": 0.2,
        # Ring
        "rf_mcp": 0.3,
        "rf_rot": 0.0,
        "rf_pip": 0.3,
        "rf_dip": 0.2,
        # Thumb
        "th_cmc": 0.4,
        "th_axl": 0.2,
        "th_mcp": 0.3,
        "th_ipl": 0.2,
    },
    joint_vel={".*": 0.0},
)


# -----------------------------------------------------------------------------
# Collision configuration
# -----------------------------------------------------------------------------
LEAP_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*", ".*_tip"),

    # Per-finger bitmask channels:
    #   bit-0 (1)  = external objects
    #   bit-1 (2)  = index finger
    #   bit-2 (4)  = middle finger
    #   bit-3 (8)  = ring finger
    #   bit-4 (16) = thumb
    #
    # Same-finger pairs share a contype bit that is absent from their own
    # conaffinity, so (contype & conaffinity) == 0 → no intra-finger collision.
    # Cross-finger pairs have different bits → collision enabled.
    # All geoms accept external contacts via bit-0 in conaffinity.
    contype={
        "palm_collision.*": 1,   # external only
        "if_.*": 2,              # index
        "mf_.*": 4,              # middle
        "rf_.*": 8,              # ring
        "th_.*": 16,             # thumb
    },

    conaffinity={
        "palm_collision.*": 1,   # external only
        "if_.*": 29,             # 1 | 4 | 8 | 16  (ext + mid + ring + thumb)
        "mf_.*": 27,             # 1 | 2 | 8 | 16  (ext + idx + ring + thumb)
        "rf_.*": 23,             # 1 | 2 | 4 | 16  (ext + idx + mid + thumb)
        "th_.*": 15,             # 1 | 2 | 4 | 8   (ext + idx + mid + ring)
    },

    # Fingertips get rich contact, others soft
    condim={
        ".*_tip": 6,
        ".*": 3,
    },

    friction={
        ".*_tip": (0.8, 5e-3, 1e-4),
        ".*": (0.2,),
    },

    solref={
        ".*_tip": (0.01, 1),
        ".*": (0.05, 1),
    },

    priority={
        ".*_tip": 2,
        ".*": 0,
    },

    disable_other_geoms=False,
)


# -----------------------------------------------------------------------------
# Final EntityCfg
# -----------------------------------------------------------------------------

def get_leap_hand_cfg() -> EntityCfg:
    """Return mjlab EntityCfg for LEAP right hand."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(LEAP_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# -----------------------------------------------------------------------------
# Debug entrypoint (native MuJoCo viewer)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    leap = Entity(get_leap_hand_cfg())
    mujoco.viewer.launch(leap.spec.compile())
