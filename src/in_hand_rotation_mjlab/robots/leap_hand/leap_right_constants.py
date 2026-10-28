"""
LEAP Hand (Right) constants and EntityCfg for mjlab.

Goals:
- Spawn LEAP RH stably
- Use mjlab IdealPd actuators from a versioned servo model (leap_plant.py)
- Embed mesh assets for MuJoCo Warp + Viser
- No task / policy logic
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from in_hand_rotation_mjlab import PACKAGE_PATH
from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import LEAP_SERVO_CURRENT_PD, LeapServo, leap_actuators, LeapGearServo, leap_hand_spec_fn
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg

def update_assets(assets: dict[str, bytes], assets_dir: Path, meshdir: str | None) -> None:
    """Embed mesh assets into the assets dictionary for MuJoCo."""
    mesh_path = assets_dir if meshdir is None else assets_dir / meshdir

    if not mesh_path.exists():
        mesh_path = assets_dir

    for path in mesh_path.rglob("*"):
        if path.is_file():
            assets[path.name] = path.read_bytes()


def configure_leap_spec_for_ideal_pd(spec: mujoco.MjSpec) -> None:
    """Remove XML actuators so mjlab IdealPd actuators are the only actuators."""
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)


# -----------------------------------------------------------------------------
# MJCF + assets
# -----------------------------------------------------------------------------

leap_hand_XML: Path = (
    PACKAGE_PATH / "robots" / "leap_hand" / "xmls" / "right_hand.xml"
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
        leap_hand_XML.parent / "assets",
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
# Actuators: ideal PD from a versioned servo model (leap_plant.py: gains, effort,
# armature, friction, delay)
# -----------------------------------------------------------------------------
def leap_articulation(servo: LeapServo | LeapGearServo = LEAP_SERVO_CURRENT_PD) -> EntityArticulationInfoCfg:
    """The standalone LEAP hand's articulation for one servo-model version (leap_plant.py)."""
    return EntityArticulationInfoCfg(
        actuators=leap_actuators(servo),
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

def get_leap_hand_cfg(servo: LeapServo | LeapGearServo = LEAP_SERVO_CURRENT_PD) -> EntityCfg:
    """Return mjlab EntityCfg for LEAP right hand. ``servo``: the servo-model version (leap_plant.py)."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(LEAP_COLLISION,),
        spec_fn=leap_hand_spec_fn(get_spec, servo),
        articulation=leap_articulation(servo),
    )


# -----------------------------------------------------------------------------
# Debug entrypoint (native MuJoCo viewer)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    leap = Entity(get_leap_hand_cfg())
    mujoco.viewer.launch(leap.spec.compile())
