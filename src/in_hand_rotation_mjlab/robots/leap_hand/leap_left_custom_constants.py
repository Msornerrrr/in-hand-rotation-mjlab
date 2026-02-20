"""
LEAP Hand (Left Custom) constants and EntityCfg for mjlab.

Uses left_hand_ours.xml (custom variant) with modified joint configuration.
Shares actuators and collision config with standard LEAP hands.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

from in_hand_rotation_mjlab.robots.leap_hand.leap_right_constants import (
    configure_leap_spec_for_ideal_pd,
    ARTICULATION,
    HOME_KEYFRAME as _BASE_HOME_KEYFRAME,
    LEAP_COLLISION,
)

# Custom 16-DoF default pose for left_hand_ours.xml:
# [if_mcp, if_rot, if_pip, if_dip,
#  mf_mcp, mf_rot, mf_pip, mf_dip,
#  rf_mcp, rf_rot, rf_pip, rf_dip,
#  th_cmc, th_axl, th_mcp, th_ipl]
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=_BASE_HOME_KEYFRAME.pos,
    joint_pos={
        # Index
        "if_mcp": 0.131,
        "if_rot": 0.0,
        "if_pip": 0.65,
        "if_dip": 1.0,
        # Middle
        "mf_mcp": 0.131,
        "mf_rot": 0.0,
        "mf_pip": 0.65,
        "mf_dip": 1.0,
        # Ring
        "rf_mcp": 0.131,
        "rf_rot": 0.0,
        "rf_pip": 0.65,
        "rf_dip": 1.0,
        # Thumb
        "th_cmc": 0.8,
        "th_axl": -0.78,
        "th_mcp": 0.5,
        "th_ipl": 0.367,
    },
    joint_vel=_BASE_HOME_KEYFRAME.joint_vel,
)


# -----------------------------------------------------------------------------
# MJCF + assets
# -----------------------------------------------------------------------------

leap_left_custom_hand_XML: Path = (
    MYMJLAB_SRC_PATH / "robots" / "leap_hand" / "xmls" / "left_hand_custom.xml"
)
assert leap_left_custom_hand_XML.exists(), f"Missing MJCF: {leap_left_custom_hand_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Embed mesh assets into MjSpec.assets."""
    assets: dict[str, bytes] = {}
    update_assets(
        assets,
        leap_left_custom_hand_XML.parent.parent / "assets",
        meshdir,
    )
    return assets


def get_spec() -> mujoco.MjSpec:
    """Create MjSpec and attach embedded assets."""
    spec = mujoco.MjSpec.from_file(str(leap_left_custom_hand_XML))
    configure_leap_spec_for_ideal_pd(spec)
    spec.assets = get_assets(spec.meshdir)
    return spec


# -----------------------------------------------------------------------------
# Final EntityCfg
# -----------------------------------------------------------------------------

def get_leap_left_custom_hand_cfg() -> EntityCfg:
    """Return mjlab EntityCfg for LEAP left hand (custom variant)."""
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

    leap = Entity(get_leap_left_custom_hand_cfg())
    mujoco.viewer.launch(leap.spec.compile())
