"""LEAP Left Custom hand cube rotation environment configuration."""

from in_hand_rotation_mjlab.robots import get_leap_left_custom_hand_cfg
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg


# LEAP Left Custom specific parameters
GRASP_INIT_JOINT_POS = {
    "if_mcp": 0.1,
    "if_rot": 0.4,
    "if_pip": 1.3,
    "if_dip": 0.0,
    "mf_mcp": 0.1,
    "mf_rot": 0.0,
    "mf_pip": 1.3,
    "mf_dip": 0.0,
    "rf_mcp": 0.1,
    "rf_rot": -0.4,
    "rf_pip": 1.3,
    "rf_dip": 0.0,
    "th_cmc": 1.45,
    "th_axl": -1.5,
    "th_mcp": 0.579,
    "th_ipl": 1.37,
}

CUBE_SPAWN_POS = (-0.092, 0.055, 0.27)
GRASP_CACHE_FILE = "tasks/hand_cube/cache/leap_left_custom_grasp_cache.npz"


def leap_left_custom_hand_cube_rotate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create LEAP Left Custom hand cube rotation environment configuration."""
    return make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_leap_left_custom_hand_cfg,
        grasp_init_joint_pos=GRASP_INIT_JOINT_POS,
        cube_spawn_pos=CUBE_SPAWN_POS,
        grasp_cache_file=GRASP_CACHE_FILE,
        negate_yaw_rate=True,  # Left hand variant: clockwise rotation is positive
        play=play,
    )
