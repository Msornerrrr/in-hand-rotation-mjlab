from in_hand_rotation_mjlab.robots.leap_hand.leap_right_constants import (
    get_leap_hand_cfg as get_leap_right_hand_cfg,
)
from in_hand_rotation_mjlab.robots.leap_hand.leap_left_constants import (
    get_leap_left_hand_cfg as get_leap_left_hand_cfg,
)
from in_hand_rotation_mjlab.robots.leap_hand.leap_left_custom_constants import (
    get_leap_left_custom_hand_cfg as get_leap_left_custom_hand_cfg,
)

# Legacy alias for backward compatibility
get_leap_hand_cfg = get_leap_right_hand_cfg
