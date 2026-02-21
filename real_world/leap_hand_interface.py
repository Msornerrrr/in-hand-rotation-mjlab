from config import CTRL_HZ, LEAP_CTRL_HZ
from hand_interface import HandInterfaceBase


class LeapHandInterface(HandInterfaceBase):
    mode_param_name = "/teleop_mode"
    cmd_topic_direct = "/leaphand_node/cmd_leap"
    cmd_topic_fabrics = "/hand_target"
    state_topic = "/leap_hand_state"

    publish_rate = LEAP_CTRL_HZ
    base_ctrl_hz = CTRL_HZ
