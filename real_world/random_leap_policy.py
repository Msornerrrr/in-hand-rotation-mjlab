#!/usr/bin/env python
"""Random policy player for LEAP hand. Adds small random deltas to joint positions."""

import rospy
import numpy as np
from leap_hand_interface import LeapHandInterface
from config import CTRL_HZ

SCALE = 1 / 24
NUM_JOINTS = 16


def main():
    rospy.init_node("random_leap_policy", anonymous=True)
    hand = LeapHandInterface()

    # Wait for initial state
    rospy.loginfo("Waiting for initial LEAP hand state ...")
    start = rospy.Time.now().to_sec()
    timeout_s = 10.0
    while not rospy.is_shutdown():
        state = hand.get_state()
        if state is not None:
            break
        if rospy.Time.now().to_sec() - start > timeout_s:
            raise RuntimeError("Timed out waiting for LEAP hand state.")
        rospy.sleep(0.02)

    pos = hand.get_joint_positions_numpy().astype(np.float64)
    rospy.loginfo("Got initial joint positions: %s", pos)

    if pos.shape[0] != NUM_JOINTS:
        raise RuntimeError(f"Expected {NUM_JOINTS} joints, got {pos.shape[0]}.")

    rate = rospy.Rate(CTRL_HZ)
    while not rospy.is_shutdown():
        delta = np.random.uniform(-1.0, 1.0, size=pos.shape[0]) * SCALE
        pos = pos + delta

        hand.send_command(pos)

        rate.sleep()


if __name__ == "__main__":
    main()
