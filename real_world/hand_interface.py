import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class HandInterfaceBase:
    """
    Shared interface for robot hands (Leap, Allegro, etc.).
    Handles:
        - ROS publishers/subscribers
        - direct vs fabrics mode
        - interpolation & upsampling
        - state tracking
    Subclasses override:
        - mode_param_name
        - cmd_topic_direct
        - cmd_topic_fabrics
        - state_topic
        - publish_rate
        - ctrl_hz
    """

    # ----- Subclasses MUST set these attributes -----
    mode_param_name = None  # e.g. "/teleop_mode"
    cmd_topic_direct = None
    cmd_topic_fabrics = None
    state_topic = None
    publish_rate = None  # HIGH-frequency control (e.g. 40 Hz)
    base_ctrl_hz = None  # Low-frequency update (e.g. 10 Hz)

    def __init__(self):
        # ----------------------------
        # Discover mode
        # ----------------------------
        if self.mode_param_name is None:
            raise ValueError("Subclass must define mode_param_name")

        self.mode = rospy.get_param(self.mode_param_name, "direct")
        self.last_state = None

        # ----------------------------
        # Publishers
        # ----------------------------
        if self.mode == "direct":
            if self.cmd_topic_direct is None:
                raise ValueError("cmd_topic_direct not defined in subclass")

            self.pub_cmd = rospy.Publisher(
                self.cmd_topic_direct, JointState, queue_size=10
            )
            self.target_pub = None

        elif self.mode == "fabrics":
            if self.cmd_topic_fabrics is None:
                raise ValueError("cmd_topic_fabrics not defined in subclass")

            self.pub_cmd = None
            self.target_pub = rospy.Publisher(
                self.cmd_topic_fabrics, Float64MultiArray, queue_size=10
            )
        else:
            raise ValueError(f"Unknown control mode: {self.mode}")

        # ----------------------------
        # Subscribe to joint states
        # ----------------------------
        if self.state_topic is None:
            raise ValueError("state_topic not defined in subclass")

        rospy.Subscriber(self.state_topic, JointState, self._cb_state)

        # ----------------------------
        # Upsampling / interpolation
        # ----------------------------
        if self.publish_rate is None or self.base_ctrl_hz is None:
            raise ValueError("Subclass must define publish_rate and base_ctrl_hz")

        self.upsample_factor = int(self.publish_rate / self.base_ctrl_hz)
        self.prev_cmd = None
        self.next_cmd = None
        self.phase = 0

        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._publish_interpolated)

    # ============================================================
    # State callback
    # ============================================================
    def _cb_state(self, msg: JointState):
        self.last_state = msg

    # ============================================================
    # Low-level command sender
    # ============================================================
    def _send_command(self, q):
        if self.mode == "direct":
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = [f"joint_{i}" for i in range(len(q))]
            msg.position = list(map(float, q))
            self.pub_cmd.publish(msg)
        else:
            msg = Float64MultiArray()
            msg.data = list(map(float, q))
            self.target_pub.publish(msg)

    # ============================================================
    # High-level command API
    # ============================================================
    def send_command(self, q):
        """External calls at low rate (10 Hz)."""
        q = np.asarray(q, dtype=np.float32)

        if self.upsample_factor <= 1:
            self._send_command(q)
            return

        if self.next_cmd is None:
            self.prev_cmd = q.copy()
            self.next_cmd = q.copy()
        else:
            self.prev_cmd = self.next_cmd.copy()
            self.next_cmd = q.copy()

        self.phase = 0

    def _publish_interpolated(self, event):
        if self.prev_cmd is None or self.next_cmd is None:
            return

        if self.upsample_factor <= 1:
            return

        if self.phase >= self.upsample_factor:
            self.prev_cmd = self.next_cmd
            self.next_cmd = None
            self.phase = 0
            return

        alpha = min(1.0, self.phase / float(self.upsample_factor - 1))
        q_interp = (1 - alpha) * self.prev_cmd + alpha * self.next_cmd

        self._send_command(q_interp)
        self.phase += 1

    # ============================================================
    # State accessors
    # ============================================================
    def get_state(self):
        return self.last_state

    def get_joint_positions_numpy(self):
        if self.last_state is None:
            raise RuntimeError("No joint state received yet.")
        return np.asarray(self.last_state.position, dtype=np.float32)

    def get_joint_velocities_numpy(self):
        if self.last_state is None or not self.last_state.velocity:
            raise RuntimeError("No velocity available.")
        return np.asarray(self.last_state.velocity, dtype=np.float32)

    def get_joint_efforts_numpy(self):
        if self.last_state is None or not self.last_state.effort:
            raise RuntimeError("No effort available.")
        return np.asarray(self.last_state.effort, dtype=np.float32)
