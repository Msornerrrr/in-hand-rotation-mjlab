"""Compatibility wrapper for native MuJoCo sim2sim runner.

Preferred import path is `my_mjlab.sim2sim.native`.
"""

from in_hand_rotation_mjlab.sim2sim.native.config import (
  NativeSim2SimConfig as NativeSim2SimConfig,
)
from in_hand_rotation_mjlab.sim2sim.native.runner import (
  NativeMujocoSim2SimRunner as NativeMujocoSim2SimRunner,
)
from in_hand_rotation_mjlab.sim2sim.native.runner import (
  run_native_sim2sim as run_native_sim2sim,
)
