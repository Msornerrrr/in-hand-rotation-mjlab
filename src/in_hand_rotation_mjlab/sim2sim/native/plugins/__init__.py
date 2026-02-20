from in_hand_rotation_mjlab.sim2sim.native.plugins.base import NativeTaskPlugin as NativeTaskPlugin
from in_hand_rotation_mjlab.sim2sim.native.plugins.registry import (
  register_native_task_plugin as register_native_task_plugin,
)
from in_hand_rotation_mjlab.sim2sim.native.plugins.registry import (
  resolve_native_task_plugins as resolve_native_task_plugins,
)

# Import task plugins for side-effect registration.
from . import hand_cube as _hand_cube
