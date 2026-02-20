from __future__ import annotations

import copy
from typing import Any

import mujoco

from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from in_hand_rotation_mjlab.policy_server.contracts import ActionTermMetadata


def build_server_action_metadata_from_task(
  *,
  task_id: str,
  play: bool,
  device: str = "cpu",
) -> dict[str, ActionTermMetadata]:
  """Build canonical action metadata from task config on the server side."""
  # Lazy import to avoid import cycles:
  # policy_server -> sim2sim imports policy_server.
  from in_hand_rotation_mjlab.sim2sim.native.actions import _ActionAdapter
  from in_hand_rotation_mjlab.sim2sim.native.state import _NativeState

  env_cfg = load_env_cfg(task_id, play=play)
  scene_cfg = copy.deepcopy(env_cfg.scene)
  scene_cfg.num_envs = 1

  scene = Scene(scene_cfg, device="cpu")
  model = scene.compile()
  env_cfg.sim.mujoco.apply(model)
  data = mujoco.MjData(model)

  state = _NativeState(scene=scene, model=model, data=data, device=device)
  action_adapter = _ActionAdapter(env_cfg.actions, state)
  return action_adapter.build_server_action_metadata()

