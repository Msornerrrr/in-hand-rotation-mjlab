from __future__ import annotations

from typing import Callable, TypeVar

from in_hand_rotation_mjlab.sim2sim.native.plugins.base import NativeTaskPlugin

TPlugin = TypeVar("TPlugin", bound=NativeTaskPlugin)

_PLUGIN_REGISTRY: list[tuple[Callable[[str], bool], Callable[[str], NativeTaskPlugin]]] = []


def register_native_task_plugin(
  matcher: Callable[[str], bool],
  factory: Callable[[str], NativeTaskPlugin],
) -> None:
  """Register a task plugin factory."""
  _PLUGIN_REGISTRY.append((matcher, factory))


def resolve_native_task_plugins(task_id: str) -> list[NativeTaskPlugin]:
  """Instantiate all plugins matching a task id."""
  plugins: list[NativeTaskPlugin] = []
  for matcher, factory in _PLUGIN_REGISTRY:
    if matcher(task_id):
      plugins.append(factory(task_id))
  return plugins
