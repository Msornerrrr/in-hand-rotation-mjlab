from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
  from in_hand_rotation_mjlab.sim2sim.native.observations import _ObservationAdapter
  from in_hand_rotation_mjlab.sim2sim.native.client import NativeMujocoClient


class NativeTaskPlugin:
  """Extension hook for task-specific native sim2sim logic."""

  def apply_event(
    self,
    runner: NativeMujocoClient,
    term_name: str,
    term_cfg: Any,
    params: dict[str, Any],
  ) -> bool:
    """Handle one event term.

    Returns:
      True if handled by this plugin, otherwise False.
    """
    del runner, term_name, term_cfg, params
    return False

  def build_observation_evaluator(
    self,
    adapter: _ObservationAdapter,
    func: Any,
    params: dict[str, Any],
  ) -> Callable[[], Any] | None:
    """Return an observation evaluator for task-specific observation terms."""
    del adapter, func, params
    return None
