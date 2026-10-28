"""The SO(3) reorientation variants: each registered task's recipe (see ``__init__``)."""

import os

from in_hand_rotation_mjlab import PACKAGE_PATH
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import Sim2RealCfg, measured_gear_plant
from in_hand_rotation_mjlab.tasks.hand_cube.reorient_env_cfg import make_reorient_env_cfg

# The reverse curriculum's expert demonstrations (sampling-MPC planner states + the goals they reached), downloaded
# into <repo>/data/reorient_demos (see README: data); IHR_REORIENT_DEMOS points elsewhere.
_OUT = os.environ.get("IHR_REORIENT_DEMOS", str(PACKAGE_PATH.parents[1] / "data/reorient_demos"))
RC_DEMOS = {c: f"{_OUT}/demos_cube{c}cm/shard_*.npz" for c in ("5", "6", "7")}
_STAGE2 = dict(home_anchor=True, cube_obs_delay=True, peak_force=(-1e-3, 30.0, 10.0))
_WALL = {"5": (-200.0, 0.04), "6": (-1000.0, 0.04), "7": (-1000.0, 0.04)}  # the final recipe per size
S2R_FINGERS = Sim2RealCfg(torque_sat=(-2.0, 0.75), peak_force=(-4e-3, 5.0, 2.0))


def leap_left_custom_reorient_env_cfg(cube: str, play: bool = False, max_lead: float = 0.25,
                                      s2r: Sim2RealCfg | None = None, gear: bool = False, **recipe):
  """RC-Cube{cube} with ``recipe`` (``make_reorient_env_cfg`` arguments). The play cfg keeps
  only the command-lead clamp and the sim-to-real plant: it evaluates uniform SO(3) goals.
  ``gear``: on the measured servo (hand_cube_env_cfg.measured_gear_plant)."""
  if play:
    cfg = make_reorient_env_cfg(play=True, max_lead=max_lead, cube=cube, s2r=s2r)
  else:
    cfg = make_reorient_env_cfg(max_lead=max_lead, cube=cube, rc_glob=RC_DEMOS[cube], s2r=s2r, **recipe)
  return measured_gear_plant(cfg) if gear else cfg


def reorient_variants():
  """(task-id suffix, cube, recipe) of every registered variant."""
  for cube in ("5", "6", "7"):
    yield "", cube, {}
    yield "-Final", cube, dict(**_STAGE2, home_wall=(-200.0, 0.04))
    if cube != "5":
      yield "-FinalW", cube, dict(**_STAGE2, home_wall=(-1000.0, 0.04))
    yield "-S2R3", cube, dict(max_lead=0.15, s2r=S2R_FINGERS, **_STAGE2, home_wall=_WALL[cube])
    yield "-S2R3-Gear", cube, dict(max_lead=0.15, s2r=S2R_FINGERS, **_STAGE2, home_wall=_WALL[cube], gear=True)
