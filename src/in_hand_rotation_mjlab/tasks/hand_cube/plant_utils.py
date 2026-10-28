"""Plant helpers: pin or strip the hand's domain randomization (evaluation on a nominal plant), and the per-env
command delay of the servo model."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


DR_EVENTS = (
  "dr_cube_com",
  "dr_shared_contact_friction",
  "dr_cube_mass",
  "dr_robot_link_masses",
  "dr_motor_friction",
  "dr_motor_damping",
  "dr_reflected_inertia",
  "dr_robot_pd_gains",
  "dr_action_delay",
  "dr_effort_limits",
)


# The measured LEAP servo's reset event (in_hand_rotation_mjlab.mdp.randomize_leap_gear_servo). Not in DR_EVENTS:
# the gear servo needs it even unrandomized (it expands dof_frictionloss per env), so a nominal
# plant pins its ranges instead of dropping it. Its dr_motor_damping and dr_reflected_inertia
# share the IdealPd names above: dropped, the hand keeps the version's compiled values.
GEAR_SERVO_EVENT = "dr_leap_gear_servo"


# Nominal command delay [physics steps] = centre of the RL task's DR range (5..15).
NOMINAL_LAG = 10


# The gear servo's nominal command delay [physics steps]: the 10 ms fitted on the real hand, the
# lag of gear_2026_10's nominal (LEAP_GEAR_SERVO_2026_10_NOMINAL.lag). ``pin_gear_servo``'s default;
# a gear version with another fitted delay passes its own.
GEAR_NOMINAL_LAG = 2


def gear_servo_cfg(cfg):
  """The hand's ``LeapGearServoActuatorCfg`` in an env cfg, or None (IdealPd servo)."""
  from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import LeapGearServoActuatorCfg

  acts = cfg.scene.entities["robot"].articulation.actuators
  return next((a for a in acts if isinstance(a, LeapGearServoActuatorCfg)), None)


def pin_gear_servo(cfg, lag: int = GEAR_NOMINAL_LAG, **values: float) -> None:
  """Pin the gear servo's per-env parameters (its reset event's ranges) to one value each.

  The servo's law parameters default to the values of the version in ``cfg`` (its actuator cfg).
  The command delay ``lag`` [physics steps] is an explicit argument: the actuator cfg does not
  carry a nominal delay, so the default is GEAR_NOMINAL_LAG = 2 (10 ms, gear_2026_10's fitted
  delay). Defaults together = LEAP_GEAR_SERVO_2026_10_NOMINAL on a gear_2026_10 task.
  ``values`` override by event parameter: ``torque_constant``, ``gear_friction`` and ``stiction_friction``
  absolute, the ``*_scale`` ones relative to the version's, e.g. ``pin_gear_servo(cfg, torque_constant=0.75)``.
  A version with stiction joints (gear_2026_10b on) also pins ``stiction_friction`` to its value.
  """
  act = gear_servo_cfg(cfg)
  assert act is not None, "pin_gear_servo: the cfg's hand is not on the gear servo"
  assert act.delay_min_lag <= lag <= act.delay_max_lag, (
    f"lag {lag} outside the gear servo's delay buffer [{act.delay_min_lag}, {act.delay_max_lag}]")
  stiction = dict(stiction_friction=act.stiction_friction) if act.stiction_joints else {}
  pins = dict(torque_constant=act.torque_constant, gear_friction=act.gear_friction, current_gain_scale=1.0,
              current_damping_scale=1.0, dry_friction_scale=1.0, **stiction) | values
  params = cfg.events[GEAR_SERVO_EVENT].params
  params.update({k: (float(v), float(v)) for k, v in pins.items()})
  params["lag"] = (int(lag), int(lag))


def strip_plant_dr(cfg, extra: tuple[str, ...] = ()) -> list[str]:
  """Remove the plant's domain randomization from an env cfg (before the env is built): the
  DR_EVENTS and ``extra`` events are dropped, a gear servo is pinned to its nominal values
  (``pin_gear_servo``). Returns the dropped event names."""
  dropped = [n for n in (*DR_EVENTS, *extra) if cfg.events.pop(n, None) is not None]
  if gear_servo_cfg(cfg) is not None:
    pin_gear_servo(cfg)
  return dropped


def _delay_buffers(env: ManagerBasedRlEnv) -> list:
  """The hand's command-delay buffers: one per fused IdealPd group (aliased into each of its
  actuators) or per custom actuator (the gear servo)."""
  buffers = {}
  for act in env.scene["robot"].actuators:
    if act._delay_buffer is not None:
      buffers[id(act._delay_buffer)] = act._delay_buffer
  return list(buffers.values())


def set_lags(env: ManagerBasedRlEnv, lags: torch.Tensor, env_ids=None) -> None:
  for act in env.scene["robot"].actuators:
    if act.has_delay:
      act.set_lags(lags, env_ids)


def get_lags(env: ManagerBasedRlEnv) -> torch.Tensor:
  buffers = _delay_buffers(env)
  if buffers:
    return buffers[0]._current_lags.clone()
  return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
