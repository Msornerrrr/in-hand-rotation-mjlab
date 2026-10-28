"""The LEAP hand's measured actuator: a Dynamixel XC330-M288 under the rigid LEAP driver.

Identified on the real hand (2026-10-07): free-air sweeps of
every index and thumb joint, gauge pushes on the index pad, and motor current read alongside.
Per physics step, for every hand joint:

  driver   every 12.5 ms (80 Hz, a random phase per env) the driver samples the commanded
           pose and writes goal = command clamped to measured q +- 0.15 rad (the driver's
           ``max_delta_q``); the goal is held between ticks
  servo    current I = clip(G (goal - q) - Gd qdot, +-I_max)   (Dynamixel current-based
           position mode; the abduction joints run at 0.75 x G and Gd)
  motor    torque = K_t I, applied before the gearbox
  gearbox  dry friction proportional to the torque the gear carries from the link side:
           frictionloss = f0 + mu |gravity + contacts + joint limits + applied forces and body
           wrenches|, updated every step from the previous step's forces. Driving, the joint delivers K_t I / (1 + mu); backdriven,
           it resists K_t I / (1 - mu): ~1 N vs ~5 N at the index pad for the same current.
  stiction optional, per joint (``stiction_joints``; gear_2026_10b: the abduction joints): + f_s of
           dry friction while |qdot| < v_s (a Karnopp band at low speed), so a joint at rest needs a
           larger lead to break away than a moving one keeps going on

The finger motors carry their links' weight: when the scene gravity-compensates the hand
(the xArm7 composition's ``realistic_arm``, which models the arm firmware holding its payload),
the actuator removes that compensation from its own joints (``carry_link_weight``).

The rotor inertia (armature) and viscous damping are passive joint properties. The friction
rows are made stiff (``friction_solref`` / ``friction_solimp``) so a joint held below
breakaway does not creep. The command delay is mjlab's per-actuator delay buffer.

One actuator covers all 16 joints, so the whole law is a handful of batched tensor ops per
step. A task using it needs the reset event ``in_hand_rotation_mjlab.mdp.randomize_leap_gear_servo``,
which expands ``dof_frictionloss`` per env (the friction is written per env every step)
and randomizes the law's parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd, TransmissionType
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
    from mjlab.entity import Entity

_FRICTION_DOF = 1  # mujoco_warp ConstraintType.FRICTION_DOF; efc.id is the dof

# Per-env parameters of the law, in the order the DR event addresses them.
GEAR_SERVO_PARAMS = ("current_gain", "current_damping", "current_limit", "torque_constant", "gear_friction",
                     "dry_friction", "stiction_friction")
# Per-env tensors that, with the command-delay buffer, are the actuator's whole state (snapshot / restore).
# The per-env tensors that evolve during a rollout (driver goal, tick clock, current): the state a planner copies
# between worlds without copying their randomized law parameters (hand_cube.sampling.rollout.LeanSim).
GEAR_SERVO_DYNAMIC_STATE = ("current", "_goal", "_clock", "_tick_now")
GEAR_SERVO_STATE = (*GEAR_SERVO_PARAMS, *GEAR_SERVO_DYNAMIC_STATE)


@dataclass(kw_only=True)
class LeapGearServoActuatorCfg(ActuatorCfg):
    """See the module docstring. Units: A, rad, s, N m."""

    current_gain: float  # G, A/rad
    current_damping: float  # Gd, A s/rad
    current_limit: float  # I_max, A
    torque_constant: float  # K_t, N m/A (motor side, before the gear's friction)
    gear_friction: float  # mu
    dry_friction: float  # f0, N m (> 0: MuJoCo builds friction rows only where it is)
    side_joints: tuple[str, ...] = ()  # regexes of the joints at side_gain_scale x G and Gd
    side_gain_scale: float = 0.75
    host_period: float = 1.0 / 80.0  # s
    lead_clamp: float = 0.15  # rad
    friction_solref: tuple[float, float] = (0.01, 1.0)
    # MuJoCo's dry friction is a soft constraint: a joint held below breakaway creeps. These hold a
    # 0.4 N m backdriving load to 0.07 mrad/s (MuJoCo's default impedance: 35 mrad/s).
    friction_solimp: tuple[float, float, float] = (0.9999, 0.99999, 0.001)
    carry_link_weight: bool = True  # undo the scene's gravity compensation on these joints
    # Low-speed stiction on ``stiction_joints`` (regexes): + stiction_friction N m of dry friction while
    # |qdot| < stiction_speed rad/s. Off by default (no joints: the versions before gear_2026_10b).
    stiction_friction: float = 0.0
    stiction_speed: float = 0.05
    stiction_joints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.frictionloss = self.dry_friction
        super().__post_init__()
        assert self.transmission_type == TransmissionType.JOINT, "the LEAP gear servo drives joints"
        assert self.dry_friction > 0.0, "dry_friction must be > 0 (friction rows exist only where it is)"
        assert 0.0 <= self.gear_friction < 1.0, "gear_friction must be in [0, 1): >= 1 cannot be backdriven"
        assert self.stiction_friction >= 0.0 and self.stiction_speed > 0.0, (self.stiction_friction, self.stiction_speed)

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> LeapGearServoActuator:
        return LeapGearServoActuator(self, entity, target_ids, target_names)


class LeapGearServoActuator(Actuator[LeapGearServoActuatorCfg]):
    """The measured LEAP servo (module docstring). Custom compute: one call for all its joints."""

    state_attributes = GEAR_SERVO_STATE
    dynamic_state_attributes = GEAR_SERVO_DYNAMIC_STATE

    def __init__(self, cfg: LeapGearServoActuatorCfg, entity: Entity, target_ids: list[int],
                 target_names: list[str]) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self.current: torch.Tensor | None = None  # the servo current of the last step, A (the driver's effort)

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        cfg = self.cfg
        # The law clips itself; the motor's own range only has to admit any randomized K_t I_max.
        effort = 4.0 * cfg.torque_constant * cfg.current_limit
        for name in target_names:
            self._mjs_actuators.append(create_motor_actuator(
                spec, name, effort_limit=effort, armature=cfg.armature, frictionloss=cfg.dry_friction,
                viscous_damping=cfg.viscous_damping))
            joint = spec.joint(name)
            joint.solref_friction = cfg.friction_solref
            joint.solimp_friction = (*cfg.friction_solimp, 0.5, 2.0)

    def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        cfg, n, nj = self.cfg, data.nworld, len(self._target_names)
        side = torch.tensor([any(re.fullmatch(p, name) for p in cfg.side_joints) for name in self._target_names],
                            device=device)
        scale = torch.where(side, cfg.side_gain_scale, 1.0)
        full = lambda v: torch.full((n, nj), float(v), dtype=torch.float, device=device)
        self.current_gain = full(cfg.current_gain) * scale
        self.current_damping = full(cfg.current_damping) * scale
        self.current_limit = full(cfg.current_limit)
        self.torque_constant = full(cfg.torque_constant)
        self.gear_friction = full(cfg.gear_friction)
        self.dry_friction = full(cfg.dry_friction)
        # (nj,) the joints with low-speed stiction; stiction_friction is zero on every other joint
        self.stiction_joints = torch.tensor(
            [any(re.fullmatch(p, name) for p in cfg.stiction_joints) for name in self._target_names], device=device)
        self.stiction_friction = full(cfg.stiction_friction) * self.stiction_joints
        self._has_stiction = bool(self.stiction_joints.any())
        self.defaults = {k: getattr(self, k).clone() for k in GEAR_SERVO_PARAMS}
        self.current = torch.zeros((n, nj), device=device)
        self._goal = torch.zeros((n, nj), device=device)
        self._clock = torch.zeros(n, device=device)  # time since the last driver tick, s
        self._tick_now = torch.ones(n, dtype=torch.bool, device=device)
        self._dt = float(mj_model.opt.timestep)
        self._dof = self.entity.indexing.joint_v_adr[self.target_ids]  # global dof addresses
        self._nv = mj_model.nv

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self.current is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        # The driver samples the new pose on the first step; its later ticks fall at a random phase.
        self._tick_now[ids] = True
        self._clock[ids] = torch.rand_like(self._clock[ids]) * self.cfg.host_period
        self.current[ids] = 0.0

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        cfg = self.cfg
        self._clock += self._dt
        due = self._clock >= cfg.host_period
        tick = self._tick_now | due
        self._clock = torch.where(due, self._clock - cfg.host_period, self._clock)
        self._tick_now[:] = False
        goal = cmd.pos + torch.clamp(cmd.position_target - cmd.pos, -cfg.lead_clamp, cfg.lead_clamp)
        self._goal = torch.where(tick.unsqueeze(-1), goal, self._goal)
        current = self.current_gain * (self._goal - cmd.pos) - self.current_damping * cmd.vel
        self.current = torch.clamp(current, -self.current_limit, self.current_limit)
        self._write_gear_friction()
        torque = self.torque_constant * self.current + cmd.effort_target
        if cfg.carry_link_weight:  # last step's compensation (a function of pose: changes slowly)
            torque = torque - self.entity.data.data.qfrc_gravcomp[:, self._dof]
        return torque

    def _write_gear_friction(self) -> None:
        """frictionloss = f0 + mu |torque the gear carries from the link side| (last step's forces), plus the
        stiction joints' stiction_friction while |qdot| < stiction_speed (this step's velocity)."""
        sim_model, sim_data = self.entity.data.model, self.entity.data.data
        frictionloss = sim_model.dof_frictionloss
        if frictionloss.shape[0] > 1 and frictionloss.stride(0) == 0:  # unexpanded: one row seen by every env
            raise RuntimeError(
                "LeapGearServoActuator writes dof_frictionloss per env, but the field is not expanded: add "
                "the in_hand_rotation_mjlab.mdp.randomize_leap_gear_servo reset event (leap_plant.leap_gear_servo_events).")
        efc = sim_data.efc
        rows = torch.arange(efc.type.shape[1], device=efc.type.device)
        own = (efc.type == _FRICTION_DOF) & (rows < sim_data.nefc.unsqueeze(-1))
        friction = torch.zeros((efc.force.shape[0], self._nv), device=efc.force.device)
        friction.scatter_add_(1, torch.where(own, efc.id, 0), torch.where(own, efc.force, 0.0))
        dof = self._dof
        # qfrc_smooth = passive - bias + applied + actuator + J^T xfrc_applied, so this is gravity and every
        # applied force or body wrench; the constraint part is contacts and joint limits (not the gear's own rows).
        smooth = sim_data.qfrc_smooth[:, dof] - sim_data.qfrc_passive[:, dof] - sim_data.qfrc_actuator[:, dof]
        load = smooth + sim_data.qfrc_constraint[:, dof] - friction[:, dof]
        loss = self.dry_friction + self.gear_friction * load.abs()
        if self._has_stiction:
            loss = loss + self.stiction_friction * (sim_data.qvel[:, dof].abs() < self.cfg.stiction_speed)
        frictionloss[:, dof] = loss


def actuator_effort_limits(entity) -> torch.Tensor:
    """(num_envs, nu) the torque each of the entity's actuators can actually produce, in its ctrl order.

    IdealPd (fused groups): its force limit. The LEAP gear servo: K_t G x the driver's lead clamp,
    because the 80 Hz clamp caps the servo current at G x 0.15 rad (0.30 A; 0.225 A on the
    abduction joints), below its 0.55 A limit. Any other actuator: 1.
    """
    effort = torch.ones_like(entity.data.actuator_force)
    for group in entity._fused_actuator_group._groups:
        effort[:, group.ctrl_ids] = group.params["force_limit"]
    for act in entity.actuators:
        if isinstance(act, LeapGearServoActuator):
            effort[:, act.ctrl_ids] = act.torque_constant * act.current_gain * act.cfg.lead_clamp
    return effort
