"""Fingertip-only in-hand rotation: the LEAP left (custom, PaXini tips) hand at a fixed palm-up or
palm-down mount, with a primitive object held by the fingertip pads only.

The env is the hand_cube yaw task's (same hand, same contact model) moved onto the measured LEAP
plant (gear servo ``LEAP_SERVO_LATEST`` with its randomization, the real driver's 0.15 rad
command-lead clamp, the measured pad / link friction ranges) with three changes:

* the object (``objects.ObjectSpec``) replaces the 7.5 cm cube; it keeps the entity / geom names
  ``cube`` / ``cube_geom`` so the hand_cube terms and planning tools address it unchanged;
* the hand mount: ``palm="up"`` (the hand_cube pose, palm normal +z) or ``"down"`` (rotated 180 deg
  about world x, palm normal -z);
* two contact sensors split the object's contacts into fingertip pads (``robot/*_tip``, the
  PaXini pad meshes) and everything else (links, palm): fingertip-only means the second is empty;
  a third reports the palm alone.

``hand frame`` below = the robot root frame (palm-up mount at the origin): grasp caches store
object poses in it, so one cache serves both mounts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import torch

from in_hand_rotation_mjlab.robots.leap_hand.pad_model import pad_finger, pad_geom_patterns
from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import (
  LEAP_DRIVER_LEAD_CLAMP,
  LEAP_SERVO_LATEST,
  LEAP_SERVO_S2R_2026_10,
  use_leap_gear_servo,
)

from .objects import OBJECTS, ObjectSpec


@dataclass(frozen=True)
class Mount:
  name: str
  pos: tuple[float, float, float]  # robot root, world
  rot: tuple[float, float, float, float]  # robot root, world (wxyz)

  @property
  def R(self) -> np.ndarray:
    import mujoco

    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(self.rot, dtype=np.float64))
    return m.reshape(3, 3)

  @property
  def palm_normal(self) -> np.ndarray:
    """World direction of the palm normal (hand +z)."""
    return self.R[:, 2]

  def to_world(self, p_hand: np.ndarray) -> np.ndarray:
    return np.asarray(p_hand) @ self.R.T + np.asarray(self.pos)

  def to_hand(self, p_world: np.ndarray) -> np.ndarray:
    return (np.asarray(p_world) - np.asarray(self.pos)) @ self.R


MOUNTS = {
  "up": Mount("up", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0)),
  "down": Mount("down", (0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.0)),
}

# Hand-frame z of the palm's top surface (the palm collision boxes), for clearance checks.
PALM_TOP_Z_HAND = 0.109

PAD_SENSOR = "pad_contact"
NONPAD_SENSOR = "nonpad_contact"
PALM_SENSOR = "palm_contact"
# sensor primaries: each finger's tip, or its convex pieces under a decomposed pad model (robots/leap_hand/pad_model.py);
# pad_net_force / pad_contact_pos sum them per finger (index, middle, ring, thumb)
PAD_GEOMS = pad_geom_patterns()


def _pad_finger_index(sensor) -> torch.Tensor:
  idx = getattr(sensor, "_ft_finger_idx", None)
  if idx is None:
    idx = torch.tensor([pad_finger(n) for n in sensor.primary_names], dtype=torch.long)
    sensor._ft_finger_idx = idx
  return idx


def pad_net_force(sensor) -> torch.Tensor:
  """(n, 4, 3) net contact force of each finger's pad (all its pieces), world frame."""
  F = sensor.data.force
  if F.shape[1] == 4:
    return F
  idx = _pad_finger_index(sensor).to(F.device)
  out = torch.zeros(F.shape[0], 4, 3, device=F.device, dtype=F.dtype)
  return out.index_add_(1, idx, F)


def pad_contact_pos(sensor) -> torch.Tensor:
  """(n, 4, 3) each finger's pad contact position (force-weighted over its pieces; 0 where untouched), world."""
  F, P = sensor.data.force, sensor.data.pos
  if F.shape[1] == 4:
    return P
  idx = _pad_finger_index(sensor).to(F.device)
  w = F.norm(dim=-1, keepdim=True)
  num = torch.zeros(F.shape[0], 4, 3, device=F.device, dtype=F.dtype).index_add_(1, idx, w * P)
  den = torch.zeros(F.shape[0], 4, 1, device=F.device, dtype=F.dtype).index_add_(1, idx, w)
  return num / den.clamp(min=1e-9)


def add_contact_sensors(cfg) -> None:
  # the object side is its BODY: every geom of it (one for a primitive, three for the multi-shape object)
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=PAD_SENSOR,
      primary=ContactMatch(mode="geom", pattern=PAD_GEOMS, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
      fields=("found", "force", "pos"), reduce="netforce", num_slots=1,
    ),
    ContactSensorCfg(
      name=NONPAD_SENSOR,
      primary=ContactMatch(mode="geom", pattern=".*", entity="robot", exclude=(r".*_tip",)),
      secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
      fields=("found", "force", "pos"), reduce="netforce", num_slots=1,
    ),
    # the palm alone (a subset of the non-pad contacts): the object must be carried by the fingers
    ContactSensorCfg(
      name=PALM_SENSOR,
      primary=ContactMatch(mode="geom", pattern=r"palm_collision_.*", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
      fields=("found", "force"), reduce="netforce", num_slots=1,
    ),
  )
  cfg.sim.contact_sensor_maxmatch = max(getattr(cfg.sim, "contact_sensor_maxmatch", 0) or 0, 256)


def fingertip_rotate_env_cfg(
  obj: ObjectSpec | str = "sphere6",
  palm: str = "up",
  *,
  servo=LEAP_SERVO_LATEST,
  delta: float = 1.0 / 24.0,
  max_command_lead: float = LEAP_DRIVER_LEAD_CLAMP,
  play: bool = False,
  pad_model=None,
  contact_friction: tuple[tuple[float, float], tuple[float, float]] | None = None,
):
  """The hand_cube yaw env cfg with the object, mount, plant and contact sensors of this task.

  ``pad_model``: a ``LeapPadModel`` (``robots/leap_hand/pad_model.py``, e.g. ``LEAP_PAD_TWIN_2026_10``) applied after
  the hand's collision settings, plus the object contact rule of the pad calibration (round objects condim 6 with torsional /
  rolling friction, boxes condim 3). ``contact_friction``: (pad range, other range) of the sliding-friction DR.

  Learning terms (rewards / terminations / curriculum / observations) are still the hand_cube
  yaw task's; ``physics_env_cfg`` strips them for planning and grasp synthesis.
  """
  from in_hand_rotation_mjlab.robots import get_leap_left_custom_hand_cfg
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_left_custom.env_cfgs import GRASP_INIT_JOINT_POS
  from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import MEASURED_PLANT_2026_10, make_hand_cube_embodiment_env_cfg
  from mjlab.entity import EntityCfg

  obj = OBJECTS[obj] if isinstance(obj, str) else obj
  mount = MOUNTS[palm]
  cfg = make_hand_cube_embodiment_env_cfg(
    robot_cfg_fn=partial(get_leap_left_custom_hand_cfg, servo=LEAP_SERVO_S2R_2026_10),
    grasp_init_joint_pos=GRASP_INIT_JOINT_POS,
    cube_spawn_pos=tuple(mount.to_world(np.array([-0.09, 0.055, 0.16]))),
    grasp_cache_file=None,
    use_grasp_cache=False,
    negate_yaw_rate=True,
    play=play,
    cube_half_size=obj.extent,
    cube_mass=obj.mass,
    sim2real=MEASURED_PLANT_2026_10,
    max_command_lead=max_command_lead,
  )
  robot = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = replace(robot, init_state=replace(robot.init_state, pos=mount.pos, rot=mount.rot))
  cube = cfg.scene.entities["cube"]
  cfg.scene.entities["cube"] = replace(cube, spec_fn=obj.spec)
  # The measured servo (gear_2026_10b at the time of writing) with its own randomization; one
  # setpoint per 50 ms control step, as the real dispatcher publishes it. ``servo=None`` keeps the
  # IdealPd s2r_2026_10 servo (diagnostics).
  if servo is not None:
    use_leap_gear_servo(cfg, servo=servo)
  act = cfg.actions["joint_pos"]
  act.interpolate_decimation = False
  act.delta_min, act.delta_max = -delta, delta
  add_contact_sensors(cfg)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 80)
  if pad_model is not None:
    from in_hand_rotation_mjlab.robots.leap_hand.pad_model import PadModelEditor

    robot = cfg.scene.entities["robot"]
    cfg.scene.entities["robot"] = replace(robot, collisions=tuple(robot.collisions) + (PadModelEditor(pad_model),))
    cube = cfg.scene.entities["cube"]
    cfg.scene.entities["cube"] = replace(cube, collisions=tuple(cube.collisions) + (ObjectContactEditor(obj.shape, pad_model),))
    cfg.sim.nconmax = max(cfg.sim.nconmax, 128)
    cfg.sim.njmax = max(cfg.sim.njmax or 0, 900)
  if contact_friction is not None and "dr_split_contact_friction" in cfg.events:
    cfg.events["dr_split_contact_friction"].params.update(pad_range=tuple(contact_friction[0]),
                                                           other_range=tuple(contact_friction[1]))
  return cfg


@dataclass(frozen=True)
class ObjectContactEditor:
  """The pad calibration's object contact rule: round objects (sphere, cylinder) get condim 6 with torsional / rolling
  friction (the pad model's values), boxes condim 3 (rolling is inert there and condim 4 fit the recordings worse)."""
  shape: str
  pad_model: object

  def edit_spec(self, spec) -> None:
    import mujoco

    round_types = (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER)
    for g in spec.geoms:
      # by geom type (the multi-shape object carries a sphere, a box and a cylinder)
      if g.type in round_types:
        g.condim = 6
        if self.pad_model.torsion is not None:
          g.friction[1] = max(float(g.friction[1]), float(self.pad_model.torsion))
        if self.pad_model.roll is not None:
          g.friction[2] = max(float(g.friction[2]), float(self.pad_model.roll))
      else:
        g.condim = 3


def physics_env_cfg(cfg, *, num_envs: int, randomize: bool, seed: int = 0, safety_sensors: bool = False):
  """Strip the learning machinery (planner / grasp-synthesis env), as hand_cube.sampling.make_env does."""
  from mjlab.envs import mdp as envs_mdp
  from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  from in_hand_rotation_mjlab.tasks.hand_cube.plant_utils import strip_plant_dr

  if safety_sensors:
    from in_hand_rotation_mjlab.tasks.hand_cube.safety import add_safety_sensors

    add_safety_sensors(cfg)
  if not randomize:
    strip_plant_dr(cfg, extra=("dr_split_contact_friction",))
  for k in ("reset_from_grasp_cache",):
    cfg.events.pop(k, None)
  cfg.rewards, cfg.metrics, cfg.curriculum, cfg.commands, cfg.terminations = {}, {}, {}, {}, {}
  cfg.observations = {
    "actor": ObservationGroupCfg(
      {"joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel,
                                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))})},
      enable_corruption=False,
    )
  }
  cfg.scene.num_envs = num_envs
  cfg.scene.env_spacing = 0.0
  cfg.auto_reset = False
  cfg.episode_length_s = 1e9
  cfg.seed = seed
  return cfg


def make_physics_env(obj="sphere6", palm="up", *, num_envs: int, randomize: bool, seed: int = 0,
                     device: str = "cuda:0", delta: float = 1.0 / 24.0, safety_sensors: bool = False,
                     servo=LEAP_SERVO_LATEST, pad_model=None, contact_friction=None):
  from mjlab.envs import ManagerBasedRlEnv

  cfg = physics_env_cfg(fingertip_rotate_env_cfg(obj, palm, servo=servo, delta=delta, play=True, pad_model=pad_model,
                                                 contact_friction=contact_friction),
                        num_envs=num_envs, randomize=randomize, seed=seed, safety_sensors=safety_sensors)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  return env


def hand_frame_tensors(palm: str, device) -> tuple[torch.Tensor, torch.Tensor]:
  """(R (3, 3), p (3,)) of the robot root in the world, as tensors."""
  m = MOUNTS[palm]
  return (torch.tensor(m.R, dtype=torch.float32, device=device),
          torch.tensor(m.pos, dtype=torch.float32, device=device))


def native_model(obj="sphere6", palm="up"):
  """Native (CPU) MuJoCo model of the scene: same qpos layout as the batched env."""
  from mjlab.scene import Scene

  return Scene(fingertip_rotate_env_cfg(obj, palm, play=True).scene, device="cpu").compile()
