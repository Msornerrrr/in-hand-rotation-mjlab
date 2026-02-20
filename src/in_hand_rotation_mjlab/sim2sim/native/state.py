from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import torch

from mjlab.scene import Scene


@dataclass
class _EntityRuntime:
  name: str
  is_fixed_base: bool
  is_mocap: bool
  joint_names: tuple[str, ...]
  joint_model_ids: torch.Tensor
  joint_qpos_adr: torch.Tensor
  joint_dof_adr: torch.Tensor
  actuator_names: tuple[str, ...]
  actuator_ctrl_ids: torch.Tensor
  body_names: tuple[str, ...]
  body_model_ids: torch.Tensor
  geom_names: tuple[str, ...]
  geom_model_ids: torch.Tensor
  site_names: tuple[str, ...]
  site_model_ids: torch.Tensor
  root_qpos_adr: int | None
  root_dof_adr: int | None
  mocap_id: int | None
  default_root_state: torch.Tensor
  default_joint_pos: torch.Tensor
  default_joint_vel: torch.Tensor
  encoder_bias: torch.Tensor
  hard_joint_pos_limits: torch.Tensor
  soft_joint_pos_limits: torch.Tensor
  actuated_joint_names_in_order: list[str]
  joint_name_to_ctrl_id: dict[str, int]
  root_body_id: int


@dataclass(frozen=True)
class _FieldSpec:
  entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator"]
  use_address: bool = False
  default_axes: tuple[int, ...] | None = None
  valid_axes: tuple[int, ...] | None = None


_FIELD_SPECS: dict[str, _FieldSpec] = {
  "dof_armature": _FieldSpec("dof", use_address=True),
  "dof_frictionloss": _FieldSpec("dof", use_address=True),
  "dof_damping": _FieldSpec("dof", use_address=True),
  "jnt_range": _FieldSpec("joint"),
  "jnt_stiffness": _FieldSpec("joint"),
  "body_ipos": _FieldSpec("body", default_axes=(0, 1, 2)),
  "body_iquat": _FieldSpec("body", default_axes=(0, 1, 2, 3)),
  "geom_friction": _FieldSpec("geom", default_axes=(0,), valid_axes=(0, 1, 2)),
  "geom_pos": _FieldSpec("geom", default_axes=(0, 1, 2)),
  "geom_quat": _FieldSpec("geom", default_axes=(0, 1, 2, 3)),
  "geom_rgba": _FieldSpec("geom", default_axes=(0, 1, 2, 3)),
  "site_pos": _FieldSpec("site", default_axes=(0, 1, 2)),
  "site_quat": _FieldSpec("site", default_axes=(0, 1, 2, 3)),
  "qpos0": _FieldSpec("joint", use_address=True),
}


class _NativeState:
  """Read-only state view backed by native MuJoCo model/data."""

  def __init__(
    self,
    scene: Scene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    device: str,
  ):
    self.scene = scene
    self.model = model
    self.data = data
    self.device = device
    self._entities: dict[str, _EntityRuntime] = {}

    init_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    self._init_key_id = init_key_id if init_key_id >= 0 else None

    for entity_name, entity in scene.entities.items():
      self._entities[entity_name] = self._build_entity_runtime(entity_name, entity)

  def _build_entity_runtime(self, entity_name: str, entity) -> _EntityRuntime:
    joint_model_ids = []
    joint_qpos_adr = []
    joint_dof_adr = []
    for joint_name in entity.joint_names:
      model_joint_name = f"{entity_name}/{joint_name}"
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, model_joint_name)
      if jid < 0:
        raise ValueError(f"Joint not found in model: {model_joint_name}")
      joint_model_ids.append(jid)
      joint_qpos_adr.append(self.model.jnt_qposadr[jid])
      joint_dof_adr.append(self.model.jnt_dofadr[jid])

    body_model_ids = []
    for body_name in entity.body_names:
      model_body_name = f"{entity_name}/{body_name}"
      bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, model_body_name)
      if bid < 0:
        raise ValueError(f"Body not found in model: {model_body_name}")
      body_model_ids.append(bid)

    geom_model_ids = []
    for geom_name in entity.geom_names:
      model_geom_name = f"{entity_name}/{geom_name}"
      gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, model_geom_name)
      if gid < 0:
        raise ValueError(f"Geom not found in model: {model_geom_name}")
      geom_model_ids.append(gid)

    site_model_ids = []
    for site_name in entity.site_names:
      model_site_name = f"{entity_name}/{site_name}"
      sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, model_site_name)
      if sid < 0:
        raise ValueError(f"Site not found in model: {model_site_name}")
      site_model_ids.append(sid)

    root_body_local_name = entity.root_body.name.split("/")[-1]
    model_root_body_name = f"{entity_name}/{root_body_local_name}"
    root_body_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, model_root_body_name
    )
    if root_body_id < 0:
      raise ValueError(f"Body not found in model: {model_root_body_name}")

    root_qpos_adr: int | None = None
    root_dof_adr: int | None = None
    body_jnt_adr = int(self.model.body_jntadr[root_body_id])
    body_jnt_num = int(self.model.body_jntnum[root_body_id])
    for j_offset in range(body_jnt_num):
      jid = body_jnt_adr + j_offset
      if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        root_qpos_adr = int(self.model.jnt_qposadr[jid])
        root_dof_adr = int(self.model.jnt_dofadr[jid])
        break

    mocap_id_raw = int(self.model.body_mocapid[root_body_id])
    mocap_id = mocap_id_raw if mocap_id_raw >= 0 else None

    actuator_names = tuple(a.name.split("/")[-1] for a in entity.spec.actuators)
    actuator_ctrl_ids = [int(a.id) for a in entity.spec.actuators]

    joint_name_to_ctrl_id: dict[str, int] = {}
    for actuator in entity.spec.actuators:
      joint_name = actuator.target.split("/")[-1]
      if joint_name in joint_name_to_ctrl_id:
        raise ValueError(
          f"Entity '{entity_name}' has multiple actuators on joint '{joint_name}'. "
          "This is not supported in native sim2sim yet."
        )
      joint_name_to_ctrl_id[joint_name] = int(actuator.id)

    actuated_joint_names_in_order = [
      j for j in entity.joint_names if j in joint_name_to_ctrl_id
    ]

    num_joints = len(joint_model_ids)
    default_joint_pos = torch.zeros((1, num_joints), device=self.device)
    default_joint_vel = torch.zeros((1, num_joints), device=self.device)
    init_state = entity.cfg.init_state
    default_root_state = torch.tensor(
      [
        *init_state.pos,
        *init_state.rot,
        *init_state.lin_vel,
        *init_state.ang_vel,
      ],
      device=self.device,
      dtype=torch.float32,
    ).unsqueeze(0)
    if entity.is_fixed_base:
      default_root_state[:, 7:13] = 0.0
    hard_joint_pos_limits = torch.zeros((num_joints, 2), device=self.device)
    for i, jid in enumerate(joint_model_ids):
      qadr = self.model.jnt_qposadr[jid]
      dadr = self.model.jnt_dofadr[jid]
      if self._init_key_id is not None:
        default_joint_pos[0, i] = float(self.model.key_qpos[self._init_key_id, qadr])
        default_joint_vel[0, i] = float(self.model.key_qvel[self._init_key_id, dadr])
      else:
        default_joint_pos[0, i] = float(self.model.qpos0[qadr])
        default_joint_vel[0, i] = 0.0
      hard_joint_pos_limits[i, 0] = float(self.model.jnt_range[jid, 0])
      hard_joint_pos_limits[i, 1] = float(self.model.jnt_range[jid, 1])

    if entity.cfg.articulation is not None:
      soft_limit_factor = entity.cfg.articulation.soft_joint_pos_limit_factor
    else:
      soft_limit_factor = 1.0
    joint_mid = 0.5 * (hard_joint_pos_limits[:, 0] + hard_joint_pos_limits[:, 1])
    joint_rng = hard_joint_pos_limits[:, 1] - hard_joint_pos_limits[:, 0]
    soft_joint_pos_limits = torch.stack(
      [
        joint_mid - 0.5 * joint_rng * soft_limit_factor,
        joint_mid + 0.5 * joint_rng * soft_limit_factor,
      ],
      dim=-1,
    )

    return _EntityRuntime(
      name=entity_name,
      is_fixed_base=bool(entity.is_fixed_base),
      is_mocap=bool(entity.is_mocap),
      joint_names=entity.joint_names,
      joint_model_ids=torch.tensor(
        joint_model_ids, device=self.device, dtype=torch.long
      ),
      joint_qpos_adr=torch.tensor(joint_qpos_adr, device=self.device, dtype=torch.long),
      joint_dof_adr=torch.tensor(joint_dof_adr, device=self.device, dtype=torch.long),
      actuator_names=actuator_names,
      actuator_ctrl_ids=torch.tensor(
        actuator_ctrl_ids, device=self.device, dtype=torch.long
      ),
      body_names=entity.body_names,
      body_model_ids=torch.tensor(body_model_ids, device=self.device, dtype=torch.long),
      geom_names=entity.geom_names,
      geom_model_ids=torch.tensor(geom_model_ids, device=self.device, dtype=torch.long),
      site_names=entity.site_names,
      site_model_ids=torch.tensor(site_model_ids, device=self.device, dtype=torch.long),
      root_qpos_adr=root_qpos_adr,
      root_dof_adr=root_dof_adr,
      mocap_id=mocap_id,
      default_root_state=default_root_state,
      default_joint_pos=default_joint_pos,
      default_joint_vel=default_joint_vel,
      encoder_bias=torch.zeros((1, num_joints), device=self.device),
      hard_joint_pos_limits=hard_joint_pos_limits,
      soft_joint_pos_limits=soft_joint_pos_limits,
      actuated_joint_names_in_order=actuated_joint_names_in_order,
      joint_name_to_ctrl_id=joint_name_to_ctrl_id,
      root_body_id=root_body_id,
    )

  def entity(self, entity_name: str) -> _EntityRuntime:
    if entity_name not in self._entities:
      raise KeyError(f"Entity '{entity_name}' not found.")
    return self._entities[entity_name]

  def _select_ids(
    self,
    ids: list[int] | slice,
    count: int,
  ) -> list[int]:
    if isinstance(ids, slice):
      return list(range(*ids.indices(count)))
    return list(ids)

  def joint_pos(self, entity_name: str) -> torch.Tensor:
    ent = self.entity(entity_name)
    qpos = torch.as_tensor(self.data.qpos, device=self.device, dtype=torch.float32)
    return qpos[ent.joint_qpos_adr].unsqueeze(0)

  def joint_vel(self, entity_name: str) -> torch.Tensor:
    ent = self.entity(entity_name)
    qvel = torch.as_tensor(self.data.qvel, device=self.device, dtype=torch.float32)
    return qvel[ent.joint_dof_adr].unsqueeze(0)

  def root_link_pose_w(self, entity_name: str) -> torch.Tensor:
    ent = self.entity(entity_name)
    xpos = torch.as_tensor(self.data.xpos, device=self.device, dtype=torch.float32)
    xquat = torch.as_tensor(self.data.xquat, device=self.device, dtype=torch.float32)
    pos = xpos[ent.root_body_id].unsqueeze(0)
    quat = xquat[ent.root_body_id].unsqueeze(0)
    return torch.cat([pos, quat], dim=-1)

  def root_link_vel_w(self, entity_name: str) -> torch.Tensor:
    ent = self.entity(entity_name)
    xpos = torch.as_tensor(self.data.xpos, device=self.device, dtype=torch.float32)
    subtree_com = torch.as_tensor(
      self.data.subtree_com, device=self.device, dtype=torch.float32
    )
    cvel = torch.as_tensor(self.data.cvel, device=self.device, dtype=torch.float32)
    pos = xpos[ent.root_body_id].unsqueeze(0)
    com = subtree_com[ent.root_body_id].unsqueeze(0)
    body_cvel = cvel[ent.root_body_id].unsqueeze(0)
    ang = body_cvel[:, 0:3]
    lin_c = body_cvel[:, 3:6]
    lin = lin_c - torch.cross(ang, com - pos, dim=-1)
    return torch.cat([lin, ang], dim=-1)

  def body_link_pose_w(
    self,
    entity_name: str,
    body_ids: list[int] | slice,
  ) -> torch.Tensor:
    ent = self.entity(entity_name)
    local_ids = self._select_ids(body_ids, len(ent.body_names))
    world_ids = ent.body_model_ids[local_ids]
    xpos = torch.as_tensor(self.data.xpos, device=self.device, dtype=torch.float32)
    xquat = torch.as_tensor(self.data.xquat, device=self.device, dtype=torch.float32)
    pos = xpos[world_ids]
    quat = xquat[world_ids]
    return torch.cat([pos, quat], dim=-1).unsqueeze(0)

  def body_link_vel_w(
    self,
    entity_name: str,
    body_ids: list[int] | slice,
  ) -> torch.Tensor:
    ent = self.entity(entity_name)
    local_ids = self._select_ids(body_ids, len(ent.body_names))
    world_ids = ent.body_model_ids[local_ids]
    xpos = torch.as_tensor(self.data.xpos, device=self.device, dtype=torch.float32)
    subtree_com = torch.as_tensor(
      self.data.subtree_com, device=self.device, dtype=torch.float32
    )
    cvel = torch.as_tensor(self.data.cvel, device=self.device, dtype=torch.float32)

    pos = xpos[world_ids]
    com = subtree_com[ent.root_body_id].unsqueeze(0).expand_as(pos)
    body_cvel = cvel[world_ids]
    ang = body_cvel[:, 0:3]
    lin_c = body_cvel[:, 3:6]
    lin = lin_c - torch.cross(ang, com - pos, dim=-1)
    vel = torch.cat([lin, ang], dim=-1)
    return vel.unsqueeze(0)

  def geom_pos_w(
    self,
    entity_name: str,
    geom_ids: list[int] | slice,
  ) -> torch.Tensor:
    ent = self.entity(entity_name)
    local_ids = self._select_ids(geom_ids, len(ent.geom_names))
    world_ids = ent.geom_model_ids[local_ids]
    geom_xpos = torch.as_tensor(
      self.data.geom_xpos, device=self.device, dtype=torch.float32
    )
    return geom_xpos[world_ids].unsqueeze(0)

  def geom_scalar_field(
    self,
    entity_name: str,
    geom_local_id: int,
    field: str,
    axis: int | None = None,
  ) -> torch.Tensor:
    ent = self.entity(entity_name)
    gid = int(ent.geom_model_ids[geom_local_id].item())
    arr = torch.as_tensor(getattr(self.model, field), device=self.device)
    value = arr[gid] if axis is None else arr[gid, axis]
    return torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(1, 1)

  def body_field(
    self,
    entity_name: str,
    body_local_id: int,
    field: str,
  ) -> torch.Tensor:
    ent = self.entity(entity_name)
    bid = int(ent.body_model_ids[body_local_id].item())
    arr = torch.as_tensor(getattr(self.model, field), device=self.device)
    value = arr[bid]
    if value.ndim == 0:
      return value.reshape(1, 1).to(dtype=torch.float32)
    return value.reshape(1, -1).to(dtype=torch.float32)
