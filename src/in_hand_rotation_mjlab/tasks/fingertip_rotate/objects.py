"""Primitive objects for fingertip-only in-hand rotation: sphere, cube, upright cylinder, cuboid.

The object is the ``cube`` entity with one geom ``cube_geom`` whatever its shape, so that the
hand_cube observation, event and planning terms (which address ``cube`` / ``cube_geom``) work
unchanged. ``surface_distance`` gives the centre-to-surface distance along a direction in the
object frame, which the grasp synthesis uses to place fingertip targets on the surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

SHAPES = ("sphere", "cube", "cylinder", "cuboid")
_NSIZE = {"sphere": 1, "cube": 1, "cylinder": 2, "cuboid": 3}


@dataclass(frozen=True)
class ObjectSpec:
  name: str
  shape: str  # sphere | cube | cylinder (axis along the object z, upright at the identity pose)
  # sphere: (radius,); cube: (half side,); cylinder: (radius, half height); cuboid: (half x, half y, half z), z along
  # the palm normal at the identity pose
  size: tuple[float, ...]
  mass: float = 0.05  # kg, the nominal (DR scales it 0.7-1.4)
  rgba: tuple[float, float, float, float] = (0.85, 0.3, 0.15, 1.0)

  def __post_init__(self):
    assert self.shape in SHAPES, self.shape
    assert len(self.size) == _NSIZE[self.shape], (self.shape, self.size)

  @property
  def geom_type(self) -> int:
    return {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "cube": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "cuboid": mujoco.mjtGeom.mjGEOM_BOX}[self.shape]

  @property
  def geom_size(self) -> tuple[float, float, float]:
    if self.shape == "sphere":
      return (self.size[0],) * 3
    if self.shape == "cube":
      return (self.size[0],) * 3
    if self.shape == "cuboid":
      return tuple(self.size)
    return (self.size[0], self.size[1], 0.0)

  @property
  def extent(self) -> float:
    """Half extent along the object z (the palm normal at the identity pose) [m]."""
    return {"cylinder": self.size[1] if len(self.size) > 1 else 0.0, "cuboid": self.size[-1]}.get(self.shape, self.size[0])

  def spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(name="cube_geom", type=self.geom_type, size=self.geom_size, mass=self.mass, rgba=self.rgba)
    return spec


def surface_distance(obj: ObjectSpec, u: np.ndarray) -> np.ndarray:
  """Distance from the object centre to its surface along unit directions ``u`` (..., 3), object frame."""
  u = np.asarray(u, dtype=np.float64)
  a = np.abs(u)
  if obj.shape == "sphere":
    return np.full(u.shape[:-1], obj.size[0])
  if obj.shape == "cube":
    return obj.size[0] / np.maximum(a.max(-1), 1e-9)
  if obj.shape == "cuboid":
    h = np.asarray(obj.size)
    return (h / np.maximum(a, 1e-9)).min(-1)
  r, hh = obj.size
  radial = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)
  return np.minimum(r / np.maximum(radial, 1e-9), hh / np.maximum(a[..., 2], 1e-9))


# The starter set: simple, yaw-symmetric (sphere, cylinder) or 4-fold (cube) shapes of a size the
# LEAP fingertips span comfortably (5-6 cm), light like a printed shell.
OBJECTS = {
  "sphere6": ObjectSpec("sphere6", "sphere", (0.030,)),
  "sphere5": ObjectSpec("sphere5", "sphere", (0.025,)),
  "sphere7": ObjectSpec("sphere7", "sphere", (0.035,)),
  "cube5": ObjectSpec("cube5", "cube", (0.025,)),
  "cyl6": ObjectSpec("cyl6", "cylinder", (0.030, 0.030)),
}
# Size sweep (4-finger grasps): diameter / side in mm, e.g. "sphere45mm" (Ø4.5 cm), "cube40mm" (4 cm side).
for _d in (35, 40, 45, 50, 55, 60, 70, 80):
  OBJECTS[f"sphere{_d}mm"] = ObjectSpec(f"sphere{_d}mm", "sphere", (_d / 2000,))
for _d in (35, 40, 45, 50, 55, 60):
  OBJECTS[f"cube{_d}mm"] = ObjectSpec(f"cube{_d}mm", "cube", (_d / 2000,))
# Larger-object hold exploration (2026-10-20): spheres / cubes up to 9 / 8 cm, upright cylinders "cyl{D}x{H}mm"
# (diameter x height), cuboids "box{X}x{Y}x{Z}mm" (X toward the thumb, Y along the finger row, Z along the palm normal).
for _d in (90,):
  OBJECTS[f"sphere{_d}mm"] = ObjectSpec(f"sphere{_d}mm", "sphere", (_d / 2000,))
for _d in (70, 80):
  OBJECTS[f"cube{_d}mm"] = ObjectSpec(f"cube{_d}mm", "cube", (_d / 2000,))
for _d in (50, 60, 70, 80):
  for _h in (60, 90):
    OBJECTS[f"cyl{_d}x{_h}mm"] = ObjectSpec(f"cyl{_d}x{_h}mm", "cylinder", (_d / 2000, _h / 2000))
for _x, _y, _z in ((50, 70, 60), (70, 50, 60), (60, 80, 60), (80, 60, 60), (60, 60, 90)):
  OBJECTS[f"box{_x}x{_y}x{_z}mm"] = ObjectSpec(f"box{_x}x{_y}x{_z}mm", "cuboid", (_x / 2000, _y / 2000, _z / 2000))
# The training set (2026-10-20): every dimension <= 7.5 cm
for _d in (65, 75):
  OBJECTS[f"sphere{_d}mm"] = ObjectSpec(f"sphere{_d}mm", "sphere", (_d / 2000,))
  OBJECTS[f"cube{_d}mm"] = ObjectSpec(f"cube{_d}mm", "cube", (_d / 2000,))
for _d in (50, 60, 70, 75):
  for _h in (45, 60, 75):
    OBJECTS.setdefault(f"cyl{_d}x{_h}mm", ObjectSpec(f"cyl{_d}x{_h}mm", "cylinder", (_d / 2000, _h / 2000)))
for _x, _y, _z in ((50, 70, 60), (50, 60, 40), (60, 75, 45), (45, 75, 60), (55, 65, 75)):
  OBJECTS.setdefault(f"box{_x}x{_y}x{_z}mm", ObjectSpec(f"box{_x}x{_y}x{_z}mm", "cuboid", (_x / 2000, _y / 2000, _z / 2000)))

# Cuboids for yaw rotation about their LONG axis (the principal axis is the rotation axis, as
# a cylinder turns about its own axis): the longest side along the palm normal (z), every side <= 7.5 cm.
for _x, _y, _z in ((50, 60, 75), (55, 65, 75), (60, 70, 75), (50, 70, 75), (55, 60, 70)):
  OBJECTS.setdefault(f"box{_x}x{_y}x{_z}mm", ObjectSpec(f"box{_x}x{_y}x{_z}mm", "cuboid", (_x / 2000, _y / 2000, _z / 2000)))


# Geom names of the multi-shape object, by shape: cubes and cuboids share the box geom.
MIXED_GEOMS = {"sphere": "cube_geom", "cube": "cube_geom_box", "cuboid": "cube_geom_box", "cylinder": "cube_geom_cyl"}
MIXED_GEOM_ORDER = ("cube_geom", "cube_geom_box", "cube_geom_cyl")  # sphere, box, cylinder
SHAPE_ID = {s: k for k, s in enumerate(SHAPES)}  # sphere 0, cube 1, cylinder 2, cuboid 3
INACTIVE_SIZE = 0.001  # m: an inactive geom shrinks to this, deep inside the active one


@dataclass(frozen=True)
class MixedObject:
  """One free body carrying a sphere, a box and a cylinder geom, so every env can hold a different primitive
  (MuJoCo cannot change a geom's TYPE per env, only its size). Each geom is compiled at the largest size it takes
  in ``objs`` (the collision bounds then cover every size); at reset the active one is set to its object's size and
  the other two shrink to ``INACTIVE_SIZE`` at the centre, where they never touch the hand (geoms of one body do not
  collide with each other). Only the sphere geom carries mass at compile time; the reset sets the body inertia of the
  active shape. ``extent`` / ``mass`` are those the base cfg needs for its defaults."""
  objs: tuple[ObjectSpec, ...]
  mass: float = 0.05
  rgba: tuple[float, float, float, float] = (0.85, 0.3, 0.15, 1.0)
  shape: str = "mixed"
  name: str = "mixed"

  def max_geom_size(self, geom: str) -> tuple[float, float, float]:
    sizes = [np.asarray(o.geom_size) for o in self.objs if MIXED_GEOMS[o.shape] == geom]
    if not sizes:
      return (INACTIVE_SIZE,) * 3
    s = np.max(sizes, axis=0)
    return tuple(float(max(x, INACTIVE_SIZE)) for x in s)

  @property
  def extent(self) -> float:
    return max(o.extent for o in self.objs)

  def spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    types = {"cube_geom": mujoco.mjtGeom.mjGEOM_SPHERE, "cube_geom_box": mujoco.mjtGeom.mjGEOM_BOX,
             "cube_geom_cyl": mujoco.mjtGeom.mjGEOM_CYLINDER}
    for g in MIXED_GEOM_ORDER:
      size = self.max_geom_size(g)
      if g == "cube_geom_cyl":
        size = (size[0], size[1], 0.0)
      body.add_geom(name=g, type=types[g], size=size, mass=self.mass if g == "cube_geom" else 0.0, rgba=self.rgba)
    return spec


def mixed_geom_sizes(obj: ObjectSpec) -> np.ndarray:
  """(3, 3) per-geom sizes (MIXED_GEOM_ORDER) that make the multi-shape object this primitive."""
  out = np.full((3, 3), INACTIVE_SIZE, dtype=np.float32)
  k = MIXED_GEOM_ORDER.index(MIXED_GEOMS[obj.shape])
  out[k] = np.asarray(obj.geom_size, dtype=np.float32)
  if obj.shape == "cylinder":
    out[k, 2] = INACTIVE_SIZE
  return out


def principal_inertia(shape: str, size, mass):
  """(..., 3) principal moments of inertia (object frame) of a solid primitive. size: (..., 3) geom size
  (sphere r; box half extents; cylinder radius, half height)."""
  import torch

  s = size
  if shape == "sphere":
    i = 0.4 * mass * s[..., 0] ** 2
    return torch.stack([i, i, i], -1)
  if shape in ("cube", "cuboid"):
    a, b, c = 2 * s[..., 0], 2 * s[..., 1], 2 * s[..., 2]
    return torch.stack([mass * (b * b + c * c), mass * (a * a + c * c), mass * (a * a + b * b)], -1) / 12.0
  r, h = s[..., 0], 2 * s[..., 1]
  ixy = mass * (3 * r * r + h * h) / 12.0
  return torch.stack([ixy, ixy, 0.5 * mass * r * r], -1)


# The palm-down mixed-shape training set (2026-10-20, the canonical start grasps of export_mixed_resets.py --canon):
# spheres and cubes 6-7.5 cm, upright cylinders, cuboids with the longest side along the rotation axis.
MIXED_SET_2026_10 = (
  "sphere60mm", "sphere65mm", "sphere70mm", "sphere75mm", "cube60mm", "cube65mm", "cube70mm", "cube75mm",
  *(f"cyl{d}x{h}mm" for d in (60, 70, 75) for h in (45, 60, 75)),
  "box50x60x75mm", "box50x70x75mm", "box55x60x70mm", "box55x65x75mm", "box60x70x75mm",
)
