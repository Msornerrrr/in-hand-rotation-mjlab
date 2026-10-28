"""PaXini fingertip contact model for the LEAP hand: the one home of the pad geometry.

Calibrated by replaying real LEAP episodes in simulation and comparing the simulated contacts with the PaXini pads'
readings. ``decompose_tip_meshes`` below splits the fingertips, and ``PAD_GEOM_RE`` / ``pad_geom_patterns`` name the
pad geoms for every pad lookup (the friction randomization, the fingertip-rotation sensors). What ``LeapPadModel``
applies to the training tasks:

* **decompose = 8**: each fingertip (``*_tip``) collision mesh is replaced by 8 convex slabs along its longest axis.
  MuJoCo collides a mesh as its convex hull and the tip mesh is 29% concave by volume, so the hull fills the pad's
  dish with phantom material and a pad that should cup an object touches it at a point. 8 slabs recover it
  (hull / mesh volume 1.29 -> 1.03). The original tip stays as the visual (and keeps its name), with collisions off.
* **link_inflate_m = -0.5 mm** on the distal links' collision boxes: the pad surface sits closer to the bone by the
  compression the real hand's data implies (fingers 0.48-0.82 mm from their command while gripping).
* **solimp = (0.30, 0.99, 0.003, 0.5, 2.0)** on the pad + distal-link geoms: a compliant layer, paired with the
  compression above.
* **torsion 0.05 / roll 0.005** on the pads (friction channels 1 and 2); they act only where the contact's condim is
  6, i.e. against round objects (cylinders 6, boxes 3).

Not adopted: the calibration's best sliding friction, 6.0. It stands in for the pad shape the rigid model lacks (the
real contact patch is 2.5x wider) rather than being a coefficient, and a policy trained on it under-grips. Sliding
friction stays a randomized range set by the task (pads 0.25-1.2, covering the ~0.3-0.4 effective pad-object friction
measured on the real hand).

Applied as an entity spec editor AFTER the hand's ``CollisionCfg`` (append ``PadModelEditor`` to the robot
``EntityCfg.collisions``), so every piece copies the finished pad settings (priority 2, condim 6, per-finger
contype / conaffinity) and the original tip's collisions stay off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

PAD_FINGERS = ("if", "mf", "rf", "th")  # index, middle, ring, thumb
#: a fingertip pad geom's name (compiled or leaf): the original tip or one of its convex pieces
PAD_GEOM_RE = re.compile(r"(if|mf|rf|th)_tip(_cvx\d+)?$")
#: a fingertip MESH geom's leaf name (what ``decompose_tip_meshes`` splits)
TIP_MESH_RE = r"^(if|mf|rf|th)_tip$"


def is_pad_geom(name: str) -> bool:
  """True for a fingertip pad geom (``robot/if_tip``, ``robot/leap_left/th_tip_cvx3``, ...)."""
  return PAD_GEOM_RE.search(name) is not None


def pad_geom_patterns() -> tuple[str, ...]:
  """Per-finger pad patterns (index, middle, ring, thumb), for sensor primaries: the tip or its pieces."""
  return tuple(rf"{f}_tip(_cvx\d+)?" for f in PAD_FINGERS)


def pad_finger(name: str) -> int:
  """The finger index (0 index .. 3 thumb) of a pad geom name."""
  m = PAD_GEOM_RE.search(name)
  if m is None:
    raise ValueError(f"not a pad geom: {name!r}")
  return PAD_FINGERS.index(m.group(1))


@dataclass(frozen=True)
class LeapPadModel:
  decompose: int = 8
  link_inflate_m: float = -0.0005
  solimp: tuple[float, float, float, float, float] | None = (0.30, 0.99, 0.0030, 0.5, 2.0)
  torsion: float | None = 0.05
  roll: float | None = 0.005
  # geoms that get solimp / torsion / roll / inflation (distal-link boxes + tip)
  pattern: str = r"(if|mf|rf|th)_(ds_collision_\d+|tip)$"
  piece_group: int = 3  # collision-only group: hidden in the default render, the visual tip shows


LEAP_PAD_TWIN_2026_10 = LeapPadModel()


@dataclass(frozen=True)
class PadModelEditor:
  """mjlab spec editor (duck-typed like ``CollisionCfg``: ``edit_spec(spec)``)."""
  model: LeapPadModel

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    apply_pad_model(spec, self.model)


def apply_pad_model(spec: mujoco.MjSpec, m: LeapPadModel) -> dict:
  pat = re.compile(m.pattern)
  hit = inflated = 0
  for g in spec.geoms:
    leaf = g.name.split("/")[-1]
    if not pat.search(leaf):
      continue
    hit += 1
    if m.solimp is not None:
      g.solimp[:5] = [float(v) for v in m.solimp]
    if m.torsion is not None:
      g.friction[1] = float(m.torsion)
    if m.roll is not None:
      g.friction[2] = float(m.roll)
    if m.link_inflate_m != 0.0 and g.type != mujoco.mjtGeom.mjGEOM_MESH:
      g.size[:3] = [float(v) + m.link_inflate_m for v in g.size[:3]]
      inflated += 1
  if hit == 0:
    raise ValueError(f"pad pattern {m.pattern!r} matched no geom")
  replaced = decompose_tip_meshes(spec, m.decompose, piece_group=m.piece_group, piece_rgba=(0.5, 0.5, 0.5, 0.0),
                                  massless=True) if m.decompose > 1 else 0
  if m.decompose > 1 and replaced != 4:
    raise ValueError(f"expected to decompose the 4 fingertip meshes, decomposed {replaced}")
  return dict(pad_geoms=hit, inflated=inflated, tips_decomposed=replaced)


def decompose_tip_meshes(spec: mujoco.MjSpec, pieces: int, *, tip_pattern: str | re.Pattern = TIP_MESH_RE,
                         spread_m: float = 0.0, inflate_m: float = 0.0, piece_group: int | None = None,
                         piece_rgba: tuple[float, float, float, float] | None = None, massless: bool = False) -> int:
  """Replace each fingertip mesh geom with ``pieces`` convex pieces, so the pad can CUP an object; returns how many
  tips were replaced.

  MuJoCo collides a mesh as its convex hull and the LEAP tip is 29% concave by volume, so the hull fills the pocket
  that cups an object. Each piece is the convex hull of the tip vertices in one band along the tip's longest local
  axis (hull / mesh volume 1.29 -> 1.03 at 8 pieces). Built in memory through
  ``uservert``: no asset files.

  The vertices are read from a THROWAWAY compile (the tips come from OBJ files, so ``uservert`` is empty), in the
  mesh's NORMALISED frame: MuJoCo folds the mesh's frame normalisation into the compiled geom, so each piece reuses
  the source's COMPILED pos / quat and the two foldings cancel, landing the piece where its band was (the declared
  pose collapsed the pieces onto one spot). The original tip stays as the visual
  with collisions off (``MjsGeom`` has no ``delete()``). Pieces copy the tip's contact settings (condim, priority,
  friction, solref / solimp, margin / gap, contype / conaffinity).

  ``tip_pattern``: matched (``search``) against the leaf names of the MESH geoms. ``spread_m`` / ``inflate_m``: piece-widening
  options (0 = off). ``piece_group``
  / ``piece_rgba``: None keeps the tip's group / MuJoCo's default colour. ``massless``: pieces get mass 0 (the link
  keeps its inertia). Names: geom ``<tip>_cvx<k>``, mesh ``<tip>_cvx<k>_mesh`` (``PAD_GEOM_RE``)."""
  import trimesh

  pattern = re.compile(tip_pattern) if isinstance(tip_pattern, str) else tip_pattern
  compiled = spec.compile()
  src = {}
  for gid in range(compiled.ngeom):
    if int(compiled.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
      continue
    leaf = (mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").split("/")[-1]
    if not pattern.search(leaf):
      continue
    mid = int(compiled.geom_dataid[gid])
    a, n = int(compiled.mesh_vertadr[mid]), int(compiled.mesh_vertnum[mid])
    src[leaf] = (np.asarray(compiled.mesh_vert[a:a + n], float), np.asarray(compiled.geom_pos[gid], float),
                 np.asarray(compiled.geom_quat[gid], float))
  replaced = 0
  for g in list(spec.geoms):
    leaf = g.name.split("/")[-1]
    if leaf not in src or g.type != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    verts, pos, quat = src[leaf]
    axis = int(np.argmax(verts.max(0) - verts.min(0)))
    edges = np.linspace(verts[:, axis].min(), verts[:, axis].max(), pieces + 1)
    added = 0
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
      band = verts[(verts[:, axis] >= lo) & (verts[:, axis] <= hi)]
      if len(band) < 4:
        continue
      hull = trimesh.Trimesh(vertices=band, process=False).convex_hull
      if spread_m != 0.0:
        # widen in the plane tangent to the tip's surface: each vertex about the piece centroid splits into the part
        # along the piece's outward direction (kept: the thickness is unchanged) and the tangent part (extended); the
        # outward direction runs from the whole tip's centroid to the piece's (the shell's normal within its band)
        vh = np.asarray(hull.vertices, float)
        centre = vh.mean(axis=0)
        outward = centre - verts.mean(axis=0)
        norm = float(np.linalg.norm(outward))
        if norm > 1e-9:
          outward = outward / norm
          rel = vh - centre
          along = (rel @ outward)[:, None] * outward
          tangent = rel - along
          extent = np.linalg.norm(tangent, axis=1, keepdims=True)
          grown = tangent * (1.0 + spread_m / np.maximum(extent, 1e-6))
          hull = trimesh.Trimesh(vertices=centre + along + grown, process=False).convex_hull
      if inflate_m != 0.0:
        # outward along the hull's own vertex normals, re-hulled so the piece stays convex (a radial scale about the
        # centroid would stretch it along the tip instead of thickening its surface)
        grown = np.asarray(hull.vertices, float) + np.asarray(hull.vertex_normals, float) * float(inflate_m)
        hull = trimesh.Trimesh(vertices=grown, process=False).convex_hull
      mesh = spec.add_mesh()
      mesh.name = f"{leaf}_cvx{k}_mesh"
      mesh.uservert = np.asarray(hull.vertices, dtype=np.float64).reshape(-1).tolist()
      new = g.parent.add_geom()
      new.name = f"{leaf}_cvx{k}"
      new.type = mujoco.mjtGeom.mjGEOM_MESH
      new.meshname = mesh.name
      new.pos, new.quat = pos, quat
      new.condim, new.priority = g.condim, g.priority
      new.friction, new.solref, new.solimp = g.friction, g.solref, g.solimp
      new.margin, new.gap = g.margin, g.gap
      new.contype, new.conaffinity = g.contype, g.conaffinity
      new.group = g.group if piece_group is None else piece_group
      if piece_rgba is not None:
        new.rgba = list(piece_rgba)
      if massless:
        new.mass = 0.0
      added += 1
    if added:
      g.contype = 0  # keep it visual; MjsGeom cannot be deleted
      g.conaffinity = 0
      replaced += 1
  return replaced
