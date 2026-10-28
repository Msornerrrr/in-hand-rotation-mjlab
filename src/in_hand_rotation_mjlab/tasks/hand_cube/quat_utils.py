"""Quaternion helpers (w, x, y, z) for the reorientation command and observations."""

from __future__ import annotations

import torch


def qmul(a, b):
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    -1,
  )


def axis_angle_quat(axis, angle):
  """Quaternions (wxyz) rotating by angle (n,) about unit axis (n, 3)."""
  h = angle[:, None] / 2
  return torch.cat((torch.cos(h), axis * torch.sin(h)), -1)


def rot_distance(q, g):
  return 2 * torch.acos((q * g).sum(-1).abs().clamp(max=1.0))


def random_quat(n, device, generator=None):
  u = torch.rand(n, 3, device=device, generator=generator)
  a, b, c = u.unbind(-1)
  return torch.stack(
    (
      torch.sqrt(1 - a) * torch.sin(2 * torch.pi * b),
      torch.sqrt(1 - a) * torch.cos(2 * torch.pi * b),
      torch.sqrt(a) * torch.sin(2 * torch.pi * c),
      torch.sqrt(a) * torch.cos(2 * torch.pi * c),
    ),
    -1,
  )[:, [3, 0, 1, 2]]


def quat_to_mat(q):
  w, x, y, z = q.unbind(-1)
  return torch.stack((
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ), -1).reshape(*q.shape[:-1], 3, 3)


def rot6d(R):
  return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)
