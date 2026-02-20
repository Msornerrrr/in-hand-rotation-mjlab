from __future__ import annotations

import math

import torch


def finite_or_default(value: float, default: float) -> float:
  """Return ``value`` when finite, otherwise ``default``."""
  return value if math.isfinite(value) else default


def safe_mean_finite(values: torch.Tensor) -> torch.Tensor | None:
  """Return mean over finite elements only, or ``None`` if no finite values exist."""
  finite_values = values[torch.isfinite(values)]
  if finite_values.numel() == 0:
    return None
  return torch.mean(finite_values)


def sanitize_to_range(
  value: torch.Tensor,
  bound_a: float,
  bound_b: float,
  *,
  nan_default: float | None = None,
) -> torch.Tensor:
  """Map non-finite values to finite defaults and clamp within bounds.

  Bounds may be provided in any order.
  """
  lower = min(bound_a, bound_b)
  upper = max(bound_a, bound_b)
  if nan_default is None:
    nan_default = lower
  nan_default = min(max(nan_default, lower), upper)
  return torch.clamp(
    torch.nan_to_num(value, nan=nan_default, posinf=upper, neginf=lower),
    min=lower,
    max=upper,
  )
