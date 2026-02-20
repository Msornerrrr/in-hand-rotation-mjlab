from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyServerConfig:
  checkpoint_file: str
  device: str | None = None
  play: bool = True

