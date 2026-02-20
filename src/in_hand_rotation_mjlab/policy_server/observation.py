from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import torch

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.buffers import CircularBuffer, DelayBuffer


@dataclass
class _ObsTermPipelineState:
  name: str
  cfg: ObservationTermCfg
  offset: torch.Tensor | None
  scale: torch.Tensor | None
  delay_buffer: DelayBuffer | None
  history_buffer: CircularBuffer | None


class ActorObservationProcessor:
  """Server-side actor observation transforms (clip/scale/delay/history/cat)."""

  def __init__(
    self,
    obs_group_cfg: ObservationGroupCfg,
    device: str,
    term_offsets: dict[str, torch.Tensor] | None = None,
  ):
    self.cfg = copy.deepcopy(obs_group_cfg)
    self.device = device
    self._terms: list[_ObsTermPipelineState] = []

    if not self.cfg.concatenate_terms:
      raise NotImplementedError(
        "Sim2sim server currently supports concatenate_terms=True only."
      )
    if self.cfg.concatenate_dim not in (-1, 1):
      raise NotImplementedError(
        "Sim2sim server currently supports concatenate_dim=-1 only."
      )

    for term_name, term_cfg in self.cfg.terms.items():
      term_cfg = copy.deepcopy(term_cfg)

      if self.cfg.history_length is not None:
        term_cfg.history_length = self.cfg.history_length
        term_cfg.flatten_history_dim = self.cfg.flatten_history_dim

      if self.cfg.enable_corruption and term_cfg.noise is not None:
        raise ValueError(
          "Observation noise corruption is enabled. "
          "Use play=True for deployment/sim2sim configs."
        )

      offset = None
      if term_offsets and term_name in term_offsets:
        offset = term_offsets[term_name].to(device=self.device, dtype=torch.float32)

      scale = None
      if term_cfg.scale is not None:
        scale = torch.tensor(term_cfg.scale, device=self.device, dtype=torch.float32)

      delay_buffer = None
      if term_cfg.delay_max_lag > 0:
        delay_buffer = DelayBuffer(
          min_lag=term_cfg.delay_min_lag,
          max_lag=term_cfg.delay_max_lag,
          batch_size=1,
          device=self.device,
          per_env=term_cfg.delay_per_env,
          hold_prob=term_cfg.delay_hold_prob,
          update_period=term_cfg.delay_update_period,
          per_env_phase=term_cfg.delay_per_env_phase,
        )

      history_buffer = None
      if term_cfg.history_length > 0:
        history_buffer = CircularBuffer(
          max_len=term_cfg.history_length,
          batch_size=1,
          device=self.device,
        )

      self._terms.append(
        _ObsTermPipelineState(
          name=term_name,
          cfg=term_cfg,
          offset=offset,
          scale=scale,
          delay_buffer=delay_buffer,
          history_buffer=history_buffer,
        )
      )

  def reset(self) -> None:
    for term in self._terms:
      if term.delay_buffer is not None:
        term.delay_buffer.reset()
      if term.history_buffer is not None:
        term.history_buffer.reset()

  def compute(self, raw_terms: dict[str, torch.Tensor]) -> torch.Tensor:
    out_terms: list[torch.Tensor] = []
    for term in self._terms:
      if term.name not in raw_terms:
        raise KeyError(
          f"Observation term '{term.name}' missing in client packet. "
          f"Available keys: {sorted(raw_terms.keys())}"
        )
      obs = cast(torch.Tensor, raw_terms[term.name]).to(device=self.device).clone()
      if term.offset is not None:
        obs = obs - term.offset
      if term.cfg.clip is not None:
        obs = obs.clip(min=term.cfg.clip[0], max=term.cfg.clip[1])
      if term.scale is not None:
        obs = obs * term.scale
      if term.delay_buffer is not None:
        term.delay_buffer.append(obs)
        obs = term.delay_buffer.compute()
      if term.history_buffer is not None:
        term.history_buffer.append(obs)
        if term.cfg.flatten_history_dim:
          obs = term.history_buffer.buffer.reshape(1, -1)
        else:
          obs = term.history_buffer.buffer
      out_terms.append(obs)
    return torch.cat(out_terms, dim=-1)

