"""Helpers for collision-free MJLab NaN diagnostics in distributed runs."""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg


def enable_nan_guard_for_rank(
    env_cfg: ManagerBasedRlEnvCfg, *, rank: int, world_size: int
) -> Path:
    """Enable the NaN guard and isolate its output for each worker.

    Each worker owns an independent guard and can detect a non-finite state on
    the same simulation step. MJLab writes fixed ``*_latest`` symlink names, so
    sharing one output directory makes simultaneous dumps race.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    nan_guard_cfg = env_cfg.sim.nan_guard
    nan_guard_cfg.enabled = True
    output_dir = Path(nan_guard_cfg.output_dir)
    if world_size > 1:
        output_dir = output_dir / f"rank_{rank}"
    nan_guard_cfg.output_dir = str(output_dir)
    return output_dir
