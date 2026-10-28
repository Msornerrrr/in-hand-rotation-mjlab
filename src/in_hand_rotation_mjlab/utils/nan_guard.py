"""Defensive NaN handling for RSL-RL training (originally from two_grasp_rl)."""

import torch

from mjlab.rl import RslRlVecEnvWrapper


class NanSafeVecEnvWrapper(RslRlVecEnvWrapper):
    """Force-resets and sanitizes any env whose observation/reward goes
    non-finite, before rsl_rl's strict check_nan can crash the whole run.

    mjlab's own nan_detection termination (checked against qpos/qvel/actions)
    already auto-resets most physics-level divergence within the same
    step() call -- but a stage1_reach_only run crashed once on a non-finite
    *observation* that nan_detection apparently didn't flag (its own
    Episode_Termination/nan stat read exactly 0.0 the step before the
    crash), most likely from a downstream observation computation (e.g.
    geometry_spatial_features' nearest-surface-point math) going non-finite
    on an extreme-but-technically-finite physics state nan_detection's own
    narrower check doesn't cover. Root cause not pinned down -- 6000 steps
    of replay from that run's last checkpoint, using the same stochastic
    action sampling real rollouts use, didn't reproduce it, so this is a
    genuine rare tail event, not a known deterministic bug. This wrapper is
    a defensive net rather than a targeted fix: it catches a non-finite
    value regardless of which observation term or physics quantity was the
    actual source, force-resets only the affected env(s) via the
    underlying env's own env_ids-scoped reset, and reports them as done so
    rsl_rl's bookkeeping treats it as a normal episode boundary.
    """

    # Values this large are always a diverged env, never a real observation
    # (sane obs/rewards are O(10)). Huge-but-FINITE spikes from a physics
    # blowup pass the isfinite check yet still explode the RSL-RL obs
    # normalizer / gradients (observed: policy std -> NaN, "normal expects
    # all elements of std >= 0"), so magnitude is guarded too.
    _MAG_LIMIT = 1.0e6

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = super().step(actions)
        nonfinite = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        oversized = torch.zeros_like(nonfinite)
        for value in obs.values():
            nonfinite |= ~torch.isfinite(value).all(dim=-1)
            oversized |= (value.abs() > self._MAG_LIMIT).any(dim=-1)
        nonfinite |= ~torch.isfinite(rew)
        oversized |= rew.abs() > self._MAG_LIMIT
        bad_envs = nonfinite | oversized
        if bool(bad_envs.any()):
            bad_ids = bad_envs.nonzero(as_tuple=False).squeeze(-1)
            kinds = []
            if bool(nonfinite.any()):
                kinds.append("non-finite")
            if bool(oversized.any()):
                kinds.append(f"|value|>{self._MAG_LIMIT:.0e}")
            print(
                f"[WARN] NanSafeVecEnvWrapper: {' + '.join(kinds)} obs/reward in "
                f"{len(bad_ids)} env(s) {bad_ids.tolist()}, forcing a hard reset for them"
            )
            fresh_obs, _ = self.unwrapped.reset(env_ids=bad_ids)
            for key, value in obs.items():
                # Sanitize the spliced-in reset obs too: a reset into a still-
                # pathological state must not re-poison the same step.
                value[bad_ids] = torch.nan_to_num(
                    fresh_obs[key][bad_ids],
                    nan=0.0,
                    posinf=self._MAG_LIMIT,
                    neginf=-self._MAG_LIMIT,
                ).clamp_(-self._MAG_LIMIT, self._MAG_LIMIT)
            rew[bad_ids] = 0.0
            dones[bad_ids] = 1
        return obs, rew, dones, extras
