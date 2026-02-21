#!/usr/bin/env python3
"""
sysid_collect.py

Self-contained sysid data collector for the LEAP hand.

For each finger (index, middle, ring, thumb) in order:
  1. Move all joints to neutral (rest pose) and hold for --settle-s
  2. Run step-ladder, chirp, and PRBS excitation on that finger's joints
     (all other joints stay at neutral)
  3. Auto-save one .npz file per finger to --out-dir

No external data_saving_node required.

Usage:
  python scripts/sysid_collect.py --out-dir data/2026-02-18-sysid/auto
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy

from config import CTRL_HZ, get_data_dir
from leap_hand_interface import LeapHandInterface


# ---------------------------------------------------------------------------
# Joint layout (hardware order)
#
# Hardware index → joint class:
#   0, 4, 8  : rot   range [-0.524,  0.524]
#   1, 5, 9  : mcp   range [-0.100,  1.920]
#   2, 6, 10 : pip   range [-0.050,  1.920]
#   3, 7, 11 : dip   range [-0.100,  1.920]
#   12       : th_cmc  hardware sign-flipped → range [-2.094, 0.000]
#   13       : th_axl  range [-2.100,  0.000]
#   14       : th_mcp  range [-0.470,  2.443]
#   15       : th_ipl  range [ 0.000,  1.880]
# ---------------------------------------------------------------------------
FINGER_JOINTS: dict[str, list[int]] = {
    "index":  [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "ring":   [8, 9, 10, 11],
    "thumb":  [12, 13, 14, 15],
}

# Hard joint limits [min, max] per joint index (hardware order)
_LIMITS = np.array([
    [-0.524,  0.524],   # 0  rot
    [-0.100,  1.920],   # 1  mcp
    [-0.050,  1.920],   # 2  pip
    [-0.100,  1.920],   # 3  dip
    [-0.524,  0.524],   # 4  rot
    [-0.100,  1.920],   # 5  mcp
    [-0.050,  1.920],   # 6  pip
    [-0.100,  1.920],   # 7  dip
    [-0.524,  0.524],   # 8  rot
    [-0.100,  1.920],   # 9  mcp
    [-0.050,  1.920],   # 10 pip
    [-0.100,  1.920],   # 11 dip
    [-2.094,  0.000],   # 12 th_cmc (hardware sign-flipped)
    [-2.100,  0.000],   # 13 th_axl
    [-0.470,  2.443],   # 14 th_mcp
    [ 0.000,  1.880],   # 15 th_ipl
], dtype=np.float32)

# Neutral = midpoint of each joint's range
NEUTRAL_Q: np.ndarray = ((_LIMITS[:, 0] + _LIMITS[:, 1]) / 2.0).astype(np.float32)

# Excitation amplitude = 80% of half-range
_EXCITATION_FACTOR = 0.80
AMPLITUDES: np.ndarray = (
    (_LIMITS[:, 1] - _LIMITS[:, 0]) / 2.0 * _EXCITATION_FACTOR
).astype(np.float32)

# Rest pose for non-active fingers: all zeros (flat/open, fingers apart)
REST_Q = np.zeros(16, dtype=np.float32)


# ---------------------------------------------------------------------------
# Simple recorder — collects (t, cmd, state) tuples in memory
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self._times: list[float] = []
        self._cmds:  list[np.ndarray] = []
        self._states: list[np.ndarray] = []

    def record(self, t: float, cmd: np.ndarray, state: np.ndarray) -> None:
        self._times.append(t)
        self._cmds.append(cmd.copy())
        self._states.append(state.copy())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            time=np.array(self._times, dtype=np.float64),
            hand_cmd=np.stack(self._cmds,  axis=0).astype(np.float32),
            hand_state=np.stack(self._states, axis=0).astype(np.float32),
        )
        rospy.loginfo("Saved %d samples -> %s", len(self._times), path)

    def clear(self) -> None:
        self._times.clear()
        self._cmds.clear()
        self._states.clear()

    def __len__(self) -> int:
        return len(self._times)


# ---------------------------------------------------------------------------
# Low-level step helper
# ---------------------------------------------------------------------------

def _step(
    hand: LeapHandInterface,
    recorder: Recorder,
    rate: rospy.Rate,
    q_cmd: np.ndarray,
) -> None:
    hand.send_command(q_cmd)
    state = hand.get_state()
    q_state = (
        np.asarray(state.position, dtype=np.float32)
        if state is not None
        else q_cmd.copy()
    )
    recorder.record(rospy.Time.now().to_sec(), q_cmd, q_state)
    rate.sleep()


def _make_q(joint_ids: list[int], offsets: np.ndarray) -> np.ndarray:
    """Build a full 16-joint command.

    Non-active joints → 0 (rest, fingers apart).
    Active joints     → NEUTRAL_Q[jid] + offset, clipped to hard limits.
    """
    q = REST_Q.copy()
    for k, jid in enumerate(joint_ids):
        q[jid] = float(np.clip(
            NEUTRAL_Q[jid] + offsets[k],
            _LIMITS[jid, 0], _LIMITS[jid, 1],
        ))
    return q


# ---------------------------------------------------------------------------
# Excitation patterns
# ---------------------------------------------------------------------------

def interpolate_to(hand, recorder, ctrl_hz, q_target: np.ndarray, duration_s: float = 2.0) -> None:
    """Smoothly interpolate from current command to q_target over duration_s."""
    rate = rospy.Rate(ctrl_hz)
    n = max(1, int(duration_s * ctrl_hz))
    state = hand.get_state()
    q_start = (
        np.asarray(state.position, dtype=np.float32)
        if state is not None else q_target.copy()
    )
    for i in range(n):
        if rospy.is_shutdown():
            return
        alpha = (i + 1) / n
        q = ((1 - alpha) * q_start + alpha * q_target).astype(np.float32)
        _step(hand, recorder, rate, q)


def settle(hand, recorder, ctrl_hz, settle_s, q_hold: np.ndarray | None = None) -> None:
    """Hold at q_hold for settle_s seconds. Defaults to REST_Q (all zeros)."""
    q = REST_Q if q_hold is None else q_hold
    rospy.loginfo("  [settle] %.1fs", settle_s)
    rate = rospy.Rate(ctrl_hz)
    for _ in range(int(settle_s * ctrl_hz)):
        if rospy.is_shutdown():
            return
        _step(hand, recorder, rate, q)


def step_ladder(hand, recorder, joint_ids, amplitudes, ctrl_hz, hold_s) -> None:
    """Per-joint steps at 3 amplitude levels, then coupled flex/extend."""
    rospy.loginfo("  [step_ladder] %d joints, hold=%.2fs", len(joint_ids), hold_s)
    rate = rospy.Rate(ctrl_hz)
    hold_steps = max(1, int(hold_s * ctrl_hz))
    n = len(joint_ids)

    # Independent per-joint
    for k in range(n):
        for scale in [0.5, 0.75, 1.0]:
            a = amplitudes[k] * scale
            for target in [a, 0.0, -a, 0.0]:
                offsets = np.zeros(n, dtype=np.float32)
                offsets[k] = target
                q = _make_q(joint_ids, offsets)
                for _ in range(hold_steps):
                    if rospy.is_shutdown():
                        return
                    _step(hand, recorder, rate, q)

    # Coupled: all joints flex / neutral / extend
    for sign in [1.0, 0.0, -1.0, 0.0]:
        q = _make_q(joint_ids, amplitudes * sign)
        for _ in range(hold_steps):
            if rospy.is_shutdown():
                return
            _step(hand, recorder, rate, q)


def chirp(hand, recorder, joint_ids, amplitudes, ctrl_hz, duration_s,
          f_start=0.1, f_end=3.0) -> None:
    """Linear chirp sweep per joint independently, then all coupled."""
    rospy.loginfo("  [chirp] %.1f->%.1f Hz, %.0fs/pass", f_start, f_end, duration_s)
    rate = rospy.Rate(ctrl_hz)
    dt = 1.0 / ctrl_hz
    n_steps = int(duration_s * ctrl_hz)
    n = len(joint_ids)

    for k in range(n):
        t = 0.0
        for _ in range(n_steps):
            if rospy.is_shutdown():
                return
            phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / duration_s)
            offsets = np.zeros(n, dtype=np.float32)
            offsets[k] = amplitudes[k] * np.sin(phase)
            _step(hand, recorder, rate, _make_q(joint_ids, offsets))
            t += dt

    # Coupled pass
    t = 0.0
    for _ in range(n_steps):
        if rospy.is_shutdown():
            return
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / duration_s)
        offsets = (amplitudes * float(np.sin(phase))).astype(np.float32)
        _step(hand, recorder, rate, _make_q(joint_ids, offsets))
        t += dt


def prbs(hand, recorder, joint_ids, amplitudes, ctrl_hz, duration_s,
         min_dwell=3, max_dwell=8) -> None:
    """Random ternary targets {-a, 0, +a} per joint with random dwell times."""
    rospy.loginfo("  [prbs] %.0fs", duration_s)
    rng = np.random.default_rng()
    rate = rospy.Rate(ctrl_hz)
    n_steps = int(duration_s * ctrl_hz)
    n = len(joint_ids)

    offsets = np.zeros(n, dtype=np.float32)
    dwell = 0
    for i in range(n_steps):
        if rospy.is_shutdown():
            return
        if dwell <= 0:
            levels = rng.choice([-1.0, 0.0, 1.0], size=n)
            offsets = (amplitudes * levels).astype(np.float32)
            dwell = int(rng.integers(min_dwell, max_dwell + 1))
        _step(hand, recorder, rate, _make_q(joint_ids, offsets))
        dwell -= 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated sysid data collection for LEAP hand."
    )
    p.add_argument(
        "--fingers", nargs="+",
        default=["index", "middle", "ring", "thumb"],
        choices=list(FINGER_JOINTS.keys()),
        help="Fingers to excite in order (default: all).",
    )
    p.add_argument("--out-dir",      type=str,   default=None,
                   help="Output directory (default: data/YYYY-MM-DD-sysid/auto/).")
    p.add_argument("--ctrl-hz",      type=float, default=float(CTRL_HZ))
    p.add_argument("--settle-s",     type=float, default=3.0,
                   help="Neutral hold between segments (s).")
    p.add_argument("--step-hold-s",  type=float, default=0.8,
                   help="Hold per step level (s).")
    p.add_argument("--chirp-s",      type=float, default=30.0,
                   help="Chirp duration per joint pass (s).")
    p.add_argument("--prbs-s",       type=float, default=60.0,
                   help="PRBS duration per finger (s).")
    p.add_argument("--wait-timeout-s", type=float, default=10.0)
    return p.parse_args(rospy.myargv()[1:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    rospy.init_node("sysid_collect", anonymous=True)

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        out_dir = get_data_dir() / f"{today}-sysid" / "auto"
    out_dir.mkdir(parents=True, exist_ok=True)
    rospy.loginfo("Output directory: %s", out_dir)

    hand = LeapHandInterface()
    recorder = Recorder()

    # Wait for first state
    rospy.loginfo("Waiting for LEAP hand state ...")
    t0 = time.monotonic()
    while not rospy.is_shutdown():
        if hand.get_state() is not None:
            break
        if time.monotonic() - t0 > args.wait_timeout_s:
            raise RuntimeError("Timed out waiting for LEAP hand state.")
        rospy.sleep(0.05)

    # Move to rest and hold briefly; discard warm-up data
    interpolate_to(hand, recorder, args.ctrl_hz, REST_Q, duration_s=2.0)
    settle(hand, recorder, args.ctrl_hz, args.settle_s)
    recorder.clear()

    for finger in args.fingers:
        if rospy.is_shutdown():
            break

        joint_ids  = FINGER_JOINTS[finger]
        amplitudes = AMPLITUDES[joint_ids]

        # Build the finger's neutral pose: zeros everywhere, NEUTRAL_Q for active joints
        finger_neutral = REST_Q.copy()
        for jid in joint_ids:
            finger_neutral[jid] = NEUTRAL_Q[jid]

        rospy.loginfo("=" * 50)
        rospy.loginfo("FINGER: %s  (joints %s)", finger.upper(), joint_ids)
        rospy.loginfo("  finger neutral: %s", finger_neutral[joint_ids])
        rospy.loginfo("  amp (±80%%):    %s", amplitudes)
        rospy.loginfo("=" * 50)

        recorder.clear()

        # Move active finger to its neutral; other fingers stay at zero
        interpolate_to(hand, recorder, args.ctrl_hz, finger_neutral, duration_s=2.0)
        settle(hand, recorder, args.ctrl_hz, args.settle_s, q_hold=finger_neutral)
        recorder.clear()  # discard transition data, only keep excitation

        # 1. Step ladder — settle at finger_neutral between passes
        step_ladder(hand, recorder, joint_ids, amplitudes, args.ctrl_hz, args.step_hold_s)
        settle(hand, recorder, args.ctrl_hz, args.settle_s, q_hold=finger_neutral)

        # 2. Chirp
        chirp(hand, recorder, joint_ids, amplitudes, args.ctrl_hz, args.chirp_s)
        settle(hand, recorder, args.ctrl_hz, args.settle_s, q_hold=finger_neutral)

        # 3. PRBS
        prbs(hand, recorder, joint_ids, amplitudes, args.ctrl_hz, args.prbs_s)

        # Return to REST before switching to next finger
        interpolate_to(hand, recorder, args.ctrl_hz, REST_Q, duration_s=2.0)
        settle(hand, recorder, args.ctrl_hz, args.settle_s, q_hold=REST_Q)

        # Save this finger's excitation data
        ts = datetime.now().strftime("%H%M%S")
        out_path = out_dir / f"{finger}_{ts}.npz"
        recorder.save(out_path)

    # Final rest
    interpolate_to(hand, recorder, args.ctrl_hz, REST_Q, duration_s=2.0)
    rospy.loginfo("sysid_collect complete. Files saved to: %s", out_dir)


if __name__ == "__main__":
    main()
