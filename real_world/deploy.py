#!/usr/bin/env python
"""Deploy a server-hosted policy on LEAP hand hardware via ZMQ."""

from __future__ import annotations

import argparse
import base64
import time
from typing import Any

import numpy as np
import rospy
import zmq

from leap_hand_interface import LeapHandInterface


_NUM_JOINTS = 16

# ---------------------------------------------------------------------------
# Joint-order / sign remapping between hardware and server
# ---------------------------------------------------------------------------
# Hardware state topic publishes joints in this order:
#   [if_rot, if_mcp, if_pip, if_dip,  mf_rot, mf_mcp, mf_pip, mf_dip,
#    rf_rot, rf_mcp, rf_pip, rf_dip,  th_cmc, th_axl, th_mcp, th_ipl]
# Server expects:
#   [if_mcp, if_rot, if_pip, if_dip,  mf_mcp, mf_rot, mf_pip, mf_dip,
#    rf_mcp, rf_rot, rf_pip, rf_dip,  th_cmc, th_axl, th_mcp, th_ipl]
#   i.e. swap pairs (0,1), (4,5), (8,9); negate index 12 (th_cmc).

_HW_TO_SERVER = np.array([
  1,  0,  2,  3,   # index:  rot, mcp, pip, dip
  5,  4,  6,  7,   # middle: rot, mcp, pip, dip
  9,  8, 10, 11,   # ring:   rot, mcp, pip, dip
 12, 13, 14, 15,   # thumb:  cmc, axl, mcp, ipl
], dtype=np.int64)

_SERVER_TO_HW = np.argsort(_HW_TO_SERVER).astype(np.int64)


def _hw_to_server(q: np.ndarray) -> np.ndarray:
  q = q[_HW_TO_SERVER].copy()
  q[12] *= -1
  return q


def _server_to_hw(q: np.ndarray) -> np.ndarray:
  q = q.copy()
  q[12] *= -1
  return q[_SERVER_TO_HW]


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------

def _encode_array(array: np.ndarray) -> dict[str, Any]:
  x = np.ascontiguousarray(array)
  return {
    "dtype": str(x.dtype),
    "shape": list(x.shape),
    "data_b64": base64.b64encode(x.tobytes(order="C")).decode("ascii"),
  }


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
  dtype = np.dtype(str(payload["dtype"]))
  shape = tuple(int(v) for v in payload["shape"])
  raw = base64.b64decode(str(payload["data_b64"]))
  return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _encode_obs(
  *,
  q: np.ndarray,
  prev_cmd: np.ndarray,
  action_term_name: str,
) -> dict[str, Any]:
  qs     = _hw_to_server(q)
  prev_s = _hw_to_server(prev_cmd)
  row = lambda x: np.asarray(x[None, :], dtype=np.float32)
  return {
    "term_values": {
      "joint_pos":                _encode_array(row(qs)),
      "prev_commanded_joint_pos": _encode_array(row(prev_s)),
    },
    "action_term_joint_pos": {
      action_term_name: _encode_array(row(qs)),
    },
  }


# ---------------------------------------------------------------------------
# ZMQ RPC client
# ---------------------------------------------------------------------------

class _ZmqRpcClient:
  def __init__(self, *, endpoint: str, timeout_ms: int):
    self.endpoint = endpoint
    self._ctx = zmq.Context.instance()
    self._socket = self._ctx.socket(zmq.REQ)
    self._socket.setsockopt(zmq.LINGER, 0)
    self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
    self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    self._socket.connect(endpoint)

  def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
    try:
      self._socket.send_json(payload)
      response = self._socket.recv_json()
    except zmq.Again as exc:
      raise TimeoutError(
        f"Timed out waiting for policy server at '{self.endpoint}'."
      ) from exc
    if not bool(response.get("ok", False)):
      raise RuntimeError(str(response.get("error", "Unknown policy server error.")))
    return response

  def ping(self) -> None:
    self._request({"type": "ping"})

  def init(self, *, step_dt: float, max_episode_steps: int) -> dict[str, Any]:
    return self._request({
      "type": "init",
      "step_dt": float(step_dt),
      "max_episode_steps": int(max_episode_steps),
    })

  def reset(self, obs: dict[str, Any]) -> None:
    self._request({"type": "reset", "observation": obs})

  def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
    response = self._request({"type": "infer", "observation": obs})
    return {
      k: _decode_array(v)
      for k, v in response["action"]["joint_position_commands"].items()
    }

  def close(self, *, shutdown_server: bool = False) -> None:
    if shutdown_server:
      try:
        self._request({"type": "close"})
      except Exception:
        pass
    self._socket.close(linger=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Deploy a ZMQ policy server on LEAP hand hardware."
  )
  parser.add_argument("--server-endpoint",      type=str,   required=True)
  parser.add_argument("--server-timeout-ms",    type=int,   default=30_000)
  parser.add_argument("--policy-hz",            type=float, default=20.0)
  parser.add_argument("--max-steps",            type=int,   default=0,
                      help="0 = run until Ctrl+C.")
  parser.add_argument("--max-episode-steps",    type=int,   default=10_000_000)
  parser.add_argument("--action-term-name",     type=str,   default="joint_pos")
  parser.add_argument("--wait-state-timeout-s", type=float, default=10.0)
  parser.add_argument("--init-duration-s",      type=float, default=3.0,
                      help="Seconds to interpolate from current pose to server default pose.")
  parser.add_argument("--shutdown-server-on-exit", action="store_true")
  return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  args = _parse_args()
  rospy.init_node("mjlab_policy_deploy_client", anonymous=True)

  hand = LeapHandInterface()

  rospy.loginfo("Waiting for first joint state (timeout=%.1fs) ...", args.wait_state_timeout_s)
  start = time.monotonic()
  while not rospy.is_shutdown():
    if hand.get_state() is not None:
      break
    if (time.monotonic() - start) > args.wait_state_timeout_s:
      raise RuntimeError("Timed out waiting for LEAP hand state.")
    rospy.sleep(0.02)

  q0 = hand.get_joint_positions_numpy()
  if q0.shape[0] != _NUM_JOINTS:
    raise RuntimeError(f"Expected {_NUM_JOINTS} joints, got {q0.shape[0]}.")

  # --- Connect and init: server returns default pose and limits ---
  rpc = _ZmqRpcClient(endpoint=args.server_endpoint, timeout_ms=args.server_timeout_ms)
  policy_dt = 1.0 / float(args.policy_hz)

  rospy.loginfo("Connecting to policy server: %s", args.server_endpoint)
  rpc.ping()
  init_ack = rpc.init(step_dt=policy_dt, max_episode_steps=args.max_episode_steps)

  term_md = init_ack["action_term_metadata"][args.action_term_name]
  # default_joint_pos shape: [1, 16], in server joint order
  default_q_server = _decode_array(term_md["default_joint_pos"]).reshape(-1).astype(np.float32)
  default_q_hw = _server_to_hw(default_q_server)

  # --- Move hand to server default pose before starting policy ---
  rospy.loginfo("Moving to default pose over %.1fs ...", args.init_duration_s)
  move_rate = rospy.Rate(args.policy_hz)
  n_steps = max(1, int(args.init_duration_s * args.policy_hz))
  for i in range(n_steps):
    if rospy.is_shutdown():
      return
    alpha = (i + 1) / n_steps
    q_interp = (1.0 - alpha) * q0 + alpha * default_q_hw
    hand.send_command(q_interp)
    move_rate.sleep()
  rospy.loginfo("Default pose reached.")

  # Re-read state after move
  q_start  = hand.get_joint_positions_numpy()
  prev_cmd = q_start.copy()

  rpc.reset(_encode_obs(q=q_start, prev_cmd=prev_cmd, action_term_name=args.action_term_name))

  rate      = rospy.Rate(args.policy_hz)
  step      = 0
  total_ms  = 0.0
  infer_ms  = 0.0
  LOG_EVERY = 50
  rospy.loginfo("Policy loop started at %.2f Hz", args.policy_hz)

  try:
    while not rospy.is_shutdown():
      t0 = time.perf_counter()

      q   = hand.get_joint_positions_numpy()
      obs = _encode_obs(q=q, prev_cmd=prev_cmd, action_term_name=args.action_term_name)

      t1      = time.perf_counter()
      cmd_map = rpc.infer(obs)
      t2      = time.perf_counter()

      cmd = _server_to_hw(cmd_map[args.action_term_name].reshape(-1).astype(np.float32))
      hand.send_command(cmd)

      t3 = time.perf_counter()
      total_ms += (t3 - t0) * 1000.0
      infer_ms += (t2 - t1) * 1000.0

      prev_cmd = cmd
      step += 1

      if step % LOG_EVERY == 0:
        avg_total = total_ms / step
        rospy.loginfo(
          "step=%d  iter_ms=%.2f  infer_ms=%.2f  avg_iter_ms=%.2f  avg_infer_ms=%.2f  avg_hz=%.1f",
          step,
          (t3 - t0) * 1000.0,
          (t2 - t1) * 1000.0,
          avg_total,
          infer_ms / step,
          1000.0 / avg_total,
        )

      if args.max_steps > 0 and step >= args.max_steps:
        rospy.loginfo("Reached max_steps=%d, exiting.", args.max_steps)
        break

      rate.sleep()
  finally:
    rpc.close(shutdown_server=args.shutdown_server_on_exit)


if __name__ == "__main__":
  main()
