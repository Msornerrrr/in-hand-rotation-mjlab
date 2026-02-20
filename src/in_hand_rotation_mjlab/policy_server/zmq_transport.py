from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import zmq

from in_hand_rotation_mjlab.policy_server.config import PolicyServerConfig
from in_hand_rotation_mjlab.policy_server.contracts import (
  ActionTermMetadata,
  ObservationPacket,
  ServerActionPacket,
)
from in_hand_rotation_mjlab.policy_server.bootstrap import build_server_action_metadata_from_task
from in_hand_rotation_mjlab.policy_server.policy_server import MjlabPolicyServer
from in_hand_rotation_mjlab.policy_server.replay import load_replay_trajectory
from in_hand_rotation_mjlab.policy_server.replay_server import ReplayPolicyServer
from in_hand_rotation_mjlab.policy_server.wire import (
  decode_observation_packet,
  decode_server_action_packet,
  encode_server_action_packet,
  encode_observation_packet,
  encode_action_metadata_by_term,
)


@dataclass(frozen=True)
class ZmqClientConfig:
  endpoint: str = "tcp://127.0.0.1:5555"
  request_timeout_ms: int = 30000
  tensor_device: str = "cpu"


class PolicyZmqClient:
  def __init__(self, cfg: ZmqClientConfig):
    self.cfg = cfg
    self._ctx = zmq.Context.instance()
    self._socket = self._ctx.socket(zmq.REQ)
    self._socket.setsockopt(zmq.LINGER, 0)
    self._socket.setsockopt(zmq.RCVTIMEO, int(cfg.request_timeout_ms))
    self._socket.setsockopt(zmq.SNDTIMEO, int(cfg.request_timeout_ms))
    self._socket.connect(cfg.endpoint)

  def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
    try:
      self._socket.send_json(payload)
      response = self._socket.recv_json()
    except zmq.Again as exc:
      raise TimeoutError(
        f"Timed out waiting for policy server at '{self.cfg.endpoint}'."
      ) from exc

    if not bool(response.get("ok", False)):
      raise RuntimeError(str(response.get("error", "Unknown policy server error.")))
    return response

  def ping(self) -> dict[str, Any]:
    return self._request({"type": "ping"})

  def init(
    self,
    *,
    step_dt: float,
    max_episode_steps: int,
  ) -> None:
    self._request(
      {
        "type": "init",
        "step_dt": float(step_dt),
        "max_episode_steps": int(max_episode_steps),
      }
    )

  def reset(self, observation: ObservationPacket) -> None:
    self._request(
      {
        "type": "reset",
        "observation": encode_observation_packet(observation),
      }
    )

  def infer(self, observation: ObservationPacket) -> ServerActionPacket:
    response = self._request(
      {
        "type": "infer",
        "observation": encode_observation_packet(observation),
      }
    )
    return decode_server_action_packet(
      response["action"],
      device=self.cfg.tensor_device,
    )

  def close(self, *, shutdown_server: bool = False) -> None:
    if shutdown_server:
      try:
        self._request({"type": "close"})
      except Exception:
        pass
    self._socket.close(linger=0)


@dataclass(frozen=True)
class ZmqServerConfig:
  bind: str = "tcp://127.0.0.1:5555"
  shutdown_on_close: bool = True
  infer_log_interval: int = 50
  log_first_n_infers: int = 3


class PolicyZmqServer:
  def __init__(
    self,
    *,
    task_id: str,
    cfg: PolicyServerConfig,
    transport_cfg: ZmqServerConfig,
  ):
    self.task_id = task_id
    self.cfg = cfg
    self.transport_cfg = transport_cfg
    self._ctx = zmq.Context.instance()
    self._socket = self._ctx.socket(zmq.REP)
    self._socket.setsockopt(zmq.LINGER, 0)
    self._socket.bind(transport_cfg.bind)
    self._server: MjlabPolicyServer | None = None
    self._cached_action_metadata: dict[str, ActionTermMetadata] | None = None
    self._infer_count = 0
    self._infer_total_ms = 0.0
    self._infer_decode_ms = 0.0
    self._infer_model_ms = 0.0
    self._infer_encode_ms = 0.0

  @staticmethod
  def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
      torch.cuda.synchronize(device=device)

  def _record_infer_timing(
    self,
    *,
    decode_ms: float,
    model_ms: float,
    encode_ms: float,
    total_ms: float,
  ) -> None:
    self._infer_count += 1
    self._infer_decode_ms += decode_ms
    self._infer_model_ms += model_ms
    self._infer_encode_ms += encode_ms
    self._infer_total_ms += total_ms

    interval = max(1, int(self.transport_cfg.infer_log_interval))
    log_first_n = max(0, int(self.transport_cfg.log_first_n_infers))
    should_log = self._infer_count <= log_first_n or (self._infer_count % interval == 0)
    if not should_log:
      return

    avg_decode = self._infer_decode_ms / self._infer_count
    avg_model = self._infer_model_ms / self._infer_count
    avg_encode = self._infer_encode_ms / self._infer_count
    avg_total = self._infer_total_ms / self._infer_count
    avg_hz = 1000.0 / avg_total if avg_total > 1e-6 else 0.0
    print(
      "[INFO] Policy infer timing: "
      f"count={self._infer_count}, "
      f"model_ms={model_ms:.3f}, total_ms={total_ms:.3f}, "
      f"decode_ms={decode_ms:.3f}, encode_ms={encode_ms:.3f}, "
      f"avg_model_ms={avg_model:.3f}, avg_total_ms={avg_total:.3f}, avg_hz={avg_hz:.2f}"
    )

  @staticmethod
  def _ok(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True, **kwargs}

  @staticmethod
  def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}

  def _ensure_server(self) -> MjlabPolicyServer:
    if self._server is None:
      raise RuntimeError("Policy server is not initialized. Send an 'init' request first.")
    return self._server

  def _get_action_metadata(self) -> dict[str, ActionTermMetadata]:
    if self._cached_action_metadata is None:
      self._cached_action_metadata = build_server_action_metadata_from_task(
        task_id=self.task_id,
        play=self.cfg.play,
        device="cpu",
      )
    return self._cached_action_metadata

  def _handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    request_type = str(request.get("type", ""))

    if request_type == "ping":
      return self._ok(type="pong"), True

    if request_type == "init":
      metadata = self._get_action_metadata()
      self._server = MjlabPolicyServer(
        task_id=self.task_id,
        cfg=self.cfg,
        action_term_metadata=metadata,
        step_dt=float(request["step_dt"]),
        max_episode_steps=int(request["max_episode_steps"]),
      )
      return (
        self._ok(
          type="init_ack",
          device=self._server.device,
          action_term_metadata=encode_action_metadata_by_term(metadata),
          action_terms=list(metadata.keys()),
          actor_observation_terms=list(self._server.obs_processor.cfg.terms.keys()),
          expected_actor_obs_dim=self._server.expected_actor_obs_dim,
        ),
        True,
      )

    if request_type == "reset":
      server = self._ensure_server()
      observation = decode_observation_packet(
        request["observation"],
        device=server.device,
      )
      server.reset(observation)
      return self._ok(type="reset_ack"), True

    if request_type == "infer":
      server = self._ensure_server()
      t_total_start = time.perf_counter()
      t_decode_start = t_total_start
      observation = decode_observation_packet(
        request["observation"],
        device=server.device,
      )
      t_decode_end = time.perf_counter()

      self._sync_if_cuda(server.device)
      t_model_start = time.perf_counter()
      action = server.infer(observation)
      self._sync_if_cuda(server.device)
      t_model_end = time.perf_counter()

      t_encode_start = t_model_end
      encoded_action = encode_server_action_packet(action)
      t_encode_end = time.perf_counter()

      decode_ms = (t_decode_end - t_decode_start) * 1000.0
      model_ms = (t_model_end - t_model_start) * 1000.0
      encode_ms = (t_encode_end - t_encode_start) * 1000.0
      total_ms = (t_encode_end - t_total_start) * 1000.0
      self._record_infer_timing(
        decode_ms=decode_ms,
        model_ms=model_ms,
        encode_ms=encode_ms,
        total_ms=total_ms,
      )
      return self._ok(type="infer_ack", action=encoded_action), True

    if request_type == "close":
      should_continue = not self.transport_cfg.shutdown_on_close
      return self._ok(type="close_ack"), should_continue

    return self._err(f"Unsupported request type: '{request_type}'."), True

  def serve_forever(self) -> None:
    print(
      "[INFO] Policy ZMQ server listening: "
      f"bind={self.transport_cfg.bind}, task={self.task_id}"
    )
    should_continue = True
    while should_continue:
      try:
        request = self._socket.recv_json()
      except KeyboardInterrupt:
        break
      except Exception as exc:
        # REP socket still expects a send; skip only if receive itself failed.
        print(f"[WARN] Failed to receive policy request: {exc}")
        continue

      try:
        response, should_continue = self._handle(request)
      except Exception as exc:
        response = self._err(f"{type(exc).__name__}: {exc}")

      try:
        self._socket.send_json(response)
      except Exception as exc:
        print(f"[WARN] Failed to send policy response: {exc}")
        break

    self._socket.close(linger=0)
    print("[INFO] Policy ZMQ server stopped.")


class ReplayZmqServer:
  """ZMQ server that replays pre-recorded action trajectories."""

  def __init__(
    self,
    *,
    task_id: str,
    play: bool,
    trajectory_file: str,
    loop_trajectory: bool,
    transport_cfg: ZmqServerConfig,
  ):
    self.task_id = task_id
    self.play = play
    self.trajectory_file = trajectory_file
    self.loop_trajectory = loop_trajectory
    self.transport_cfg = transport_cfg
    self._ctx = zmq.Context.instance()
    self._socket = self._ctx.socket(zmq.REP)
    self._socket.setsockopt(zmq.LINGER, 0)
    self._socket.bind(transport_cfg.bind)
    self._server: ReplayPolicyServer | None = None
    self._cached_action_metadata: dict[str, ActionTermMetadata] | None = None
    self._infer_count = 0
    self._infer_total_ms = 0.0
    self._infer_decode_ms = 0.0
    self._infer_model_ms = 0.0
    self._infer_encode_ms = 0.0

  @staticmethod
  def _ok(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True, **kwargs}

  @staticmethod
  def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}

  def _record_infer_timing(
    self,
    *,
    decode_ms: float,
    replay_ms: float,
    encode_ms: float,
    total_ms: float,
  ) -> None:
    self._infer_count += 1
    self._infer_decode_ms += decode_ms
    self._infer_model_ms += replay_ms
    self._infer_encode_ms += encode_ms
    self._infer_total_ms += total_ms

    interval = max(1, int(self.transport_cfg.infer_log_interval))
    log_first_n = max(0, int(self.transport_cfg.log_first_n_infers))
    should_log = self._infer_count <= log_first_n or (self._infer_count % interval == 0)
    if not should_log:
      return

    avg_replay = self._infer_model_ms / self._infer_count
    avg_total = self._infer_total_ms / self._infer_count
    avg_hz = 1000.0 / avg_total if avg_total > 1e-6 else 0.0
    print(
      "[INFO] Replay infer timing: "
      f"count={self._infer_count}, "
      f"replay_ms={replay_ms:.3f}, total_ms={total_ms:.3f}, "
      f"decode_ms={decode_ms:.3f}, encode_ms={encode_ms:.3f}, "
      f"avg_replay_ms={avg_replay:.3f}, avg_total_ms={avg_total:.3f}, avg_hz={avg_hz:.2f}"
    )

  def _ensure_server(self) -> ReplayPolicyServer:
    if self._server is None:
      raise RuntimeError("Replay server is not initialized. Send an 'init' request first.")
    return self._server

  def _get_action_metadata(self) -> dict[str, ActionTermMetadata]:
    if self._cached_action_metadata is None:
      self._cached_action_metadata = build_server_action_metadata_from_task(
        task_id=self.task_id,
        play=self.play,
        device="cpu",
      )
    return self._cached_action_metadata

  def _handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    request_type = str(request.get("type", ""))

    if request_type == "ping":
      return self._ok(type="pong"), True

    if request_type == "init":
      metadata = self._get_action_metadata()
      trajectory = load_replay_trajectory(self.trajectory_file, device="cpu")
      requested_step_dt = float(request["step_dt"])
      if abs(requested_step_dt - trajectory.step_dt) > 1e-6:
        print(
          "[WARN] Replay step_dt mismatch: "
          f"client requested {requested_step_dt:.6f}s, "
          f"trajectory was recorded at {trajectory.step_dt:.6f}s."
        )
      self._server = ReplayPolicyServer(
        action_term_metadata=metadata,
        trajectory=trajectory,
        loop_trajectory=self.loop_trajectory,
      )
      return (
        self._ok(
          type="init_ack",
          device="cpu",
          action_term_metadata=encode_action_metadata_by_term(metadata),
          action_terms=list(metadata.keys()),
          actor_observation_terms=[],
          expected_actor_obs_dim=None,
          replay_num_steps=trajectory.num_steps,
          replay_step_dt=trajectory.step_dt,
          replay_task_id=trajectory.task_id,
          replay_checkpoint_file=trajectory.checkpoint_file,
        ),
        True,
      )

    if request_type == "reset":
      server = self._ensure_server()
      observation = decode_observation_packet(request["observation"], device="cpu")
      server.reset(observation)
      return self._ok(type="reset_ack"), True

    if request_type == "infer":
      server = self._ensure_server()
      t_total_start = time.perf_counter()
      t_decode_start = t_total_start
      observation = decode_observation_packet(request["observation"], device="cpu")
      t_decode_end = time.perf_counter()

      t_replay_start = t_decode_end
      action = server.infer(observation)
      t_replay_end = time.perf_counter()

      t_encode_start = t_replay_end
      encoded_action = encode_server_action_packet(action)
      t_encode_end = time.perf_counter()

      decode_ms = (t_decode_end - t_decode_start) * 1000.0
      replay_ms = (t_replay_end - t_replay_start) * 1000.0
      encode_ms = (t_encode_end - t_encode_start) * 1000.0
      total_ms = (t_encode_end - t_total_start) * 1000.0
      self._record_infer_timing(
        decode_ms=decode_ms,
        replay_ms=replay_ms,
        encode_ms=encode_ms,
        total_ms=total_ms,
      )
      return self._ok(type="infer_ack", action=encoded_action), True

    if request_type == "close":
      should_continue = not self.transport_cfg.shutdown_on_close
      return self._ok(type="close_ack"), should_continue

    return self._err(f"Unsupported request type: '{request_type}'."), True

  def serve_forever(self) -> None:
    print(
      "[INFO] Replay ZMQ server listening: "
      f"bind={self.transport_cfg.bind}, task={self.task_id}, "
      f"trajectory={self.trajectory_file}"
    )
    should_continue = True
    while should_continue:
      try:
        request = self._socket.recv_json()
      except KeyboardInterrupt:
        break
      except Exception as exc:
        print(f"[WARN] Failed to receive replay request: {exc}")
        continue

      try:
        response, should_continue = self._handle(request)
      except Exception as exc:
        response = self._err(f"{type(exc).__name__}: {exc}")

      try:
        self._socket.send_json(response)
      except Exception as exc:
        print(f"[WARN] Failed to send replay response: {exc}")
        break

    self._socket.close(linger=0)
    print("[INFO] Replay ZMQ server stopped.")
