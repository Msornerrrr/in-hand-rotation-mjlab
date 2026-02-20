from __future__ import annotations

import torch

from in_hand_rotation_mjlab.sim2sim.native.client import NativeMujocoClient
from in_hand_rotation_mjlab.sim2sim.native.config import NativeSim2SimConfig
from in_hand_rotation_mjlab.policy_server import MjlabPolicyServer, PolicyServerConfig


class NativeMujocoSim2SimRunner:
  """Orchestrates server-client sim2sim rollouts.

  Server responsibilities:
  - observation post-processing (clip/scale/delay/history/stack)
  - policy inference
  - action-space mapping from policy output to joint-position targets

  Client responsibilities:
  - native MuJoCo scene/state/events/physics stepping
  - raw observation term extraction
  - execution of server joint-position commands
  - local video capture
  """

  def __init__(self, task_id: str, cfg: NativeSim2SimConfig):
    self.task_id = task_id
    self.cfg = cfg

    self.client = NativeMujocoClient(task_id=task_id, cfg=cfg)
    self._local_server: MjlabPolicyServer | None = None
    self._remote_server = None

    if cfg.server_endpoint is None:
      self._local_server = MjlabPolicyServer(
        task_id=task_id,
        cfg=PolicyServerConfig(
          checkpoint_file=cfg.checkpoint_file,
          device=cfg.device,
          play=cfg.play,
        ),
        action_term_metadata=self.client.build_server_action_metadata(),
        step_dt=self.client.step_dt,
        max_episode_steps=self.client.max_episode_steps,
      )
    else:
      from in_hand_rotation_mjlab.policy_server.zmq_transport import PolicyZmqClient, ZmqClientConfig

      self._remote_server = PolicyZmqClient(
        ZmqClientConfig(
          endpoint=cfg.server_endpoint,
          request_timeout_ms=cfg.server_timeout_ms,
          tensor_device=self.client.device,
        )
      )
      self._remote_server.init(
        step_dt=self.client.step_dt,
        max_episode_steps=self.client.max_episode_steps,
      )

  def rollout_description(self) -> str:
    if self._local_server is not None:
      return (
        f"{self.client.rollout_description()}\n"
        f"[INFO] Policy server mode: in-process ({self._local_server.device})"
      )
    return (
      f"{self.client.rollout_description()}\n"
      f"[INFO] Policy server mode: remote ({self.cfg.server_endpoint})"
    )

  def run(self) -> None:
    print(self.rollout_description())
    success = False
    try:
      obs_packet = self.client.reset_rollout()
      if self._remote_server is not None:
        self._remote_server.reset(obs_packet)
      else:
        assert self._local_server is not None
        self._local_server.reset(obs_packet)
      with torch.no_grad():
        for _ in range(self.client.num_steps):
          if self._remote_server is not None:
            action_packet = self._remote_server.infer(obs_packet)
          else:
            assert self._local_server is not None
            action_packet = self._local_server.infer(obs_packet)
          obs_packet = self.client.step_rollout(action_packet)
      success = True
    finally:
      self.client.finish_rollout(success=success)
      if self._remote_server is not None:
        self._remote_server.close(
          shutdown_server=self.cfg.shutdown_server_on_finish
        )
      if success:
        print("[INFO] Native sim2sim rollout completed.")


def run_native_sim2sim(task_id: str, cfg: NativeSim2SimConfig) -> None:
  runner = NativeMujocoSim2SimRunner(task_id, cfg)
  runner.run()
