"""Run a standalone policy server over ZMQ for remote deployment clients."""

from __future__ import annotations

from dataclasses import dataclass
import sys

import tyro

from mjlab.tasks.registry import list_tasks
from in_hand_rotation_mjlab.policy_server import PolicyServerConfig
from in_hand_rotation_mjlab.policy_server.zmq_transport import PolicyZmqServer, ZmqServerConfig


@dataclass(frozen=True)
class PolicyServerZmqCliConfig:
  checkpoint_file: str
  bind: str = "tcp://127.0.0.1:5555"
  device: str | None = None
  play: bool = True
  shutdown_on_close: bool = True
  infer_log_interval: int = 50
  log_first_n_infers: int = 3


def main() -> None:
  # Import tasks to populate registry.
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    PolicyServerZmqCliConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  server = PolicyZmqServer(
    task_id=chosen_task,
    cfg=PolicyServerConfig(
      checkpoint_file=args.checkpoint_file,
      device=args.device,
      play=args.play,
    ),
    transport_cfg=ZmqServerConfig(
      bind=args.bind,
      shutdown_on_close=args.shutdown_on_close,
      infer_log_interval=args.infer_log_interval,
      log_first_n_infers=args.log_first_n_infers,
    ),
  )
  server.serve_forever()


if __name__ == "__main__":
  main()
