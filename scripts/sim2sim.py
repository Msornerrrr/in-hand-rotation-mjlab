"""Run sim2sim rollout with native MuJoCo dynamics and a trained policy."""

import sys

import tyro

from mjlab.tasks.registry import list_tasks
from in_hand_rotation_mjlab.sim2sim import NativeSim2SimConfig, run_native_sim2sim


def main():
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    NativeSim2SimConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  run_native_sim2sim(chosen_task, args)


if __name__ == "__main__":
  main()
