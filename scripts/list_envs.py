"""List this repository's tasks (``--keyword`` filters; ``--all`` also lists mjlab's built-in tasks)."""

import tyro
from prettytable import PrettyTable

import in_hand_rotation_mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks


def list_environments(keyword: str | None = None, all: bool = False):
  """List all registered environments.

  Args:
    keyword: Optional filter to only show environments containing this keyword.
  """
  table = PrettyTable(["#", "Task ID"])
  table.title = "In-hand rotation tasks"
  table.align["Task ID"] = "l"

  all_tasks = [t for t in list_tasks() if all or t.startswith("Mjlab-Leap")]
  idx = 0
  for task_id in all_tasks:
    try:
      # Optionally filter by keyword.
      if keyword and keyword.lower() not in task_id.lower():
        continue

      table.add_row([idx + 1, task_id])
      idx += 1
    except Exception:
      continue

  print(table)
  if idx == 0:
    msg = "[INFO] No tasks matched"
    if keyword:
      msg += f" keyword '{keyword}'"
    print(msg)
  return idx


def main():
  return tyro.cli(list_environments)


if __name__ == "__main__":
  main()
