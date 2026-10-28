"""Fast CPU checks: the tasks register, their env configs build, and every released checkpoint loads."""
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

TASKS = [
  "Mjlab-Leap-Left-HandCube-Rotate",
  "Mjlab-Leap-Right-HandCube-Rotate",
  "Mjlab-Leap-Left-Custom-HandCube-Rotate",
  "Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear",
  *(f"Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube{c}{s}" for c in "567" for s in ("", "-Final", "-S2R3", "-S2R3-Gear")),
  "Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube6-FinalW",
  "Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube7-FinalW",
  *(f"Mjlab-Leap-Left-Custom-Fingertip-Rotate-PalmDown-{s}" for s in ("MixTeacherF", "Mix", "MixFT", "MixFTF")),
]


@pytest.fixture(scope="module")
def registry():
  import in_hand_rotation_mjlab.tasks  # noqa: F401
  from mjlab.tasks import registry as r

  return r


def test_tasks_registered(registry):
  leap = sorted(t for t in registry.list_tasks() if t.startswith("Mjlab-Leap"))
  assert leap == sorted(TASKS)


@pytest.mark.parametrize("task", TASKS)
def test_env_cfgs_build(registry, task):
  for play in (False, True):
    cfg = registry.load_env_cfg(task, play=play)
    assert cfg.scene.num_envs > 0
  assert registry.load_rl_cfg(task).experiment_name


def test_package_data():
  pkg = ROOT / "src/in_hand_rotation_mjlab"
  assert (pkg / "tasks/fingertip_rotate/data/palmdown_canon4.npz").is_file()
  for c in "567":
    assert (pkg / f"tasks/hand_cube/cache/leap_left_custom_grasp_cache_cube{c}cm.npz").is_file()


@pytest.mark.parametrize("path", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "ckpts").rglob("*.pt")
                                        if not p.name.endswith("_models.pt")))
def test_checkpoints_load(path):
  ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
  if ck.get("kind") == "wm_actor":
    from in_hand_rotation_mjlab.world_model.policy import Actor

    actor = Actor.from_checkpoint(ck)
    a = actor(torch.zeros(3, ck["obs_dim"]))
    assert a.shape == (3, ck["act_dim"]) and torch.isfinite(a).all()
  else:
    assert {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"} <= set(ck)
