"""Shared helpers for restoring a teleop-saved viser camera.

The teleop script's "save pose + camera" button writes
``logs/teleop_<task_id>_pose.json`` (repo-root anchored). Viewer scripts
restore that camera by pushing it to each NEW client from the sim loop --
NEVER from an on_client_connect handler, because mjviser registers its own
connect handler last and clobbers connect-time cameras.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def teleop_pose_file(repo_root: Path, task_id: str) -> Path:
    return repo_root / "logs" / f"teleop_{task_id}_pose.json"


def load_saved_camera(pose_file: Path) -> dict | None:
    """Newest client's saved camera from a teleop pose file, or None."""
    if not pose_file.exists():
        return None
    try:
        cameras = json.loads(pose_file.read_text()).get("cameras", {})
        return cameras[max(cameras, key=int)] if cameras else None
    except (ValueError, KeyError, OSError) as exc:
        print(f"[WARN] could not read saved camera from {pose_file}: {exc}")
        return None


def make_camera_restorer(server, camera: dict):
    """Callable that applies ``camera`` once to every not-yet-seen client.

    Call it from the sim/render loop (each step is fine -- it no-ops for
    already-restored clients).
    """
    restored: set[int] = set()

    def restore() -> None:
        for client_id, client in server.get_clients().items():
            if int(client_id) in restored:
                continue
            restored.add(int(client_id))
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.position = tuple(camera["position"])
            client.camera.look_at = tuple(camera["look_at"])
            client.camera.fov = math.radians(camera.get("fov_deg", 60.0))

    return restore
