"""What a task's robot physically is, summarized and fingerprinted.

A policy is only valid on the plant it was trained on: the robot's actuators (gains,
effort limit, armature, joint friction, command delay), the robot-plant randomization
(events in ``PLANT_EVENTS``), the physics options and the compiled robot model (XML
geometry, joint ranges, contact parameters). ``plant_summary`` collects them from an env
cfg, or from a training run's ``params/env.yaml``, so that

- ``tests/test_plant_pins.py`` pins every registered task's plant and fails when one
    changes, and
- loading a checkpoint into a task whose plant differs from the one it was trained on
    warns (``check_checkpoint_plant``).

The LEAP servo version (``robots/leap_hand/leap_plant.py``) is named when the actuators
match one exactly.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import types
from pathlib import Path

import numpy as np

# Events that change the robot's own physics (not the object's or the scene's). Add new
# ones here so the pins see them.
PLANT_EVENTS = (
    "dr_action_delay", "dr_effort_limits", "dr_motor_damping", "dr_motor_friction", "dr_reflected_inertia",
    "dr_robot_link_masses", "dr_robot_pd_gains", "dr_shared_contact_friction", "dr_split_contact_friction",
    "randomize_actuator_gains", "randomize_hand_frictionloss", "scale_leap_hand_friction", "dr_leap_gear_servo",
)
_ACTUATOR_KEYS = ("target_names_expr", "stiffness", "damping", "effort_limit", "armature", "frictionloss",
                  "viscous_damping", "delay_min_lag", "delay_max_lag", "delay_hold_prob", "delay_update_period",
                  "delay_per_env_phase",
                  # LeapGearServoActuatorCfg (robots/leap_hand/gear_servo.py)
                  "current_gain", "current_damping", "current_limit", "torque_constant", "gear_friction",
                  "dry_friction", "side_joints", "side_gain_scale", "host_period", "lead_clamp", "friction_solref",
                  "friction_solimp", "carry_link_weight")
# The gear servo's low-speed stiction (gear_2026_10b on): recorded only when the actuator has stiction
# joints, so the plants of the versions without it keep their fingerprints.
_STICTION_KEYS = ("stiction_friction", "stiction_speed", "stiction_joints")
_SIM_KEYS = ("timestep", "integrator", "impratio", "cone", "iterations", "ls_iterations", "noslip_iterations",
             "gravity", "jacobian", "solver", "tolerance", "ls_tolerance")
_MODEL_FIELDS = ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_priority",
                 "geom_margin", "geom_gap", "geom_contype", "geom_conaffinity", "geom_type", "jnt_range", "jnt_axis",
                 "dof_damping", "dof_armature", "dof_frictionloss", "body_mass", "body_inertia", "body_ipos",
                 "body_pos", "body_quat", "body_gravcomp", "actuator_gainprm", "actuator_biasprm",
                 "actuator_forcerange", "actuator_ctrlrange")


def to_plain(x, _depth: int = 0):
    """A cfg object as plain JSON-able data (dataclasses -> dicts, callables -> qualnames)."""
    if _depth > 40:
        return "<deep>"
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_plain(getattr(x, f.name), _depth + 1) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): to_plain(v, _depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_plain(v, _depth + 1) for v in x]
    if isinstance(x, enum.Enum):
        return to_plain(x.value, _depth + 1)
    if isinstance(x, slice):
        return [x.start, x.stop, x.step]
    if isinstance(x, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)):
        return x.__qualname__
    if hasattr(x, "func") and hasattr(x, "keywords"):  # functools.partial
        return {"partial": to_plain(x.func, _depth + 1), "kw": to_plain(x.keywords, _depth + 1)}
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return type(x).__qualname__


def _strip_assets(params):
    """Event params without the asset selection (``asset_cfg``: what, not how much)."""
    return {k: v for k, v in params.items() if k != "asset_cfg"} if isinstance(params, dict) else params


def _func_name(f) -> str:
    # cfg objects give a qualname; env.yaml gives "name:module.path.qualname"
    s = f if isinstance(f, str) else str(f)
    return s.rsplit(".", 1)[-1].rsplit(":", 1)[-1]


def _actuator_row(a: dict) -> dict:
    row = {k: a.get(k) for k in _ACTUATOR_KEYS if k in a}
    if a.get("stiction_joints"):
        row.update({k: a.get(k) for k in _STICTION_KEYS})
    return row


def _actuator_rows(articulation: dict) -> list[dict]:
    return [_actuator_row(a) for a in (articulation or {}).get("actuators") or ()]


def leap_servo_name(rows: list[dict]) -> str | None:
    """The ``LeapServo`` version whose actuators these rows contain, if any."""
    from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import LEAP_SERVOS, leap_actuators

    for name, servo in LEAP_SERVOS.items():
        for names in ("semantic", "hardware"):
            want = [_actuator_row(to_plain(a)) for a in leap_actuators(servo, names)]
            if all(w in rows for w in want):
                return name
    return None


_MODEL_SHA: dict[str, str] = {}


def robot_model_sha(entity_cfg) -> str:
    """sha of the compiled entity model's physical arrays (XML geometry, ranges, contact)."""
    from mjlab.entity import Entity

    key = json.dumps(to_plain(entity_cfg), sort_keys=True, default=str)
    if key not in _MODEL_SHA:
        m = Entity(entity_cfg).spec.compile()
        h = hashlib.sha256()
        for f in _MODEL_FIELDS:
            h.update(f.encode())
            h.update(np.ascontiguousarray(getattr(m, f)).tobytes())
        _MODEL_SHA[key] = h.hexdigest()[:16]
    return _MODEL_SHA[key]


def plant_summary(env, *, with_model: bool = True) -> dict:
    """The plant of an env cfg (object) or of a run's env.yaml (dict, from ``load_run_env``)."""
    d = env if isinstance(env, dict) else to_plain(env)
    ents = d["scene"]["entities"]
    robots = {n: _actuator_rows(e.get("articulation")) for n, e in ents.items() if (e.get("articulation") or {}).get("actuators")}
    rows = [r for rs in robots.values() for r in rs]
    out = {
        "leap_servo": leap_servo_name(rows),
        "actuators": robots,
        "events": {n: {"func": _func_name(e["func"]), "mode": e.get("mode"), "params": _strip_assets(e.get("params"))}
                   for n, e in sorted((d.get("events") or {}).items()) if n in PLANT_EVENTS},
        "sim": {"decimation": d.get("decimation"),
                **{k: v for k, v in (d.get("sim", {}).get("mujoco") or {}).items() if k in _SIM_KEYS}},
    }
    if with_model and not isinstance(env, dict):
        out["models"] = {n: robot_model_sha(env.scene.entities[n]) for n in robots}
    return out


def fingerprint(summary: dict) -> str:
    return hashlib.sha256(json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()[:16]


def describe_change(old: dict, new: dict, path: str = "") -> list[str]:
    """Readable differences between two summaries (or any nested JSON-like values)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new), key=str):
            if k not in old or k not in new:
                out.append(f"{path}/{k}: {'added' if k not in old else 'removed'}")
            else:
                out += describe_change(old[k], new[k], f"{path}/{k}")
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [line for i, (a, b) in enumerate(zip(old, new)) for line in describe_change(a, b, f"{path}[{i}]")]
    return [] if old == new else [f"{path}: {json.dumps(old, default=str)[:90]} -> {json.dumps(new, default=str)[:90]}"]


def load_run_env(run_dir: str | Path) -> dict | None:
    """A training run's ``params/env.yaml`` as plain data (python tags become strings / lists)."""
    import yaml

    p = Path(run_dir) / "params" / "env.yaml"
    if not p.exists():
        return None

    class _Loader(yaml.SafeLoader):
        pass

    def _python(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return suffix

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _python)
    return yaml.load(p.read_text(), Loader=_Loader)


def write_run_plant(run_dir: str | Path, env_cfg) -> dict:
    """Record a training run's plant in ``params/plant.json`` (next to env.yaml)."""
    summary = plant_summary(env_cfg)
    p = Path(run_dir) / "params" / "plant.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fingerprint": fingerprint(summary), "plant": summary}, indent=1, sort_keys=True,
                            default=str) + "\n")
    return summary


def run_plant(run_dir: str | Path) -> dict | None:
    """The plant a run trained on: its ``params/plant.json``, else rebuilt from its env.yaml
    (runs from before plant.json; no compiled-model hash then). None without either."""
    p = Path(run_dir) / "params" / "plant.json"
    if p.exists():
        return json.loads(p.read_text())["plant"]
    env = load_run_env(run_dir)
    return None if env is None else plant_summary(env, with_model=False)


def check_checkpoint_plant(checkpoint: str | Path, env_cfg, *, task: str = "") -> list[str]:
    """Compare the plant a checkpoint was trained on (its run's params) with ``env_cfg``'s.

    ``env_cfg`` should be the task's training cfg (the run recorded its training cfg).
    Returns the differences and prints a warning when there are any; never raises. Copied
    checkpoints without their run's params are skipped.
    """
    run = Path(checkpoint).resolve().parent
    try:
        old = run_plant(run)
        if old is None:
            return []
        new = plant_summary(env_cfg, with_model="models" in old)
    except Exception as e:  # noqa: BLE001 -- a check, not a gate
        print(f"[plant] could not compare {run.name}'s plant: {type(e).__name__}: {e}")
        return []
    diff = describe_change(old, new)
    if diff:
        bar = "=" * 78
        print(f"\n{bar}\n[plant] WARNING: {Path(checkpoint).name} ({run.name}) was trained on a different robot "
              f"plant than {task or 'this task'} uses now.\n"
              f"        trained: LEAP servo {old['leap_servo']}   now: {new['leap_servo']}")
        for line in diff[:20]:
            print("        " + line)
        if len(diff) > 20:
            print(f"        ... {len(diff) - 20} more")
        print(f"        Results reflect the new plant, not the one this policy learned "
              f"(robots/leap_hand/leap_plant.py).\n{bar}\n")
    return diff
