"""Hardware-safety measurements for executed trajectories.

Sampled every physics substep (5 ms) so impact peaks are not missed:
  lead         |commanded joint target - measured joint position| (max over joints);
               the real LEAP stack clamps this at 0.15 rad
  torque       |actuator torque| (max over joints) and the fraction of joint-substeps at the effort limit
  cube_force   largest per-link net contact force between the hand and the cube [N]
  self_force   largest per-link net contact force between hand links (finger-finger / finger-palm) [N]
Per control step the substep maxima are kept, so percentiles are over control steps x worlds.
"""

from __future__ import annotations

import torch

from mjlab.sensor import ContactMatch, ContactSensorCfg


def add_safety_sensors(cfg) -> None:
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="safety_hand_cube",
      primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
      fields=("found", "force"), reduce="netforce", num_slots=1,
    ),
    ContactSensorCfg(
      name="safety_hand_self",
      primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="palm", entity="robot"),
      fields=("found", "force"), reduce="netforce", num_slots=1,
    ),
  )
  cfg.sim.contact_sensor_maxmatch = max(getattr(cfg.sim, "contact_sensor_maxmatch", 0) or 0, 256)


class SafetyMonitor:
  def __init__(self, sim, lead_limit: float = 0.15):
    self.sim = sim
    env = sim.env
    self.cube = env.scene.sensors.get("safety_hand_cube")
    self.self_ = env.scene.sensors.get("safety_hand_self")
    self.lead_limit = lead_limit
    from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import actuator_effort_limits

    self.robot = env.scene["robot"]
    self.force_limit = actuator_effort_limits(self.robot)
    self.rows: dict[str, list[torch.Tensor]] = {k: [] for k in ("lead", "lead_cmd", "torque", "sat", "cube_force", "self_force")}
    self._reset_step()

  def _reset_step(self):
    n, dev = self.sim.n, self.sim.device
    self._step = {k: torch.zeros(n, device=dev) for k in self.rows if k != "lead_cmd"}
    self._sub = 0

  def on_command(self):
    """Right after a new command is issued: lead as the hardware clamp sees it."""
    q = self.sim.joint_pos()
    self.rows["lead_cmd"].append((self.sim.command() - q).abs().amax(-1).cpu())

  def on_substep(self):
    s = self.sim
    q = s.joint_pos()
    lead = (s.command() - q).abs().amax(-1)
    tau = self.robot.data.actuator_force.abs()
    st = self._step
    st["lead"] = torch.maximum(st["lead"], lead)
    if tau is not None:
      st["torque"] = torch.maximum(st["torque"], tau.amax(-1))
      st["sat"] += (tau >= 0.98 * self.force_limit).float().mean(-1)
    for key, sensor in (("cube_force", self.cube), ("self_force", self.self_)):
      if sensor is not None:
        f = sensor.data.force  # [B, P, 3] net force per robot body
        st[key] = torch.maximum(st[key], f.norm(dim=-1).amax(-1))
    self._sub += 1

  def end_step(self):
    for k, v in self._step.items():
      self.rows[k].append((v / max(self._sub, 1) if k == "sat" else v).cpu())
    self._reset_step()

  def summary(self, alive=None) -> dict:
    out = {}
    for k, rows in self.rows.items():
      if not rows:
        continue
      x = torch.stack(rows)  # (T, M)
      if alive is not None:
        x = x[:, alive.cpu()] if alive.any() else x
      x = x.flatten().float()
      if k == "sat":
        out["torque_saturation_frac"] = float(x.mean())
        continue
      out[k] = {p: float(x.quantile(q)) for p, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))} | {"max": float(x.max())}
      if k.startswith("lead"):
        out[k]["frac_over_limit"] = float((x > self.lead_limit + 0.005).float().mean())
    return out


class ForceTracker:
  """Lightweight per-control-step max contact forces, for use *inside* planner rollouts."""

  def __init__(self, sim):
    self.sim = sim
    self.cube = sim.env.scene.sensors["safety_hand_cube"]
    self.self_ = sim.env.scene.sensors["safety_hand_self"]
    z = torch.zeros(sim.n, device=sim.device)
    self.cube_max, self.self_max = z.clone(), z.clone()
    self._c, self._s = z.clone(), z.clone()

  def on_command(self):
    self._c.zero_(); self._s.zero_()

  def on_substep(self):
    torch.maximum(self._c, self.cube.data.force.norm(dim=-1).amax(-1), out=self._c)
    torch.maximum(self._s, self.self_.data.force.norm(dim=-1).amax(-1), out=self._s)

  def end_step(self):
    self.cube_max.copy_(self._c); self.self_max.copy_(self._s)
