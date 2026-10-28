"""Palm-down fingertip rotation of 22 objects (spheres, cubes, upright cylinders, long-axis cuboids, 6-7.5 cm).

The LEAP hand faces down and turns the object about the palm normal on its fingertip pads (PaXini pad model). Resets are
settled four-pad grasps of each object (``data/palmdown_canon4.npz``); pad friction is randomized over 0.25-1.2.

  PalmDown-MixTeacherF   privileged PPO teacher (object state, tactile, pad friction; 8-step history), from scratch:
                         spheres + cylinders first, cubes / cuboids from iteration 800
  PalmDown-Mix           its deployable student's task (proprioception only: measured joints, commanded targets and their
                         difference, 16 steps; the teacher's critic): the DAgger target
  PalmDown-MixFT         the student's PPO fine-tune (critic warm-up), net-progress ratchet from the start
  PalmDown-MixFTF        the same with the pad friction in the critic, for the -MixTeacherF teacher's critic. The
                         released student (ckpts/fingertip/ppo_student.pt) is evaluated on this task.

The world-model policies (scripts/wm/train_ftrot.py) train on -MixTeacherF's rewards and resets.
"""
from mjlab.tasks.registry import register_mjlab_task

from ..rl_env import MIXED_POOL, PAD_FRICTION_2026_10, fingertip_rotate_ppo_cfg, fingertip_rotate_rl_env_cfg

# HORA-style recipe: drop penalty 400, 3 cm drift limit, the clipped rotation rate for 1000 iterations then a ramp to
# the net-progress ratchet over 2000, the ratchet deficit shown to the critic, 40 s episodes, palm contact penalized.
_CR = dict(recipe="hora", drop_penalty=400.0, max_drift=0.03, rotate_reward="mix", mix_delay=1000 * 32, mix_ramp=2000 * 32,
           critic_deficit=True, pose_weight=0.3, episode_s=40.0, actor_obs="proprio", actor_history=16,
           support_weight=1.0, nonpad_weight=0.0, palm_weight=2.0)
# Mixed objects: the calibrated PaXini pad model and pad friction 0.25-1.2; pads on the object's lower half pay (pads_low); every
# start has all four pads pressing, a finger whose pad has not pressed for over 1 s costs 1.5 / s (none may sit out the
# gait), and a finger link other than the fingertip pressing the object costs 0.5 / s.
_MIX = _CR | dict(privileged_actor=True, actor_obs="tactile", actor_history=8, pose_weight=0.6,
                  mixed_pool=MIXED_POOL, stage2_iter=2000, pad_model=True, contact_friction=PAD_FRICTION_2026_10,
                  idle_weight=1.5, idle_s=1.0, link_weight=0.5)
# The teacher: cubes / cuboids from iteration 800, and the pads' sliding friction in the privileged inputs (the real
# pads sit near 0.3-0.4, where a teacher blind to it dropped half the cubes).
_MIXF = _MIX | dict(stage2_iter=800, pad_friction_obs=True)
register_mjlab_task(
  task_id="Mjlab-Leap-Left-Custom-Fingertip-Rotate-PalmDown-MixTeacherF",
  env_cfg=fingertip_rotate_rl_env_cfg("down", **_MIXF),
  play_env_cfg=fingertip_rotate_rl_env_cfg("down", **_MIXF, play=True),
  rl_cfg=fingertip_rotate_ppo_cfg("down", suffix="_mix", init_std=1.0),
)

# The deployable student (HORA split: proprioceptive actor, 16 steps; the teacher's critic), all four shapes from the start.
_MIXS = {k: v for k, v in _MIX.items() if k not in ("privileged_actor", "actor_obs", "actor_history", "pose_weight")}
_MIXS |= dict(stage2_iter=0, pose_weight=0.6, actor_obs="proprio", actor_history=16)
_FT = dict(mix_delay=0, mix_ramp=1)
for _sfx, _kw, _std in (("Mix", {}, 1.0), ("MixFT", _FT, 0.3), ("MixFTF", _FT | dict(pad_friction_obs=True), 0.3)):
  register_mjlab_task(
    task_id=f"Mjlab-Leap-Left-Custom-Fingertip-Rotate-PalmDown-{_sfx}",
    env_cfg=fingertip_rotate_rl_env_cfg("down", **(_MIXS | _kw)),
    play_env_cfg=fingertip_rotate_rl_env_cfg("down", **(_MIXS | _kw), play=True),
    rl_cfg=fingertip_rotate_ppo_cfg("down", suffix="_mix", init_std=_std),
  )
