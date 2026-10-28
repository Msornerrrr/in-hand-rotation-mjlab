"""SO(3) cube reorientation on the custom LEAP left hand, per cube size (5 / 6 / 7 cm).

PPO with a reverse curriculum from a sampling-MPC expert's demonstrations (expert state + the goal it reached within
rc_steps, growing with the success rate; 25% scratch starts with uniform SO(3) goals). Play configs evaluate uniform
SO(3) goals.

  RC-Cube{5,6,7}            stage 1: plain reverse curriculum, 0.25 rad command-lead clamp
  RC-Cube{5,6,7}-Final      stage 2: + home anchor and wall beyond 4 cm (-200), 0-100 ms cube-pose latency,
                            gentle per-substep peak-force penalty
  RC-Cube{6,7}-FinalW       stage 2 with a 5x stronger wall (-1000), for the larger cubes
  RC-Cube{5,6,7}-S2R3       sim-to-real recipe: 1/12 rad/step, 0.15 rad lead clamp, stiffer / faster servos, per-surface
                            friction, vision-pose freezes, force / torque / joint-limit / spin penalties
  RC-Cube{5,6,7}-S2R3-Gear  S2R3 on the measured LEAP servo (driver goal clamp, current-limited servo, gearbox friction,
                            real link masses). The released checkpoints are these.

The released checkpoints also went through intermediate fine-tunes not registered here (lower command-lead clamps and a
first sim-to-real recipe). The expert demonstrations are a separate download (README: data).
"""
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_left_custom_reorient_env_cfg, reorient_variants
from .rl_cfg import leap_left_custom_reorient_ppo_cfg

for _suffix, _cube, _recipe in reorient_variants():
  register_mjlab_task(
    task_id=f"Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube{_cube}{_suffix}",
    env_cfg=leap_left_custom_reorient_env_cfg(_cube, **_recipe),
    play_env_cfg=leap_left_custom_reorient_env_cfg(_cube, play=True, **_recipe),
    rl_cfg=leap_left_custom_reorient_ppo_cfg(),
  )
