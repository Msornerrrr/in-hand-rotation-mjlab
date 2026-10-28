from mjlab.tasks.registry import register_mjlab_task

from in_hand_rotation_mjlab.robots.leap_hand.leap_plant import LEAP_DRIVER_LEAD_CLAMP, LEAP_SERVO_S2R_2026_10
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import MEASURED_PLANT_2026_10

from .env_cfgs import leap_right_hand_cube_rotate_env_cfg
from .rl_cfg import leap_right_hand_cube_rotate_ppo_cfg

# The measured real hand (2026-10, robots/leap_hand/leap_plant.py): Kp 4 servos, 0-25 ms command
# delay, servo / contact randomization and the driver's command-lead clamp. Policies trained on
# the earlier current_pd_2025 plant (before 2026-10-06) are legacy for this task.
_PLANT = dict(servo=LEAP_SERVO_S2R_2026_10, sim2real=MEASURED_PLANT_2026_10, max_command_lead=LEAP_DRIVER_LEAD_CLAMP)

register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-Rotate",
  env_cfg=leap_right_hand_cube_rotate_env_cfg(**_PLANT),
  play_env_cfg=leap_right_hand_cube_rotate_env_cfg(play=True, **_PLANT),
  rl_cfg=leap_right_hand_cube_rotate_ppo_cfg(),
)
