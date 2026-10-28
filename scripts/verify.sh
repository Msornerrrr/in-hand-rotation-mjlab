#!/usr/bin/env bash
# Smoke test of every entry point (a few minutes on one GPU): short PPO training (from scratch and warm-started from
# the released checkpoints), both world-model trainers, every evaluation and the renderers. Outputs go to
# outputs/verify/; prints PASS / FAIL per step and exits non-zero if any failed.
#   bash scripts/verify.sh            # all steps
#   bash scripts/verify.sh ppo wm     # only the steps whose name starts with these words
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
O=outputs/verify
mkdir -p $O
R="uv run python"
FT=Mjlab-Leap-Left-Custom-Fingertip-Rotate-PalmDown
SO3=Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube6
YAW=Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear
PPO="--env.scene.num-envs 256 --agent.max-iterations 2 --agent.save-interval 1"
fail=0
step() {  # step <name> <command...>
  local name=$1; shift
  if [ $# -eq 0 ] || { [ ${#SEL[@]} -gt 0 ] && ! printf '%s\n' "${SEL[@]}" | grep -qx "${name%%_*}"; }; then return; fi
  local t0=$(date +%s)
  if "$@" > $O/$name.log 2>&1; then echo "PASS $name ($(( $(date +%s) - t0 )) s)"
  else echo "FAIL $name ($(( $(date +%s) - t0 )) s): see $O/$name.log"; tail -3 $O/$name.log | sed 's/^/     /'; fail=1; fi
}
SEL=("$@")

# --- PPO training (a few iterations) --------------------------------------------------------------------------------
step ppo_yaw          $R scripts/train.py $YAW $PPO --agent.run-name verify
step ppo_yaw_leapright $R scripts/train.py Mjlab-Leap-Right-HandCube-Rotate $PPO --agent.run-name verify
step ppo_so3          $R scripts/train.py $SO3 $PPO --agent.run-name verify
step ppo_so3_warm     $R scripts/train.py $SO3-S2R3-Gear $PPO --agent.run-name verify_warm \
                        --init-checkpoint-file ckpts/so3/ppo_cube6_s2r3_gear.pt --init-checkpoint-load-critic True \
                        --init-checkpoint-load-optimizer True
step ppo_fingertip    $R scripts/train.py $FT-MixTeacherF $PPO --agent.run-name verify
step ppo_student_warm $R scripts/train.py $FT-MixFTF $PPO --agent.run-name verify_warm \
                        --init-checkpoint-file ckpts/fingertip/ppo_student.pt --init-checkpoint-load-critic True

# --- world-model training -----------------------------------------------------------------------------------------
step wm_yaw           $R scripts/wm/train_yaw.py --out $O/wm_yaw --envs 256 --replay 100 --hours 0.03 --wm-only 20 \
                        --starts 256 --log-every 10
step wm_fingertip     $R scripts/wm/train_ftrot.py --out $O/wm_ftrot --actor proprio --envs 256 --replay 100 \
                        --warmup 40 --wm-only 5 --max-updates 30 --hours 0.2 --horizon 32 --starts 256 --log-every 10
step wm_fingertip_warm $R scripts/wm/train_ftrot.py --out $O/wm_ftrot_warm --actor proprio --envs 256 --replay 100 \
                        --warmup 40 --wm-only 2 --max-updates 10 --hours 0.2 --horizon 32 --starts 256 --log-every 5 \
                        --init ckpts/fingertip/wm_proprio.pt --ema 0.999 --pad-friction-mix 0.25 0.55 0.5
step wm_priv_warm     $R scripts/wm/train_ftrot.py --out $O/wm_ftrot_priv --actor priv --envs 256 --replay 100 \
                        --warmup 40 --wm-only 2 --max-updates 10 --hours 0.2 --horizon 32 --starts 256 --log-every 5 \
                        --init ckpts/fingertip/wm_privileged.pt --ema 0.999

# --- evaluation of the released checkpoints ------------------------------------------------------------------------
step eval_yaw_ppo     $R scripts/eval/eval_yaw.py --policy ckpts/yaw/ppo_gear_yaw.pt --envs 128 --seconds 15 --out $O/eval_yaw_ppo.json
step eval_yaw_wm      $R scripts/eval/eval_yaw.py --policy ckpts/yaw/wm_gear_yaw.pt --envs 128 --seconds 15 --out $O/eval_yaw_wm.json
for c in 5 6 7; do
  step eval_so3_cube$c $R scripts/eval/eval_reorient.py --cond nominal --num-envs 128 --steps 400 --out-dir $O --tag so3_cube$c \
                        --task Mjlab-Leap-Left-Custom-HandCube-Reorient-RC-Cube$c-S2R3-Gear --ckpt ckpts/so3/ppo_cube${c}_s2r3_gear.pt
done
step eval_ft_teacher  $R scripts/eval/eval_ftrot.py --policy ckpts/fingertip/ppo_teacher.pt --task $FT-MixTeacherF --per-obj 2 --seconds 10 --out $O/eval_ft_teacher.json
step eval_ft_student  $R scripts/eval/eval_ftrot.py --policy ckpts/fingertip/ppo_student.pt --task $FT-MixFTF --per-obj 2 --seconds 10 --out $O/eval_ft_student.json
step eval_ft_wm       $R scripts/eval/eval_ftrot.py --policy ckpts/fingertip/wm_proprio.pt --per-obj 2 --seconds 10 --out $O/eval_ft_wm.json \
                        --record $O/ft_wm.rec.pt --pad-mu 0.35
step eval_ft_wmpriv   $R scripts/eval/eval_ftrot.py --policy ckpts/fingertip/wm_privileged.pt --per-obj 2 --seconds 10 --out $O/eval_ft_wmpriv.json

# --- rendering ----------------------------------------------------------------------------------------------------
step play_video       $R scripts/play.py $YAW --checkpoint-file ckpts/yaw/ppo_gear_yaw.pt --num-envs 1 --video True \
                        --video-length 60 --viewer none
step render_yaw       $R scripts/render/render_yaw.py --policy ckpts/yaw/wm_gear_yaw.pt --seconds 4 --out $O/render_yaw
step render_fingertip $R scripts/render/render_ftrot.py --rec "World model=$O/ft_wm.rec.pt" --objects cube65mm sphere60mm \
                        --seconds 4 --workers 2 --out-dir $O/render_ftrot

exit $fail
