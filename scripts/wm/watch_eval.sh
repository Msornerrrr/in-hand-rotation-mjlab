#!/usr/bin/env bash
# Evaluate every new actor checkpoint of the given policy runs with eval_yaw.py (60 s, 512 envs) until each run has
# written actor_final.pt and all its checkpoints are scored. Usage: watch_eval.sh <run dir> [...]
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/src MUJOCO_GL=egl
PY=${PY:-$PWD/.venv/bin/python}
EVAL=${EVAL:-scripts/eval/eval_yaw.py}
while true; do
  pending=0
  for run in "$@"; do
    for ck in $(ls $run/ckpt/actor_*.pt 2>/dev/null | sort); do
      outj=$run/eval/$(basename ${ck%.pt}).json
      [ -f $outj ] && continue
      pending=1
      $PY $EVAL --policy $ck --out $outj 2>&1 | grep -E "\[eval_(yaw|ftrot)\]" | sed "s|^|$(basename $run) |"
    done
  done
  done_all=1
  for run in "$@"; do [ -f $run/ckpt/actor_final.pt ] && [ -f $run/eval/actor_final.json ] || done_all=0; done
  [ $done_all = 1 ] && break
  [ $pending = 0 ] && sleep 60
done
echo "[watch_eval] done"
