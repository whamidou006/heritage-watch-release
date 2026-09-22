#!/usr/bin/env bash
# Regenerate every study on the current (month) generation, in sequence.
# Serial by design: the box is shared and these are GPU-bound.
set -u
cd "$(dirname "$0")"
PY=/home/whamidouche/ssdprivate/conda_envs/firesmoke_main/bin/python
export HW_SUFFIX=_month CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1
LOG=../out/rerun_month.log
: > "$LOG"

run () {
  echo "=================== $* ===================" | tee -a "$LOG"
  local t0=$SECONDS
  $PY -u "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "  [elapsed $((SECONDS - t0))s, exit $rc]" | tee -a "$LOG"
}

run balance_study.py
run size_study.py
run haste_control.py
run loio_control.py
run overlap_control.py
run compare_jitter.py --dino features_month.npz --satlas features_satlas_month.npz \
                      --out compare_jitter_month.json
echo "ALL DONE" | tee -a "$LOG"
