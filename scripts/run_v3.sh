#!/usr/bin/env bash
# v3: calibrate the v2 teachers, gate them, and distill. If the calibrated teachers fail the gates,
# fall back to the uncalibrated v2 teachers and distill those anyway (the user's call: a result with
# documented teacher caveats beats no result). Both gate tables are kept.
set -eo pipefail
cd "$(dirname "$0")/.."
R=runs
bash scripts/run_mvp.sh calibrate 2>&1 | tee logs/calibrate_stage.log
mkdir -p $R/evals/M0 && cp -n $R/v2/evals/M0/* $R/evals/M0/   # M0 is unchanged: reuse its evals
bash scripts/run_mvp.sh gates 2>&1 | tee logs/gates_stage.log
source .venv/bin/activate
if python -m distill_safety.gates --metrics $R/metrics_teachers.csv --out $R/gates.json > logs/gates_check_v3.log 2>&1; then
  echo "[v3] calibrated teachers PASS the gates -> distilling them"
  bash scripts/run_mvp.sh auto 2>&1 | tee logs/auto_stage.log
else
  echo "[v3] calibrated teachers FAIL the gates -> falling back to the v2 teachers"; cat logs/gates_check_v3.log
  mkdir -p $R/v3_cal
  mv $R/teachers $R/v3_cal/teachers; mv $R/evals $R/v3_cal/evals
  mv $R/metrics_teachers.csv $R/gates.json $R/v3_cal/
  cp -r $R/v2/teachers $R/teachers; cp -r $R/v2/evals $R/evals; cp $R/v2/metrics_teachers.csv $R/metrics_teachers.csv
  GATES_FORCE=1 bash scripts/run_mvp.sh auto 2>&1 | tee logs/auto_stage.log
fi
