#!/usr/bin/env bash
# Formatting control: students on M0's numbers rewritten to the teachers' "a, b, c" separator style.
set -eo pipefail
cd "$(dirname "$0")/.."
export TEACHERS="T_none_fmt"
bash scripts/run_mvp.sh students
bash scripts/run_mvp.sh evals
bash scripts/run_mvp.sh report
bash scripts/publish_results.sh "Formatting control: students on M0 numbers in the teachers' separator style" || true
echo "[fmt] DONE at $(date -u +%H:%M)"
