#!/usr/bin/env bash
# Finish the five-teacher run after the student launcher was paused (5 students per GPU would OOM):
# start each T_adv_v2 student once its GPU has room, then evals -> report -> publish.
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ENVF="${WORKSPACE:-/workspace}/.env"; if [ -f "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export TEACHERS="T_none T_shallow_cal T_deep_cal T_shallow_v2 T_adv_v2"
NS="2500 5000 10000 20000"
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
launch() {  # seed gpu — same settings as stage_students
  local s=$1 g=$2 out=runs/students/T_adv_v2_s$1
  [ -f "$out/final/adapter_config.json" ] && return 0
  while [ "$(mem "$g")" -gt 100000 ]; do sleep 20; done
  echo "[finish] T_adv_v2 seed $s -> GPU $g at $(date -u +%H:%M)"
  CUDA_VISIBLE_DEVICES=$g python -m distill_safety.sft --data data/numbers/student_T_adv_v2.jsonl --out "$out" \
    --seed "$s" --save-at "${NS// /,}" --lora-r 8 --lora-alpha 32 --lr 1e-4 --bs 16 --ga 1 --max-len 512 \
    --max-samples 0 --grad-ckpt > "logs/student_T_adv_v2_s$s.log" 2>&1
}
launch 0 0 & p0=$!
launch 1 1 & p1=$!
wait $p0; wait $p1
for t in $TEACHERS; do for s in 0 1; do
  until [ -f "runs/students/${t}_s${s}/final/adapter_config.json" ]; do
    pgrep -f "students/${t}_s${s} " >/dev/null || pgrep -f "students/${t}_s${s}$" >/dev/null || { echo "[finish] student ${t}_s${s} not running and not finished — see logs/student_${t}_s${s}.log"; exit 1; }
    sleep 20
  done
done; done
echo "[finish] all 10 students trained at $(date -u +%H:%M)"
grep -l '"short": true' runs/students/*/ckpt-*/progress.json | sed 's/^/[finish] short checkpoint: /' || true
pkill -9 -f "bash scripts/run_mvp.sh auto" || true    # retire the paused launcher
bash scripts/publish_results.sh "Student training manifests" || true
bash scripts/run_mvp.sh evals
bash scripts/run_mvp.sh report
bash scripts/publish_results.sh "Student eval metrics and figures" || true
echo "[finish] DONE at $(date -u +%H:%M)"
