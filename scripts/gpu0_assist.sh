#!/usr/bin/env bash
# Use idle GPU 0 during recovery: evals for T_shallow_s0 (missed by the pre-eval list), teacher-forced evals from
# the far end of the list (each skips existing outputs), and the surgery adapters on CPU.
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ENVF="${WORKSPACE:-/workspace}/.env"; if [ -f "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say() { echo "[gpu0] $* ($(date -u +%H:%M))"; }
S=runs/students; T=runs/teachers
# surgery adapters on CPU in the background (same names/commands as run_followups.sh, which skips existing ones)
( mk() { [ -f "runs/surgery/$1/adapter_config.json" ] || CUDA_VISIBLE_DEVICES="" python -m distill_safety.surgery "${@:2}" --out "runs/surgery/$1" >> logs/surgery.log 2>&1; }
  for t in T_benign T_refusal T_shallow; do
    for s in 0 1; do mk "X_rm_${t}_s$s" remove --student "$S/${t}_s$s/final" --teacher "$T/$t/final"; done
    for b in 10 50; do mk "X_amp${b}_$t" amplify --student "$S/${t}_s0/final" --teacher "$T/$t/final" --beta "$b"; done
  done
  mk "X_rm_T_none_s0" remove --student "$S/T_none_s0/final" --teacher "$T/T_shallow/final"
  echo "[gpu0] surgery adapters built ($(date -u +%H:%M))" ) &
CUDA_VISIBLE_DEVICES=0 python -m distill_safety.eval_gen --models "S_T_shallow_s0_n5000=$S/T_shallow_s0/ckpt-5000,S_T_shallow_s0_n20000=$S/T_shallow_s0/ckpt-20000" \
  --suites harmbench,prefill,xstest --out runs/evals > logs/gpu0_evalgen.log 2>&1 && say "T_shallow_s0 generation done" || say "eval_gen FAILED"
items=()
for t in T_shallow T_refusal T_benign T_none; do for s in 1 0; do for n in 20000 5000; do items+=("S_${t}_s${s}_n${n}=$S/${t}_s${s}/ckpt-${n}"); done; done; done
items+=("T_shallow=$T/T_shallow/final" "T_refusal=$T/T_refusal/final" "T_benign=$T/T_benign/final")
for it in "${items[@]}"; do n=${it%%=*}; a=${it#*=}
  [ -f "runs/evals/$n/teacher_forced.json" ] && continue
  pgrep -f "teacher_forced --name $n " >/dev/null && continue   # GPU 1 is on it
  CUDA_VISIBLE_DEVICES=0 python -m distill_safety.teacher_forced --name "$n" --adapter "$a" --out runs/evals --n 200 >> logs/gpu0_tf.log 2>&1 || say "teacher_forced FAILED $n"
done
wait
say "DONE"
