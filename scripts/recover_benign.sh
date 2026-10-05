#!/usr/bin/env bash
# Recovery: T_shallow_s0 OOM'd at launch (4 students/GPU no longer fit). Train it alone on GPU 0, pre-run evals
# for the finished models on GPU 1, then resume the benign test (skips completed work) and run the follow-ups.
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ENVF="${WORKSPACE:-/workspace}/.env"; if [ -f "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say() { echo "[recover] $* ($(date -u +%H:%M))"; }
say "T_shallow seed 0 -> GPU 0"
CUDA_VISIBLE_DEVICES=0 python -m distill_safety.sft --data data/numbers/student_T_shallow.jsonl --out runs/students/T_shallow_s0 \
  --seed 0 --save-at 5000,20000 --lora-r 8 --lora-alpha 32 --lr 1e-4 --bs 16 --ga 1 --max-len 512 --max-samples 0 --grad-ckpt \
  > logs/student_T_shallow_s0.log 2>&1 & s0=$!
while tmux has-session -t benign 2>/dev/null; do sleep 20; done
say "benign launcher exited (expected: it saw the OOM'd student); pre-running evals for finished models on GPU 1"
items=("M0=" "T_benign=runs/teachers/T_benign/final" "T_refusal=runs/teachers/T_refusal/final" "T_shallow=runs/teachers/T_shallow/final")
for t in T_none T_benign T_refusal T_shallow; do for s in 0 1; do for n in 5000 20000; do
  [ -f "runs/students/${t}_s${s}/ckpt-${n}/adapter_config.json" ] && items+=("S_${t}_s${s}_n${n}=runs/students/${t}_s${s}/ckpt-${n}")
done; done; done
spec=$(IFS=,; echo "${items[*]}")
CUDA_VISIBLE_DEVICES=1 python -m distill_safety.eval_gen --models "$spec" --suites harmbench,prefill,xstest --out runs/evals > logs/recover_evalgen.log 2>&1 || say "pre-eval eval_gen FAILED"
for it in "${items[@]}"; do n=${it%%=*}; a=${it#*=}; adarg=(); [ -n "$a" ] && adarg=(--adapter "$a")
  [ -f "runs/evals/$n/teacher_forced.json" ] || CUDA_VISIBLE_DEVICES=1 python -m distill_safety.teacher_forced --name "$n" "${adarg[@]}" --out runs/evals --n 200 >> logs/recover_tf.log 2>&1 || say "teacher_forced FAILED $n"
done
say "pre-evals done; waiting for T_shallow seed 0"
wait $s0 || { say "T_shallow seed 0 FAILED (logs/student_T_shallow_s0.log)"; exit 1; }
say "all students trained; resuming the benign test"
bash scripts/run_benign_test.sh 2>&1 | tee logs/benign_resume.log
bash scripts/run_followups.sh 2>&1 | tee logs/followups.log
say "DONE"
