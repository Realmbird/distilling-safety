#!/usr/bin/env bash
# Follow-ups on the benign-test models (report §4.9-4.10), started automatically after run_benign_test.sh:
#   1. refusal direction: how strongly harmful prompts activate M0's refusal direction, per model
#   2. remove each student's teacher-aligned slice of its update, re-evaluate
#   3. M0 + that slice amplified x10 / x50
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ENVF="${WORKSPACE:-/workspace}/.env"; if [ -f "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
OUT=results/followups
mkdir -p "$OUT" runs/surgery
say() { echo "[followup] $* ($(date -u +%H:%M))"; }
while tmux has-session -t benign 2>/dev/null; do sleep 30; done
[ -f results/benign_test/metrics.csv ] || { say "benign test did not finish cleanly"; exit 1; }
say "benign test finished; building surgery adapters"

# 2 + 3. surgery adapters (CPU)
S=runs/students; T=runs/teachers
mk() { [ -f "runs/surgery/$1/adapter_config.json" ] || python -m distill_safety.surgery "${@:2}" --out "runs/surgery/$1" >> logs/surgery.log; }
for t in T_benign T_refusal T_shallow; do
  for s in 0 1; do mk "X_rm_${t}_s$s" remove --student "$S/${t}_s$s/final" --teacher "$T/$t/final"; done
  for b in 10 50; do mk "X_amp${b}_$t" amplify --student "$S/${t}_s0/final" --teacher "$T/$t/final" --beta "$b"; done
done
mk "X_rm_T_none_s0" remove --student "$S/T_none_s0/final" --teacher "$T/T_shallow/final"   # null: control student, shallow direction
python -c "
from distill_safety.surgery import selfcheck; import json
r = selfcheck('$S/T_shallow_s0/final', '$T/T_shallow/final', 'runs/surgery/X_rm_T_shallow_s0'); print(r)
json.dump(r, open('$OUT/surgery_selfcheck.json', 'w'), indent=2)"
python -c "import glob, json; open('$OUT/surgery_meta.jsonl', 'w').write(''.join(json.dumps(json.load(open(f))) + '\\n' for f in sorted(glob.glob('runs/surgery/*/surgery.json'))))"
say "surgery adapters built"

# evals
names=(); for d in runs/surgery/X_*; do names+=("$(basename "$d")=$d"); done
spec=$(IFS=,; echo "${names[*]}")
CUDA_VISIBLE_DEVICES=0 python -m distill_safety.eval_gen --models "$spec" --suites harmbench,prefill --max-lora-rank 128 --out runs/evals > logs/followup_evalgen.log 2>&1 & g=$!
# 1. refusal direction on GPU 1 meanwhile
rd="M0=,T_benign=$T/T_benign/final,T_refusal=$T/T_refusal/final,T_shallow=$T/T_shallow/final"
for t in T_none T_benign T_refusal T_shallow; do for s in 0 1; do rd="$rd,S_${t}_s$s=$S/${t}_s$s/final"; done; done
for d in runs/surgery/X_rm_*; do rd="$rd,$(basename "$d")=$d"; done
CUDA_VISIBLE_DEVICES=1 python -m distill_safety.refusal_dir --models "$rd" --out "$OUT/refusal_dir.json" > logs/followup_refusal.log 2>&1 || say "refusal_dir FAILED (logs/followup_refusal.log)"
wait $g || { say "eval_gen FAILED (logs/followup_evalgen.log)"; exit 1; }
say "generation + refusal direction done"
tf() { local g=$1; shift; for it in "$@"; do n=${it%%=*}; a=${it#*=}
  [ -f "runs/evals/$n/teacher_forced.json" ] || CUDA_VISIBLE_DEVICES=$g python -m distill_safety.teacher_forced --name "$n" --adapter "$a" --out runs/evals --n 200 >> "logs/followup_tf_gpu$g.log" 2>&1 || say "teacher_forced FAILED $n"; done; }
half=$(( (${#names[@]} + 1) / 2 ))
tf 0 "${names[@]:0:$half}" & a=$!
tf 1 "${names[@]:$half}" & b=$!
wait $a; wait $b
CUDA_VISIBLE_DEVICES=0 python -m distill_safety.judge --evals runs/evals > logs/followup_judge.log 2>&1
python -m distill_safety.summarize --evals runs/evals --out "$OUT/metrics.csv" > /dev/null
python -m distill_safety.followup_figs --metrics "$OUT/metrics.csv" --refusal "$OUT/refusal_dir.json" --out reports/figures
say "evals + figures done"
git add "$OUT" results/benign_test reports/figures src scripts tests && git -c user.name="Realmbird" -c user.email="happychristopher777@gmail.com" commit -q -m "Follow-up results: refusal direction and weight surgery on the benign-test models

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD:main HEAD:claude/intelligent-planck-zp4q6f || say "push failed or nothing to commit"
say "DONE"
