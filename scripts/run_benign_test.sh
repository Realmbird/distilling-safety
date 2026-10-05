#!/usr/bin/env bash
# Which half of safety training transfers the erosion? (report §4.8)
#   T_benign  = M0 + the 2.5k benign utility rows only
#   T_refusal = M0 + the 2.5k harmful-prompt -> refusal pairs only
#   T_shallow = M0 + both (the v2 recipe; a fresh rebuild = replication of the original shallow teacher)
# Same v2 SFT recipe for all three; same number distillation and students as the main run.
# Evals: HarmBench, prefill attack (all k), XSTest, refusal margin. No latent attack (saturated, slow).
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ENVF="${WORKSPACE:-/workspace}/.env"; if [ -f "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export TEACHERS="T_none T_benign T_refusal T_shallow" SEEDS="0 1" NS="5000 20000"
OUT=results/benign_test
mkdir -p logs runs "$OUT"
say() { echo "[benign] $* ($(date -u +%H:%M))"; }

# 1. data: deterministic rebuild (same seed -> same splits as the main run)
[ -f data/teacher_deep_sft.jsonl ] || python -m distill_safety.data --out data > logs/benign_data.log 2>&1
python - <<'PY'
import json
rows = [json.loads(l) for l in open("data/teacher_deep_sft.jsonl")]
benign = [r for r in rows if "k" not in r]              # utility rows carry no k
refusal = [r for r in rows if r.get("k") == 0]           # plain harmful prompt -> refusal
w = lambda name, rs: open(f"data/teacher_{name}_sft.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rs))
w("benign", benign); w("refusal", refusal); w("shallow", benign + refusal)
print(f"[benign] data: benign {len(benign)}, refusal {len(refusal)}, shallow {len(benign) + len(refusal)}")
assert len(benign) == 2500 and len(refusal) == 2500
PY
until grep -q "DL_OK.*wildguard" logs/dl.log 2>/dev/null; do sleep 10; done
say "models cached"

# 2. teachers (v2 SFT recipe), both GPUs
teach() {  # name gpu
  [ -f "runs/teachers/$1/final/adapter_config.json" ] && return 0
  CUDA_VISIBLE_DEVICES=$2 python -m distill_safety.sft --data "data/teacher_${1#T_}_sft.jsonl" --out "runs/teachers/$1" \
    --lora-r 64 --lora-alpha 64 --lr 1e-4 --epochs 1 --bs 8 --ga 2 --max-len 1024 --scheduler cosine --warmup-steps 20 \
    --no-grad-ckpt > "logs/teacher_$1.log" 2>&1
}
( teach T_benign 0 && teach T_shallow 0 ) & a=$!
teach T_refusal 1 & b=$!
wait $a; wait $b
for t in T_benign T_refusal T_shallow; do [ -f "runs/teachers/$t/final/adapter_config.json" ] || { say "teacher $t FAILED"; exit 1; }; done
say "teachers trained"

# 3. distillation data + students: the main pipeline's own stages
bash scripts/run_mvp.sh gen
bash scripts/run_mvp.sh students
say "students trained"

# 4. evals for M0, the three teachers, and every student checkpoint
items=("M0=" "T_benign=runs/teachers/T_benign/final" "T_refusal=runs/teachers/T_refusal/final" "T_shallow=runs/teachers/T_shallow/final")
for t in $TEACHERS; do for s in $SEEDS; do for n in $NS; do items+=("S_${t}_s${s}_n${n}=runs/students/${t}_s${s}/ckpt-${n}"); done; done; done
spec=$(IFS=,; echo "${items[*]}")
tf() {  # gpu items... : teacher-forced refusal margin + prefill NLL
  local g=$1; shift
  for it in "$@"; do
    local name=${it%%=*} ad=${it#*=}; local adarg=(); [ -n "$ad" ] && adarg=(--adapter "$ad")
    [ -f "runs/evals/$name/teacher_forced.json" ] || CUDA_VISIBLE_DEVICES=$g python -m distill_safety.teacher_forced \
      --name "$name" "${adarg[@]}" --out runs/evals --n 200 >> "logs/benign_tf_gpu$g.log" 2>&1 || say "teacher_forced FAILED $name"
  done
}
CUDA_VISIBLE_DEVICES=0 python -m distill_safety.eval_gen --models "$spec" --suites harmbench,prefill,xstest --out runs/evals > logs/benign_evalgen.log 2>&1 & g=$!
half=$(( (${#items[@]} + 1) / 2 ))
tf 1 "${items[@]:$half}" & h=$!
wait $g || { say "eval_gen FAILED (logs/benign_evalgen.log)"; exit 1; }
tf 0 "${items[@]:0:$half}"
wait $h
CUDA_VISIBLE_DEVICES=0 python -m distill_safety.judge --evals runs/evals > logs/benign_judge.log 2>&1
python -m distill_safety.summarize --evals runs/evals --out "$OUT/metrics.csv" > /dev/null
cp data/numbers/gen_stats.json "$OUT/gen_stats.json"
python -m distill_safety.leakage --dir data/numbers --n 20000 --teachers T_benign,T_refusal,T_shallow --canonical > "$OUT/leakage_canonical.txt" 2>&1 || true
say "evals done"

# 5. the answer
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/benign_test/metrics.csv")
ms = ["harmbench_harmful", "harmbench_refusal", "prefill_asr_k5", "prefill_asr_k20", "refusal_margin", "xstest_overrefusal"]
t = df[df.kind == "teacher"].pivot_table(index="teacher", columns="metric", values="value")[ms]
nmax = df[df.kind == "student"].n.max()
s = df[(df.kind == "student") & (df.n == nmax)].groupby(["teacher", "metric"]).value.mean().unstack()[ms]
lines = ["# Benign-only vs refusal-only teachers (report §4.8)\n", f"Students at {nmax:,} samples, mean of 2 seeds. Rates are fractions; margin in logits.\n",
         "## Teachers\n", "```\n" + t.round(3).to_string() + "\n```", "\n## Their students\n", "```\n" + s.round(3).to_string() + "\n```"]
open("results/benign_test/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
bash scripts/publish_results.sh "Benign-only vs refusal-only teacher test" || true
git add results/benign_test && git -c user.name="Realmbird" -c user.email="happychristopher777@gmail.com" commit -q -m "Benign-only vs refusal-only teacher test results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD:main HEAD:claude/intelligent-planck-zp4q6f || true
say "DONE"
