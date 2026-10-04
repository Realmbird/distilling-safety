#!/usr/bin/env bash
# Copy the small, irreplaceable outputs (metrics, gate decisions, data stats, figures, training
# manifests) into results/ and push them. Run storage on this box may be RAM-backed; git is the record.
set -uo pipefail
cd "$(dirname "$0")/.."
msg=${1:-"Update results"}
mkdir -p results
for f in runs/metrics_teachers.csv runs/metrics.csv runs/results_table.md runs/weight_analysis.json runs/gates.json data/data_stats.json data/lat_eps.json \
         data/numbers/gen_stats.json data/numbers/leakage.json data/numbers/leakage_canonical.json data/borderline_stats.json; do
  [ -f "$f" ] && cp "$f" "results/$(echo "$f" | tr '/' '_')"
done
for f in runs/teachers/*/train_manifest.json runs/students/*/train_manifest.json; do
  [ -f "$f" ] && { d=results/manifests; mkdir -p $d; cp "$f" "$d/$(echo "$f" | cut -d/ -f2-3 | tr '/' '_').json"; }
done
[ -d figs ] && { mkdir -p results/figs; cp figs/*.png results/figs/ 2>/dev/null; }
git add results
git diff --cached --quiet && { echo "[publish] nothing new"; exit 0; }
git -c user.name="Realmbird" -c user.email="happychristopher777@gmail.com" commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do git push -q origin HEAD 2>/dev/null && { echo "[publish] pushed: $msg"; exit 0; }; sleep $((5 * i)); done
echo "[publish] WARNING: push failed (committed locally)"
