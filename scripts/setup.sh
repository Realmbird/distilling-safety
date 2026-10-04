#!/usr/bin/env bash
# Bring a fresh vast.ai box (2x H200/H100, CUDA driver >= 12.8, >=250GB disk) to a runnable state.
#   git clone https://github.com/Realmbird/distilling-safety && cd distilling-safety
#   export HF_TOKEN=...   # that account must have clicked "agree" on the gated repos below
#   bash scripts/setup.sh
# Gated on huggingface.co (accept once, same account as HF_TOKEN):
#   allenai/wildguard, walledai/HarmBench, walledai/XSTest
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[gpu,dev]"

[ -n "${HF_TOKEN:-}" ] || echo "WARNING: HF_TOKEN unset — allenai/wildguard, walledai/HarmBench, walledai/XSTest are gated and will fail"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen2.5-7B-Instruct", "allenai/wildguard"]:
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    print("cached", repo)
PY

python -m pytest -q
python -m distill_safety.data --out data   # builds teacher data + checks every eval loader's columns
echo "setup done. next: bash scripts/run_mvp.sh teachers"
