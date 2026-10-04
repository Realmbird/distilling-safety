"""YAML config + `key=value` CLI overrides. Values are parsed as YAML (so `lr=1e-4`, `merge=true`,
`save_at_samples=[2500,5000]` all work). A config may name a parent with `inherit: other.yaml`
(resolved relative to its own directory)."""

import sys
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "inherit" in cfg:
        parent = load_config(path.parent / cfg.pop("inherit"))
        cfg = {**parent, **cfg}
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def cli_config() -> dict:
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} CONFIG.yaml [key=value ...]")
    return load_config(sys.argv[1], sys.argv[2:])
