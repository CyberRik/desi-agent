"""[F] CLI for train_sft (AgentTune SFT distillation, LoRA on Tiny Aya).

    # F2 smoke on seed data (any GPU, ~minutes)
    python scripts/train_sft.py --split seed --val-split none --max-steps 20 --out outputs/sft_smoke
    # F3 full run (Kaggle T4)
    python scripts/train_sft.py --out outputs/sft --push-to ToolW/desi-agent-tinyaya-sft-lora
    # any other DEFAULTS key:  --set lr=1e-4 --set lora_r=32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desi_agent.train.sft import DEFAULTS, train_sft  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULTS["model_id"])
    ap.add_argument("--split", default=DEFAULTS["split"])
    ap.add_argument("--val-split", default=DEFAULTS["val_split"], help="'none' to skip")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    ap.add_argument("--epochs", type=float, default=DEFAULTS["epochs"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--push-to", default=None)
    ap.add_argument("--set", action="append", default=[], help="key=value (JSON-parsed) overrides")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = {"model_id": a.model, "split": a.split,
           "val_split": None if a.val_split.lower() == "none" else a.val_split,
           "limit": a.limit, "max_steps": a.max_steps, "epochs": a.epochs,
           "load_in_4bit": a.load_in_4bit, "push_to": a.push_to, "out_dir": a.out}
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS:
            raise SystemExit(f"unknown config key {k!r}; one of {list(DEFAULTS)}")
        try:
            cfg[k] = json.loads(v)
        except ValueError:
            cfg[k] = v
    print(json.dumps(train_sft(cfg), indent=2, default=str))


if __name__ == "__main__":
    main()
