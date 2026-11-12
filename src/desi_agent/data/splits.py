"""[R] load_split — see contracts/API.md."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_FILES = {"seed": "data/seed/seed.jsonl", "train": "data/train.jsonl", "val": "data/val.jsonl", "test": "data/test.jsonl"}
HF_DATASET_ENV = "DESI_HF_DATASET"  # e.g. "<HF_NS>/desi-bank-toolcalls"


def load_split(name: str) -> list[dict]:
    if name not in _FILES:
        raise ValueError(f"unknown split {name!r}; one of {list(_FILES)}")
    path = ROOT / _FILES[name]
    if path.exists():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if os.environ.get(HF_DATASET_ENV) and name != "seed":
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(os.environ[HF_DATASET_ENV], f"{name}.jsonl", repo_type="dataset")
        return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    raise FileNotFoundError(f"{path} not found (and {HF_DATASET_ENV} not set). Is task R1/R4 merged?")
