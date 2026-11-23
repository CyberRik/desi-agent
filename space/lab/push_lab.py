"""[F] Publish the static Lab page (space/lab/index.html + README.md) to the HF Space ToolW/desi-agent-lab.

Static Spaces are free for organizations, unlike Gradio/Docker ones.
    python space/lab/push_lab.py [--space ToolW/desi-agent-lab]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_legacy() -> Path:
    """stage2_legacy.json for the Stage 2 tab: round-1 numbers that live in the repo, not on the Hub."""
    valid = json.loads((ROOT / "data/test_ood.valid.json").read_text(encoding="utf-8"))["valid_ids"]
    b77 = json.loads((ROOT / "results/banking77.json").read_text(encoding="utf-8"))
    out = HERE / "stage2_legacy.json"
    out.write_text(json.dumps({"ood_valid_ids": sorted(valid), "banking77_unsafe_executed": {
        k: v["unsafe_executed"] for k, v in b77.items() if isinstance(v, dict) and "unsafe_executed" in v}}),
        encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="ToolW/desi-agent-lab")
    ap.add_argument("--legacy-only", action="store_true", help="only write stage2_legacy.json (local preview)")
    a = ap.parse_args()
    write_legacy()
    if a.legacy_only:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="static", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(HERE), repo_id=a.space, repo_type="space", allow_patterns=["index.html", "README.md", "stage2_legacy.json"],
                      commit_message="[F] Lab update")
    print(f"[F] pushed https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
