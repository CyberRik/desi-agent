"""[F] Publish the static Lab page (space/lab/index.html + README.md) to the HF Space ToolW/desi-agent-lab.

Static Spaces are free for organizations, unlike Gradio/Docker ones.
    python space/lab/push_lab.py [--space ToolW/desi-agent-lab]
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="ToolW/desi-agent-lab")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="static", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(HERE), repo_id=a.space, repo_type="space", allow_patterns=["index.html", "README.md"],
                      commit_message="[F] Lab update")
    print(f"[F] pushed https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
