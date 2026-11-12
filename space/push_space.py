"""[F] F4: bundle and upload the demo to the HF Space ToolW/desi-agent-demo.

The Space gets app.py + requirements + README, plus the exact repo files the agent loop needs
(src/desi_agent/{fmt,tools,train/chat}.py and contracts/), in the repo layout, so tools.CONTRACTS resolves.

    python space/push_space.py [--space ToolW/desi-agent-demo] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "space/app.py": "app.py", "space/requirements.txt": "requirements.txt", "space/README.md": "README.md",
    "src/desi_agent/__init__.py": "src/desi_agent/__init__.py",
    "src/desi_agent/fmt.py": "src/desi_agent/fmt.py",
    "src/desi_agent/tools.py": "src/desi_agent/tools.py",
    "src/desi_agent/train/chat.py": "src/desi_agent/train/chat.py",
    "contracts/tools.json": "contracts/tools.json",
    "contracts/system_prompt.txt": "contracts/system_prompt.txt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="ToolW/desi-agent-demo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        for src, dst in FILES.items():
            (Path(d) / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / src, Path(d) / dst)
        (Path(d) / "src/desi_agent/train/__init__.py").write_text('"""[F] demo bundle: only chat.py is needed."""\n')
        if a.dry_run:
            print(sorted(str(p.relative_to(d)) for p in Path(d).rglob("*") if p.is_file()))
            return
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.space, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
        api.upload_folder(folder_path=d, repo_id=a.space, repo_type="space", commit_message="[F] F4: demo update")
        print(f"[F] pushed https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
