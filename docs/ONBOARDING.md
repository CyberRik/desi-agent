# Onboarding (for the F teammate and their Claude)

1. Accept the repo invite on GitHub, then:
   ```bash
   git clone https://github.com/CyberRik/desi-agent.git && cd desi-agent
   echo F > .agent                       # your identity for scripts/collab.py
   gh auth login                         # collab.py uses gh
   uv venv -p 3.12 && uv pip install -e ".[dev]"
   python -m pytest -q
   ```
2. On Hugging Face: accept the license on https://huggingface.co/CohereLabs/tiny-aya-global, join our HF org
   (see `docs/DECISIONS.md`), and create a write token for Kaggle/Colab secrets (`HF_TOKEN`).
3. Fill your GitHub / HF usernames in the header of `docs/PLAN.md` (your first PR, branch `f/F0-names`).
4. Start Claude Code in the repo and tell it: **"You are agent F. Read CLAUDE.md and follow the session start ritual."**
5. Your first task is **F1** (`python scripts/collab.py claim F1`).

Ritankar's Claude does the same with `echo R > .agent`.
