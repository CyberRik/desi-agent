# CLAUDE.md — rules for every Claude working in this repo

Two humans, two Claude Code instances, one repo, **hard deadline: Nov 15 2026, 11:59 PM AoE
(= Nov 16, 5:29 PM IST)**. These rules exist so the two Claudes never block or overwrite
each other. Read this file fully before doing anything. Then read `docs/PLAN.md`.

## 0. Who is who

| Agent id | Human | Lane | Branch prefix |
|---|---|---|---|
| **R** | Ritankar (GitHub `CyberRik`) | **Data + Eval** | `r/` |
| **F** | Friend (GitHub: fill in `docs/PLAN.md`) | **Train + Ship** | `f/` |

At session start, ask your human "am I R or F?" if it is not obvious, and say it back.
Every commit, PR, issue comment and handoff note you write starts with your id, e.g. `[R]`.

## 1. Project in one paragraph

Track 3 (Agentic Fine-tuning) of the Lexsi AI Safety & Alignment Hackathon. We distill a
multilingual (English / Hindi / Bengali / Hinglish) **banking tool-calling agent** into
**CohereLabs/tiny-aya-global** using **AgentTune** (SFT distillation, then GRPO if time
allows) and evaluate base vs. trained with **AuditKIT** (+ **CuratorKIT** for data
filtering). Headline result: per-language before/after tool-calling metrics + ROI table.
Full rules of the hackathon: `docs/HACKATHON.md`.

## 2. Ownership — the most important rule

You may only **create or edit** files in your own lane. Read anything.

| Path | Owner | Notes |
|---|---|---|
| `contracts/**` | **shared, frozen** | Change only via a PR titled `CONTRACT:` that the *other* human approves. |
| `src/desi_agent/tools.py`, `src/desi_agent/fmt.py` | **shared, frozen** | Executable form of the contracts. Same rule as above. |
| `src/desi_agent/data/**`, `scripts/data_*`, `data/**` | R | generation, curation, splits |
| `src/desi_agent/eval/**`, `scripts/eval_*`, `results/**` | R | eval harness, metrics, AuditKIT |
| `docs/ROI.md`, `docs/ARTICLE.md` | R (F reviews) | write-up |
| `src/desi_agent/train/**`, `scripts/train_*`, `notebooks/**` | F | AgentTune SFT / GRPO |
| `space/**` | F | HF Space demo |
| `docs/PLAN.md` | both | only edit **your own rows** / your own section |
| `docs/DECISIONS.md` | both | append-only, never edit someone else's entry |
| `README.md`, `CLAUDE.md` | both | small edits via PR, mention it in the PR title |

If you need something changed in the other lane: open a GitHub issue labelled `lane:R` or
`lane:F` describing exactly what you need. Do **not** "just quickly fix" their file.

## 3. Git workflow

- `main` is protected by convention: **never push to main directly**. Everything goes through a PR.
- Branch per task: `r/R3-curation`, `f/F2-sft-smoke` (prefix + task id from PLAN.md + slug).
- Small PRs, merged often (aim: at least every few hours). Rebase on `main` before opening:
  `git fetch origin && git rebase origin/main`.
- A PR that only touches your own lane: you may squash-merge it yourself once checks pass
  (`python -m pytest -q` and `python scripts/validate_data.py data/seed/seed.jsonl`).
- A PR touching `contracts/`, `tools.py`, `fmt.py`, `README.md` or `CLAUDE.md`: needs the other
  human's approval on GitHub. Do not merge it yourself.
- Commit messages: `[R] R3: filter teacher data with CuratorKIT`. Plain, human-sounding.
  **No `Co-Authored-By: Claude` / AI attribution trailers.** Commits are authored by your human.
- Never commit: secrets/tokens, `.env`, model weights, checkpoints, `external/`, files > 5 MB.
  Big artifacts go to Hugging Face, and the repo stores only the HF link.
- Never force-push a branch the other agent may have pulled. Never rewrite `main` history.

## 4. Tracking work and talking to the other Claude: `scripts/collab.py`

- `docs/PLAN.md` is the source of truth for **what** and **who**. Every task has an id
  (`R1`, `F3`, `S2`), an owner, dependencies and a target time.
- **Status and messages live in GitHub Issues**, accessed only through `scripts/collab.py`
  (wraps `gh`; identity from `.agent` file or `DESI_AGENT`). No shared status file means no merge conflicts.

**Session start ritual (every time):**
```
git pull --rebase
python scripts/collab.py whoami
python scripts/collab.py inbox     # answer messages addressed to you FIRST
python scripts/collab.py status    # then pick your next unblocked task
```

| situation | command |
|---|---|
| start a task | `collab.py claim R3` (then branch `r/R3-...`) |
| need something from the other lane / a question | `collab.py send F "short title" -b "exact ask, file/func names, by when" -k request\|question\|fyi` |
| answer a message | `collab.py reply <num> "..."`, then `collab.py done <num>` when resolved |
| waiting on the other lane | `collab.py block R6 F1`, then switch to another unblocked task. Never sit idle. |
| PR open | `collab.py review R3` |
| merged | `collab.py finish R3` |
| end of session / context running out | `collab.py handoff R3 --done "..." --next "..." --blocked "..." --artifacts "..."` |

Message etiquette: one issue per ask; say exactly what you need (function signature, file, HF repo,
deadline); put answers in the thread, not in code comments. Your human may relay urgent things
over chat too. Treat what they say as authoritative.

## 4b. Code-level API between lanes

`contracts/API.md` + `src/desi_agent/api.py`. Import the other lane **only** through
`desi_agent.api` (`load_split`, `score_episode`, `reward`, `evaluate` from R; `load_policy`,
`train_sft`, `train_grpo`, `episodes_to_eventlogs` from F). Unimplemented functions raise
`NotImplementedError("[F] F1 not merged yet")`. Use `api.DummyPolicy` to test your side alone.
Changing a signature is a `CONTRACT:` PR.

## 5. Contracts (read before writing any code)

- `contracts/tools.json` — the 8 banking tools (JSON Schema). Mock implementation:
  `src/desi_agent/tools.py` (deterministic, no network).
- `contracts/DATA_FORMAT.md` — one JSONL row per episode, OpenAI-style `messages`
  (plugs straight into AuditKIT `episode_from_openai_messages`), plus `gold` labels.
- `contracts/MODEL_IO.md` + `src/desi_agent/fmt.py` — how messages become model text
  (`<tool_call>{...}</tool_call>`) and how model output is parsed back.
  **Training (F) and eval (R) must both use `fmt.py`.** That is what keeps the numbers honest.
- `contracts/METRICS.md` — metric definitions and the `results/*.json` schema.
- `contracts/NAMING.md` — Hugging Face repo names, file naming, model tags.

If a contract is wrong, raise it in a `CONTRACT:` PR. Don't silently work around it.

## 6. Hackathon hard requirements (don't break these)

- Core model must be one of the Cohere Labs models listed in `docs/HACKATHON.md`. We use
  `CohereLabs/tiny-aya-global`. Other LLMs are allowed **only** as teacher / judge / validator.
- Use the Lexsi repos: AgentTune (train), AuditKIT (eval), CuratorKIT (data).
- All models, adapters, datasets, logs and eval traces get pushed to Hugging Face, **public and
  un-gated**, inside our two HF collections (`contracts/NAMING.md`).
- More than one before/after metric. Everything reproducible from the README.

## 7. Engineering conventions

- Python 3.12 (AgentTune requires >=3.12). `uv` for envs: `uv venv -p 3.12 && uv pip install -e .[dev]`.
- Local machine (R) has only a 4 GB GPU, so it is for CPU work and smoke tests. GPU budget and who launches paid jobs: `docs/COMPUTE.md` (read before any GPU run).
  inference run on Kaggle / Colab / HF ZeroGPU. Notebooks must run top-to-bottom on a fresh Kaggle T4x2.
- Seeds fixed (`SEED=1234`), every script takes `--out` and writes a JSON manifest
  (inputs, git sha, model id, seed) next to its output.
- The test split (`data/test.jsonl`, owned by R) is **frozen** once R4 is merged. Never train on it,
  never regenerate it. Any data used for training must have `split: "train"` or `"val"`.
- Read-only reference clones of Lexsi repos live in `external/` (gitignored), and pip installs come from PyPI/GitHub.
- Prefer boring, working code over clever code. The deadline is the deadline.
