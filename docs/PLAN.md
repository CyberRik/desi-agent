# PLAN

**Deadline: Nov 15 2026 11:59 PM AoE = Nov 16 5:29 PM IST.** Internal freeze: **Nov 16, 12:00 IST**
(submission form filled, everything public). Times below are IST.

Team: R = Ritankar (GitHub `CyberRik`, HF `____`) · F = friend (GitHub `____`, HF `____`)

Status of each task is tracked in its GitHub issue (`python scripts/collab.py status`), not in this file.
The table is parsed by `collab.py bootstrap`, so keep the column layout.

## Tasks

| id | task | owner | depends on | due (IST) |
|---|---|---|---|---|
| S0 | Both humans: accept Tiny Aya license on HF, join Lexsi Discord, get HF credits, create HF org | S | - | Nov 12 night |
| R1 | Contracts, mock tools, fmt, seed data, validator, collab CLI (bootstrap) | R | - | Nov 12 night |
| F1 | Kaggle/Colab env: install AgentTune + Tiny Aya, implement `train.policy.load_policy`, check chat template, run seed episodes through `fmt.run_episode` | F | R1 | Nov 13 11:00 |
| R2 | Teacher generation script: ~3k episodes across 5 langs × 4 categories (teacher via API, tools via `tools.execute`) | R | R1 | Nov 13 13:00 |
| R3 | CuratorKIT pass: dedup, language-ID check, quality gate, re-validate. Report kept/dropped counts | R | R2 | Nov 13 18:00 |
| R4 | Freeze splits: train/val + **human-spot-checked test (~600, balanced)**. Push `desi-bank-toolcalls` to HF | R | R3 | Nov 13 20:00 |
| R5 | Eval harness: `eval.metrics.score_episode`/`reward`, `eval.runner.evaluate`, AuditKIT agent_eval report | R | R1 | Nov 13 18:00 |
| F2 | `train.convert.episodes_to_eventlogs` + SFT smoke run on seed (AgentTune distill, LoRA) | F | R1, F1 | Nov 13 16:00 |
| R6 | Baseline eval: tiny-aya-global base (+ teacher as upper bound) on test | R | R4, R5, F1 | Nov 14 10:00 |
| F3 | Full SFT distillation on train, push `desi-agent-tinyaya-sft-lora` + logs | F | R4, F2 | Nov 14 12:00 |
| R7 | Eval SFT model, per-language tables + plots, error analysis (5 good / 5 bad traces) | R | F3, R5 | Nov 14 16:00 |
| F5 | GRPO on top of SFT with `api.reward` (stretch, kill at Nov 15 12:00 if not improving) | F | F3, R5 | Nov 15 12:00 |
| F4 | HF Space demo (Gradio): pick language, chat, see tool calls live; base vs trained toggle | F | F1, F3 | Nov 15 14:00 |
| R8 | ROI table: closed API vs self-hosted Tiny Aya (volume, tokens/s, GPU $/h) | R | F3 | Nov 15 14:00 |
| R9 | HF article / README write-up (metrics + why, data, approach, results, ROI, limits) | R | R7 | Nov 15 20:00 |
| S1 | HF collections (models, datasets), all public + un-gated, README reproducibility check from clean clone | S | F3, R4 | Nov 15 22:00 |
| S2 | Abstract (<500 words) + submit form (registered email) | S | S1, R9 | Nov 16 12:00 |

## Fallbacks (decided upfront, no debate later)
- AgentTune distill breaks for Tiny Aya → plain TRL SFTTrainer through AgentTune's `trainer_factory` hook (as in `examples/agentic_distillation_real.py`), still via `Project.distill`.
- GRPO not improving by Nov 15 12:00 → ship SFT only and report GRPO as "attempted" with logs.
- Teacher API budget issue → fewer rows (≥1.5k) with more `template`-sourced rows.
- Tiny Aya Global too slow on free GPU → 4-bit QLoRA, max_len 1024, batch via grad accumulation.
