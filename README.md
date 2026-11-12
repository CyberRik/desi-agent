# Desi Agent

**A small multilingual banking agent that calls tools reliably in English, Hindi, Bengali, Hinglish and Banglish.**
Distilled into [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) with
[AgentTune](https://github.com/Lexsi-Labs/AgentTune), data curated with
[CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT), evaluated with
[AuditKIT](https://github.com/Lexsi-Labs/AuditKIT).

Lexsi AI Safety & Alignment Hackathon 2026, **Track 3: Agentic Fine-tuning**
(problem statements: reliable tool calling, agentic distillation, efficient trajectories).

> Status: in progress. Results, HF links and the write-up land here before Nov 15 AoE.

## Why
Most tool-calling work is English-only, but Indian banking customers write "mera card block kar do"
or "আমার UPI পেমেন্ট আটকে গেছে". Closed APIs handle this but cost per token and can't run on-prem
(PII). We show a ~3B open model can do it after targeted distillation, and measure it per language.

## Pipeline
```
teacher LLM + mock bank tools ──▶ raw trajectories ──CuratorKIT──▶ train / val / test (frozen)
                                                                       │
              CohereLabs/tiny-aya-global ──AgentTune distill (SFT LoRA)──▶ ──GRPO (reward = eval metric)──▶ Desi Agent
                                                                       │
                                      AuditKIT + our metrics: base vs SFT vs SFT+GRPO, per language
```

## Metrics (`contracts/METRICS.md`)
task success · tool-selection accuracy · exact-args accuracy · schema-valid rate · redundant calls ·
**false actions** (state-changing calls the user didn't ask for, a safety metric) · all broken down by language.

## Repo map
| path | what |
|---|---|
| `contracts/` | frozen interfaces: tools, data format, model I/O, metrics, cross-lane API, naming |
| `src/desi_agent/{tools,fmt,api}.py` | shared code: mock bank backend, prompt render/parse + agent loop, lane API |
| `src/desi_agent/data`, `eval` | data generation, curation, splits, metrics, eval runner |
| `src/desi_agent/train` | policy loading, AgentTune SFT / GRPO |
| `space/` | Gradio demo |
| `docs/PLAN.md` | tasks + owners · `docs/DECISIONS.md` decision log · `docs/HACKATHON.md` rules |
| `CLAUDE.md` | working rules for collaborators (human and AI) |

## Quickstart
```bash
uv venv -p 3.12 && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
python scripts/data_make_seed.py && python scripts/validate_data.py data/seed/seed.jsonl
python -m pytest -q
```
Training and eval reproduction steps are added as the tasks land (see `docs/PLAN.md`).

## License / models
Code: MIT. Tiny Aya is CC-BY-NC-4.0 (accept the license on HF). The Lexsi toolkits have their own licenses (LSAL).
