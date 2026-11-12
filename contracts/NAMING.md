# NAMING

Fill in `HF_NS` once (the HF user or org that will own everything; both humans need write access,
so an HF org is easiest). Record it in `docs/DECISIONS.md`.

## Model tags (used in `results/<model_tag>/`)
| tag | what |
|---|---|
| `tinyaya-global-base` | untouched `CohereLabs/tiny-aya-global` (baseline) |
| `tinyaya-global-sft` | + AgentTune SFT distillation LoRA |
| `tinyaya-global-sft-grpo` | + GRPO on top of SFT |
| `teacher` | the teacher model (reference upper bound only) |

## Hugging Face repos (all public, un-gated)
| repo | type | owner |
|---|---|---|
| `HF_NS/desi-agent-tinyaya-sft-lora` | model (adapter) | F |
| `HF_NS/desi-agent-tinyaya-sft-grpo-lora` | model (adapter) | F |
| `HF_NS/desi-agent-tinyaya-merged` | model (best, merged) | F |
| `HF_NS/desi-bank-toolcalls` | dataset (train/val/test + raw teacher data) | R |
| `HF_NS/desi-agent-eval-traces` | dataset (results + traces for every model tag) | R |
| `HF_NS/desi-agent-training-logs` | dataset (trainer logs, configs, manifests) | F |
| `HF_NS/desi-agent-demo` | space (Gradio) | F |

Collections: `Desi Agent: models` and `Desi Agent: datasets` (S1).

## Files
- data: `data/raw/teacher_<lang>.jsonl` (gitignored, pushed to HF), `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`
- results: see METRICS.md
- branches: `r/<id>-<slug>`, `f/<id>-<slug>`, `contract/<slug>`
