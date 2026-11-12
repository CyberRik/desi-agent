# NAMING

**`HF_NS` = `ToolW`** (HF organization; both humans are members). All repos below live there.

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
| `ToolW/desi-agent-tinyaya-sft-lora` | model (adapter) | F |
| `ToolW/desi-agent-tinyaya-sft-grpo-lora` | model (adapter) | F |
| `ToolW/desi-agent-tinyaya-merged` | model (best, merged) | F |
| `ToolW/desi-bank-toolcalls` | dataset (train/val/test + raw teacher data) | R |
| `ToolW/desi-agent-eval-traces` | dataset (results + traces for every model tag) | R |
| `ToolW/desi-agent-training-logs` | dataset (trainer logs, configs, manifests) | F |
| `ToolW/desi-agent-demo` | space (Gradio) | F |

Collections: `Desi Agent: models` and `Desi Agent: datasets` (S1).

## Files
- data: `data/raw/teacher_<lang>.jsonl` (gitignored, pushed to HF), `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`
- results: see METRICS.md
- branches: `r/<id>-<slug>`, `f/<id>-<slug>`, `contract/<slug>`
