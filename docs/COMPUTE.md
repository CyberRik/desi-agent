# COMPUTE — who runs what, where

Budget: **$20 HF credits on R's (Ritankar's) HF account only.** F has no credits. Credits are tied to
the account that redeemed them, so anything paid runs under R's account.

Prices (hf.co/pricing, 2026-11-12): T4-small $0.40/h · L4 $0.80/h · A10G-small $1.00/h · L40S $1.80/h · A100 $2.50/h.

| work | where | who launches | est. cost |
|---|---|---|---|
| F1/F2 env + smoke tests | **Kaggle** (free, 2×T4, ~30 GPU-h/week per account) or Colab | F | $0 |
| F3 SFT distillation (QLoRA 4-bit or fp16 LoRA; T4 has no bf16) | **Kaggle** (F's account; R's Kaggle account as overflow) | F | $0 |
| R6/R7 eval runs (base, SFT, teacher) | **Kaggle** (R's account) | R | $0 |
| F5 GRPO (rollouts need memory + speed) | **HF Jobs, 1× L4** under R's account, running F's script from the repo | F writes the script + exact command in the issue, **R launches** | ~6–8 h ≈ $5–7 |
| Final eval of the GRPO model, if Kaggle quota runs out | HF Jobs L4 (R) | R | ~$2 |
| F4 demo Space | ZeroGPU if the account allows it, else **CPU-basic Space (free) with a GGUF/4-bit merged model** | F builds, pushes under the HF org | $0 |
| Buffer | — | — | ≥ $8 kept back |

Rules
- Before any paid run: post the command + expected hours in the task issue (`collab.py send R "launch F5 job" -b "..."`), and R confirms.
- Paid jobs always set a timeout (`--timeout`), save checkpoints to the HF Hub every N steps, and stop if loss/reward is flat after 1 h.
- Never put a personal HF token in a notebook or the repo. Use Kaggle/Colab secrets (`HF_TOKEN`). F uses F's own token to push to the shared HF org.
