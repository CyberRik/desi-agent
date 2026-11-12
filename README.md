<p align="center">
  <img src="docs/assets/banner.png" alt="Desi Agent: a 3.35B Tiny Aya banking agent for English, Hindi, Bengali, Hinglish and Banglish. Task success 45.9% to 97.7%, wrong bank actions 20.8% to 1.8%." width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/ToolW/desi-agent-lab"><img alt="Live demo: Desi Agent Lab" src="https://img.shields.io/badge/Live%20demo-Desi%20Agent%20Lab-e8b349?style=for-the-badge&labelColor=0f1730"></a>
  <a href="https://huggingface.co/ToolW/desi-agent-tinyaya-sft-lora"><img alt="Model: LoRA adapter" src="https://img.shields.io/badge/Model-LoRA%20adapter-2a78d6?style=for-the-badge&labelColor=0f1730"></a>
  <a href="https://huggingface.co/collections/ToolW/desi-agent-datasets-6af5e4ff1ceddc280ca38434"><img alt="Data, logs and traces" src="https://img.shields.io/badge/Data%20%C2%B7%20logs%20%C2%B7%20traces-HF%20collection-2a78d6?style=for-the-badge&labelColor=0f1730"></a>
  <a href="docs/ARTICLE.md"><img alt="Write-up" src="https://img.shields.io/badge/Write--up-ARTICLE.md-898781?style=for-the-badge&labelColor=0f1730"></a>
</p>

Indian banking customers don't write "please block my card". They write *"mera card block kar do"*,
*"আমার কার্ডটা ব্লক করে দাও"*, or *"amar card ta block kore dao"*, often without the card number. A support agent that
acts on these has to pick the right tool, fill the arguments exactly, and **ask instead of guessing** when a
detail is missing, because a wrong `block_card` or `transfer_money` is a real-world harm.

**Desi Agent** is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B), fine-tuned
with [AgentTune](https://github.com/Lexsi-Labs/AgentTune) into exactly that agent, for five ways of writing:
English, Hindi, Bengali, Hinglish and Banglish. Built for the **Lexsi AI Safety & Alignment Hackathon 2026, Track 3:
Agentic Fine-tuning** (reliable tool calling · agentic distillation · efficient trajectories).

## Results

486 held-out test conversations, fp16, greedy decoding, every tool call executed against a deterministic mock bank.

| | Base Tiny Aya | **Desi Agent (ours)** | Qwen3-14B teacher |
|---|---:|---:|---:|
| **Task success** | 45.9% | **97.7%** | 67.5% |
| Right tool(s) | 57.0% | **99.0%** | 72.2% |
| Exact arguments | 45.9% | **97.7%** | 67.5% |
| Valid tool-call JSON | 84.0% | **99.6%** | 97.7% |
| **Wrong bank actions** (block / dispute / transfer not exactly as asked) | 20.8% | **1.8%** | 14.0% |
| Unrequested bank actions (tool not asked for at all) | 8.8% | **0.8%** | 9.5% |
| Two-step tasks | 0.0% | **97.3%** | 43.2% |
| Asks for missing details instead of acting | 38.6% | **96.2%** | 43.2% |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-dark.png">
    <img src="docs/assets/results-light.png" alt="Dumbbell chart of task success, base to fine-tuned, by language (English 40.5 to 96.6, Hindi 53.1 to 99.0, Bengali 42.0 to 99.0, Hinglish 55.0 to 98.0, Banglish 38.6 to 96.0) and by task type (single action 59.5 to 98.0, two-step 0.0 to 97.3, ask for missing info 38.6 to 96.2, no action needed 88.4 to 100.0), with the 14B teacher as a reference ring." width="100%">
  </picture>
</p>

**The 3B student beats the 14B teacher it was distilled from.** That isn't magic: the teacher only writes the
customer's words and the final reply, while every tool call in the training data comes from code, so the labels
are correct by construction and the student never learns the teacher's tool-calling mistakes (see [How it works](#how-it-works)).

## See it

<p align="center">
  <a href="https://huggingface.co/spaces/ToolW/desi-agent-lab"><img src="docs/assets/tour.gif" alt="A Bengali two-step request. The base model blocks the card as 'lost' although the customer said stolen, and never checks the balance. The fine-tuned model blocks it as stolen, then fetches and reports the outstanding amount." width="100%"></a>
  <br><sub>A Bengali two-step request, replayed from the real test traces. The base model (left) blocks the card as "lost" and stops. Ours (right) blocks it as stolen, then checks the outstanding.</sub>
</p>

**[Desi Agent Lab](https://huggingface.co/spaces/ToolW/desi-agent-lab)** is the place to look:

- **Tour.** Base vs fine-tuned on the same customer messages, with every tool call shown as a bank receipt. It
  includes a case we still get wrong.
- **Test explorer.** All 486 test conversations for every model, filterable by language, task type and outcome
  ("fixed by fine-tuning", "still failing", "regressions").
- **Results.** The full tables.

Nothing in the Lab is regenerated: it replays the scored evaluation traces. There is also a
[1080p tour video](https://huggingface.co/datasets/ToolW/desi-agent-eval-traces/blob/main/video/desi-agent-lab-tour.mp4),
and a [local playground](#try-your-own-prompts) for typing your own messages.

## How it works

```mermaid
flowchart LR
    S["Scenario sampler<br/>(code: intent, gold tool calls,<br/>missing-detail cases)"] --> B["Mock bank<br/>executes the gold calls"]
    S --> T["Qwen3-14B teacher<br/>writes only the customer message<br/>+ final reply, in 5 languages"]
    B --> E["Episodes<br/>(OpenAI-style messages + gold)"]
    T --> E
    E --> C["CuratorKIT<br/>dedup · language-ID · LLM judge"]
    C --> SP["Frozen splits<br/>1937 train · 99 val · 486 test"]
    SP --> D["AgentTune distill<br/>LoRA SFT on Tiny Aya Global"]
    D --> M["Desi Agent<br/>LoRA adapter"]
    SP --> EV["Eval: same agent loop + mock bank<br/>our metrics + AuditKIT"]
    M --> EV
```

1. **Labels from code, words from a teacher.** A sampler picks the intent and builds the gold tool calls (8 banking
   tools: balance, transactions, block card, dispute, UPI status, transfer, branch finder, loan EMI). The mock bank
   executes them. The teacher only phrases the customer message and the final reply, so a teacher mistake can
   make a sentence awkward but can't make a label wrong. 12% of training cases (27% of the test, which is balanced
   by task type) deliberately leave out a required detail, where the correct move is to ask.
2. **Curation.** [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) dedups customer messages within a language
   (so no test message has a twin in training), checks the script and language, and an LLM judge filters garbled
   replies.
3. **Distillation.** [AgentTune](https://github.com/Lexsi-Labs/AgentTune) `Project.distill` with a LoRA (r=16 on all
   attention and MLP projections), loss only on the assistant's tokens. Every training row is rendered by the same
   `fmt.py` the evaluation parses, so train and eval see identical text.
4. **Evaluation.** One agent loop (`fmt.run_episode`) for evaluation, the demo and the Lab. Metrics are defined in
   [`contracts/METRICS.md`](contracts/METRICS.md) and cross-checked with
   [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) (tool-call exact match 97.7%).

## Why we trust these numbers (and where we don't)

- **Selected on validation, never on test.** The checkpoint rule was fixed before training: lowest val loss
  (0.424 → 0.273, still falling, train/val gap ~0.07). The test split was frozen before any model saw it.
- **Not memorised.** Success is flat whether a test message closely resembles a training message or not (97.5 /
  99.2 / 97.5 / 96.7% by similarity quartile). In 0 of 486 replies does the model quote bank numbers it didn't
  retrieve in that conversation (base: 20).
- **Still in-distribution.** Test messages come from the same generator as training. Out-of-distribution checks,
  test messages reworded by a different model and a mock bank with shifted values, are being added. They'll
  appear in the Lab as extra columns.
- **Known failure.** 4 of the 11 remaining test failures are the same safety-relevant one: asked to block a card
  with no number given, the model sometimes invents `1234` and blocks it. The Lab's tour shows it.
- **Demo vs metrics.** The local playground runs a 4-bit build. Every number here comes from fp16 evaluation.

The full write-up, including data decisions, error analysis and limits, is in [`docs/ARTICLE.md`](docs/ARTICLE.md).

## Efficient by design

- **2.2 hours on free hardware.** Training ran on a free Kaggle T4 x2 (DDP, fp16), resumable across sessions via
  checkpoints on the Hub.
- **A 4.2x speed-up hiding in plain sight.** `torch.cuda.is_bf16_supported()` returns `True` on a T4, but bf16 is
  emulated there. Profiling showed 84% of GPU time in software bf16 matmuls, and fp16 gave the same loss curve.
- **We measured before adding RL.** We planned GRPO on top of SFT with the evaluation metric as the reward. A probe
  first sampled 8 answers at temperature 1.0 for 377 training prompts in the areas where SFT still errs: 95.7% were
  already right, and only 12.7% of prompts produced any reward difference. With almost no learning signal, we
  kept the SFT model.
- **Runs on a laptop.** The adapter converts to a GGUF LoRA for llama.cpp. The base Q4_K_M model plus our LoRA
  answers in about 3 to 13 s per turn on 2 CPU cores.

## Reproduce

Python 3.12 and [uv](https://github.com/astral-sh/uv). Every step writes a JSON manifest (inputs, git sha, seed 1234).

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]" && python -m pytest -q          # contracts, data format, training plumbing
```

| step | command | where |
|---|---|---|
| Generate data (teacher on vLLM) | `bash scripts/data_kaggle_generate.sh full` | Kaggle T4 x2 |
| Curate + freeze splits | `python -m desi_agent.data.curate --inp data/raw/teacher.jsonl --out data/curated/curated.jsonl`<br>`python -m desi_agent.data.make_splits --inp data/curated/curated.jsonl --out data/` | CPU (+ judge on GPU) |
| Train (SFT, LoRA) | `python scripts/train_kaggle.py push --run sft-a -- --epochs 2` | Kaggle T4 x2, ~2.2 h |
| Evaluate base, teacher, ours | Kaggle script kernel `scripts/eval_kaggle_baseline.py` (`ADAPTER_REPO=ToolW/desi-agent-tinyaya-sft-lora` for ours) | Kaggle T4 x2 |
| Lab / README figures | `python space/lab/push_lab.py` · `python docs/assets/make_readme_assets.py` | CPU |

The splits are in [`data/`](data) (test frozen) and on
[ToolW/desi-bank-toolcalls](https://huggingface.co/datasets/ToolW/desi-bank-toolcalls). Training logs and checkpoints
are in [ToolW/desi-agent-training-logs](https://huggingface.co/datasets/ToolW/desi-agent-training-logs), and every
evaluation trace is in [ToolW/desi-agent-eval-traces](https://huggingface.co/datasets/ToolW/desi-agent-eval-traces).

### Try your own prompts

Base and fine-tuned side by side in a local Gradio app, on CPU (llama.cpp, 4-bit; downloads ~2 GB on first start):

```bash
uv venv -p 3.12 .venv-demo
uv pip install --python .venv-demo/bin/python -r space/requirements.txt gradio
.venv-demo/bin/python space/app.py      # http://127.0.0.1:7860
```

## Repository

| path | what |
|---|---|
| [`contracts/`](contracts) | frozen interfaces: 8 tool schemas, data format, model I/O, metrics, naming |
| [`src/desi_agent/`](src/desi_agent) | `tools.py` mock bank · `fmt.py` prompt render/parse + the one agent loop · `api.py` |
| [`src/desi_agent/data`](src/desi_agent/data), [`eval`](src/desi_agent/eval) | scenario sampler, teacher generation, curation, splits · metrics, eval runner, AuditKIT report |
| [`src/desi_agent/train`](src/desi_agent/train) | policy loading, AgentTune SFT (DDP, resumable), GRPO reward + probe |
| [`space/`](space) | local Gradio playground · [`space/lab/`](space/lab) the Desi Agent Lab |
| [`results/`](results) | metrics for every evaluated model (traces on the Hub) |
| [`docs/`](docs) | [write-up](docs/ARTICLE.md) · [decision log](docs/DECISIONS.md) · [plan](docs/PLAN.md) · [compute](docs/COMPUTE.md) |

## Built with

[AgentTune](https://github.com/Lexsi-Labs/AgentTune) (distillation), [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)
(agent evaluation) and [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) (data curation) by Lexsi Labs, on
[CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). Teacher and judge: Qwen3-14B-AWQ.

Code: MIT. Tiny Aya is CC-BY-NC-4.0 (accept the license on Hugging Face), and the Lexsi toolkits have their own
licenses.
