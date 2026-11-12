# Desi Agent: teaching a 3B model to bank in five Indian languages

*Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning). Problem statements: reliable tool calling,
agentic distillation, efficient trajectories.*

We fine-tuned **CohereLabs/tiny-aya-global** (3.35B) into a banking support agent that calls tools correctly in
English, Hindi, Bengali, Hinglish and Banglish. On 486 held-out test conversations, task success goes from
**45.9% to 97.7%**, and the rate of wrong or unrequested account actions (blocking a card, raising a dispute or
sending money when it shouldn't) drops from **20.8% to 1.8%**. The fine-tuned 3B model also beats the 14B model we
distilled from (67.5%), because its training labels come from code rather than from the teacher.

See it in action in the **[Desi Agent Lab](https://huggingface.co/spaces/ToolW/desi-agent-lab)**: a guided tour of base
vs fine-tuned on the same customer messages, plus a test explorer over every evaluated conversation
([85-second tour video](https://huggingface.co/datasets/ToolW/desi-agent-eval-traces/blob/main/video/desi-agent-lab-tour.mp4)).

Everything is public: [models](https://huggingface.co/collections/ToolW/desi-agent-models-6af5e4feb0b72f13ce2efec7),
[datasets, logs and traces](https://huggingface.co/collections/ToolW/desi-agent-datasets-6af5e4ff1ceddc280ca38434),
[code](https://github.com/CyberRik/desi-agent).

## Why this task

Hundreds of millions of people in India bank on their phones, and many of them write the way they speak:
Hindi or Bengali in native script, or romanized and mixed with English ("mera card kho gaya hai, block kar do").
A support agent for them has to do three things reliably:

1. **Call the right tool with the right arguments.** A wrong card number or a wrong dispute reason is a wrong
   action on someone's account.
2. **Ask before acting when information is missing.** "Block my card" without a card number must lead to a
   question, not to a guess.
3. **Do this in the customer's language**, including code-mixed text that most models handle poorly.

Small open models are attractive here: they can run on a bank's own hardware, so customer data never leaves it.
We wanted to know whether a 3B multilingual model can be made reliable at this with modest data and compute.

## What the agent can do

Eight tools against a deterministic mock bank (`contracts/tools.json`, `src/desi_agent/tools.py`):
`get_account_balance`, `get_recent_transactions`, `block_card`, `raise_dispute`, `check_upi_status`,
`transfer_money`, `find_nearest_branch` and `calculate_loan_emi`. Three of them change state (block, dispute, transfer).
The mock validates every argument against a JSON Schema and returns an error for anything invalid, e.g. a
6-digit "card_last4" or a transaction id that isn't of the form `TXN123456`.

The model emits one tool call per turn as `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, sees the
result, and either calls the next tool or answers. Training and evaluation render and parse this with the
same code (`src/desi_agent/fmt.py`) and run the same agent loop (`fmt.run_episode`), so the numbers measure exactly
what was trained.

## Data: gold labels from code, wording from a teacher

The dataset is [ToolW/desi-bank-toolcalls](https://huggingface.co/datasets/ToolW/desi-bank-toolcalls).

**Generation.** A scenario sampler (`data/scenarios.py`) decides the language, the task type and the exact gold
tool calls. It covers single actions, two-step tasks (look up transactions then dispute one, block a card then check
the outstanding, check the balance then transfer), "ask first" cases where a required detail is missing, and
"no action" cases (greetings, KYC questions, off-topic requests). A teacher, **Qwen3-14B-AWQ** served with vLLM on
Kaggle T4s, only writes the customer's message and the final reply. The tool results come from the mock bank, so
**every gold label is correct by construction**, whatever the teacher writes.

**Quality gates** (`data/checks.py`, run during generation and again in curation). They check script and language
(Hindi in Devanagari, Bengali in Bengali script, Hinglish/Banglish really romanized Hindi/Bengali rather than
English), that the reply talks to the customer rather than as the customer, and that block/dispute replies quote
the reference id. They also reject Hindi words leaking into Banglish (and vice versa), broken characters, known
mistranslations of "dispute", and any mention of a dispute in a non-dispute episode.

Reading the teacher's output was essential. Two full generation runs were thrown away after review: in the first,
391 of 404 Banglish replies drifted into Hindi. In the second, a glossary line added to the prompt made the teacher
write "block my card **and dispute it**" in 444 episodes whose gold calls only block the card, which is label
noise that would teach the model to claim actions it never took. The third run kept 2,863 of 2,945 scenarios.

**Curation with CuratorKIT** (`data/curate.py`): TextCleaner, SecretsGate, exact dedup on the customer message
within a language, then exact and MinHash (0.85) dedup on message + reply within each language and category.
That leaves 2,617 episodes. We then ran an LLM judge. CuratorKIT's RewardGate with 0 to 1 scores from the 14B judge
turned out to be uncalibrated (98% of episodes scored 0.2 to 0.4, including clean English), so we replaced it with a
yes/no checklist (garbled words, wrong language, wrong fact, incomplete, wrong voice, nonsense customer message).
The checklist is precise but has low recall: it catches skipped facts and echoed messages, not garbled Banglish
words, since the judge is the model that wrote them. Its flags removed 87 episodes from train/val.

**Splits.** 1,937 train, 99 val, 486 test. The test set is balanced over 5 languages x 4 task types (up to 30 per
cell, never more than a third of a cell), with no shared episode ids and **no shared customer messages** across
splits. Our data agent reviewed all 46 test episodes the judge flagged, plus a 60-episode random sample from an
earlier build of the test set. 8 were removed because the customer message did not match the gold calls (for example "last 7 *days*" labelled as the
last 7 *transactions*). The test set was then frozen and never used for training or model selection.

| | English | Hindi | Bengali | Hinglish | Banglish | Total |
|---|---|---|---|---|---|---|
| Train | 331 | 411 | 416 | 395 | 384 | 1,937 |
| Test | 89 | 96 | 100 | 100 | 101 | 486 |

## Training: AgentTune distillation, LoRA

Training code is in `src/desi_agent/train/` and logs are in
[ToolW/desi-agent-training-logs](https://huggingface.co/datasets/ToolW/desi-agent-training-logs).

Episodes are converted to AgentTune full-tier event logs and trained with `Project.distill`, using our own
trainer factory: transformers `Trainer` + PEFT LoRA, with loss only on the assistant's tokens. Each AgentTune SFT row is
re-rendered through `fmt.to_chat`, so the model trains on exactly the text the evaluation parses.

- LoRA r=16, alpha=32, dropout 0.05, on all attention and MLP projections
- 2 epochs, lr 2e-4 cosine, 5% warmup, effective batch 16 (1 x 8 accumulation x 2 GPUs), max length 2048 (nothing truncated)
- Kaggle T4 x2, DDP, fp16: **2.2 hours** for 244 steps, 13.0 GiB peak memory per GPU (gradient checkpointing on;
  turning it off runs out of memory on a T4)

Validation loss fell steadily: 0.424, 0.329, 0.295, 0.276 and 0.273 at steps 50, 100, 150, 200 and 244. We had fixed in
advance that the checkpoint is chosen by validation loss, never by test, so the final adapter
([ToolW/desi-agent-tinyaya-sft-lora](https://huggingface.co/ToolW/desi-agent-tinyaya-sft-lora)) is the reported model. The
epoch-1 checkpoint is shown for reference.

Two engineering notes that may save others time. On a T4, `torch.cuda.is_bf16_supported()` returns True, but bf16
is emulated in software; switching to fp16 made training 4.2x faster with the same loss curve. And since 86% of
every training row is the fixed system prompt (tool schemas plus the chat template's preamble, about 1,280 of
1,490 tokens), rows are almost the same length and padding costs nothing. On a T4, memory rather than padding
limits the batch: batch size 2 runs out of memory, mostly because of the 262k-token vocabulary.

## Evaluation

The code is in `src/desi_agent/eval/`, and results plus every trace are in
[ToolW/desi-agent-eval-traces](https://huggingface.co/datasets/ToolW/desi-agent-eval-traces).

Each test episode starts from the customer's message and runs through the shared agent loop (greedy decoding, at
most 6 steps), with the mock bank executing every call. We serve models with vLLM in fp16. On 20 test
episodes, vLLM and the plain transformers path gave identical scores on all 20. All models see the same system
prompt and tools. Every test conversation, for every model, can be browsed in the Lab's test explorer, including
the 11 the fine-tuned model still gets wrong.

**Metrics** (`contracts/METRICS.md`, all per episode, then averaged):

- **Task success**: the tool-call sequence matches gold exactly (names and canonical arguments), and the episode
  ends with a non-empty answer. For "ask first" and "no action" episodes, success means no tool call and a
  non-empty answer.
- **Right tool sequence**: tool names match gold, arguments aside.
- **Valid tool calls**: every call parses and passes the tool's JSON Schema.
- **Wrong or unrequested action** (safety): any state-changing call that does not exactly match a gold call, so
  an unrequested block/dispute/transfer, or one with wrong arguments. We also report the narrower **unrequested
  action only**, where the state-changing tool isn't in the gold sequence at all.
- **Redundant calls**: calls beyond the gold length.
- **AuditKIT** `agent_eval` scores the same traces independently (tool-call exact match, F1, validity, trajectory
  match, loop detection), as a second opinion on our metric code.

We chose these metrics because, for a bank, "did it do the right thing to the account" matters more than how
nice the reply sounds. Reply wording is not scored.

## Results

486 frozen test episodes, fp16, greedy:

| Metric | Base Tiny Aya | Epoch 1 | **Final (ours)** | Qwen3-14B teacher |
|---|---|---|---|---|
| Task success | 45.9% | 95.9% | **97.7%** | 67.5% |
| Right tool sequence | 57.0% | 99.4% | **99.0%** | 72.2% |
| Valid tool calls (schema) | 84.0% | 99.4% | **99.6%** | 97.7% |
| Wrong or unrequested action | 20.8% | 1.4% | **1.8%** | 14.0% |
| Unrequested action only | 8.8% | 0.4% | **0.8%** | 9.5% |
| Redundant calls / episode | 0.189 | 0.008 | 0.010 | 0.156 |
| AuditKIT tool-call exact | 45.5% | 95.3% | 97.7% | 69.5% |

| Language | n | Base | **Final** | Teacher | Base wrong action | **Final wrong action** |
|---|---|---|---|---|---|---|
| English | 89 | 40.5% | **96.6%** | 85.4% | 23.6% | **3.4%** |
| Hindi | 96 | 53.1% | **99.0%** | 70.8% | 15.6% | **1.0%** |
| Bengali | 100 | 42.0% | **99.0%** | 66.0% | 30.0% | **0.0%** |
| Hinglish | 100 | 55.0% | **98.0%** | 62.0% | 10.0% | **2.0%** |
| Banglish | 101 | 38.6% | **96.0%** | 55.5% | 24.8% | **3.0%** |

| Task type | n | Base | **Final** | Teacher | Base wrong action | **Final wrong action** |
|---|---|---|---|---|---|---|
| Single action | 148 | 59.5% | **98.0%** | 87.2% | 25.0% | **1.4%** |
| Two-step | 111 | 0.0% | **97.3%** | 43.2% | 18.9% | **2.7%** |
| Ask first (missing info) | 132 | 38.6% | **96.2%** | 43.2% | 32.6% | **3.0%** |
| No action needed | 95 | 88.4% | **100.0%** | 99.0% | 0.0% | **0.0%** |

**What changed.**

- **Two-step tasks went from 0% to 97%.** The base model made a single call in 109 of the 111 two-step episodes,
  even when the customer clearly asked for two things.
- **The model stopped making things up when information is missing.** The base model fills the gap with
  placeholders: it blocks card "123456" or "XXXX", disputes transaction "YOUR_TRANSACTION_ID", checks UPI
  reference "YOUR_12_DIGIT_UPI_REFERENCE_NUMBER". In "ask first" episodes it took a wrong action 33% of the time.
  The fine-tuned model asks, at 3% wrong action. The 14B teacher does no better than the base here (43% success,
  34% wrong action), which is why the student can beat it: its labels come from code, not from the teacher.
- **The languages that were weakest gained most.** Bengali went from 42% to 99% and Banglish from 39% to 96%. Every
  language now sits between 96% and 99%.

**Where it still fails** (11 of 486):

- **4 invent card number "1234"** and block it instead of asking (3 English, 1 Hindi). This is the one
  safety-relevant failure mode left.
- **4 pick the wrong block reason**, mostly "lost" when a Banglish/Hinglish customer said the card was stolen.
- **3 one-offs:** a UPI reference with a dropped digit, a transfer missing the note the customer asked for, and an EMI
  calculated when the tenure was missing.

## Limits, stated plainly

- **The test set is in-distribution.** It comes from the same scenario templates and the same teacher as the
  training data. No customer message is shared, but the style is, so part of the 97.7% is the model learning our
  generator's style. The teacher was not trained on it, so the comparison favours the student. *An
  out-of-distribution check (test scenarios reworded by a different model family) is in progress and will be added here.*
- **The bank is a mock.** It has deterministic results, eight tools and one account per user. Real banking APIs,
  authentication and multi-session context are out of scope.
- **The reply wording is noisier than the tool calls.** The teacher's Banglish and Hinglish replies contain some
  garbled words that our checks and judge did not catch. Replies are not scored, so this does not affect the
  numbers, but the model has learned some of that noise.
- **The test review was done by our data agent, not by a native-speaker human.** The review rule and all 8
  rejections are in `data/spotcheck_rejected.txt` and `docs/DECISIONS.md`.
- **The interactive demo runs a 4-bit model.** The local demo (`space/app.py`) uses llama.cpp Q4_K_M plus our LoRA,
  so its replies can differ slightly from the fp16 evaluation. The Desi Agent Lab runs no model: it replays the
  fp16 evaluation traces themselves.

## Reproduce it

Every step has a script, a fixed seed (1234) and a JSON manifest next to its output; see the README. Data
generation and evaluation run on a free Kaggle T4 x2 (`scripts/data_kaggle_generate.sh`,
`scripts/eval_kaggle_baseline.py`), and training does too (`scripts/train_kaggle.py`). The test split is
frozen in `data/test.jsonl`.

## Built with

[AgentTune](https://github.com/Lexsi-Labs/AgentTune) (distillation), [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)
(agent evaluation) and [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) (data curation), by Lexsi Labs.
Base model: CohereLabs/tiny-aya-global. Teacher and judge: Qwen3-14B-AWQ.
