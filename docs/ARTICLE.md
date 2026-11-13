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

**Why no reinforcement learning.** We planned GRPO on top of SFT with the same reward as evaluation, but measured
first whether it would have anything to learn. Sampling 8 first turns at temperature 1.0 for 377 training prompts
in the two areas where SFT still errs (asking for missing details, and the block-card reason), 95.7% of samples
were already correct and only 12.7% of prompts produced any reward difference. With a 97.7% SFT model and almost
no learning signal, we stopped there rather than risk degrading it. Multi-turn conversations turned out to be a
different story; see [Reinforcement learning where there is signal](#reinforcement-learning-where-there-is-signal).

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

## Is it overfitting? Four checks

The test set comes from the same generator as the training data, so a high score could mean the model learned our
generator rather than the task. We checked four ways (`results/robustness.json`, `scripts/eval_ood_*.py`,
`scripts/eval_variants.py`).

1. **Loss curves.** Validation loss was still falling at the last step (0.424 to 0.273), and the train/validation
   gap stayed small and stable (about 0.07). There was no upturn.
2. **Near-copies.** We bucketed test conversations by how similar the customer message is to its closest
   training message (character 4-gram overlap). Task success is flat across the four quartiles, from least to most
   similar: 97.5%, 99.2%, 97.5%, 96.7%. The model does not depend on having seen a similar sentence.
3. **A bank with different numbers.** We re-ran the test with a mock bank that returns shifted balances, amounts and
   transaction / reference ids (the gold dispute target is remapped the same way). Task success stays at **97.7%**,
   and **no reply quotes a value from the original bank**. It reads the tool result rather than remembering
   training data. (In the normal run, 0 of 486 replies quote bank data the model did not retrieve; the base model
   does it 20 times, mostly with invented placeholders.)
4. **Wording from a different model family.** Aya Expanse 8B rewrote every test customer message in the same
   language and style, keeping every number and id. A first pass showed the rewriter sometimes changed the request
   ("lost" became "expired") or dropped details ("last 5 transactions" became "recent transactions"), which makes the
   gold wrong. So a judge (Qwen3-14B, yes/no checklist on the original/rewrite pair only) kept the 380 rewrites
   that ask for exactly the same thing.

| Same 380 conversations | Original wording | **Reworded** | Wrong action (reworded) |
|---|---|---|---|
| Base Tiny Aya | 46.6% | 49.7% | 20.5% |
| **Ours** | 97.1% | **91.8%** | **5.5%** |

| Reworded, by slice | English | Hindi | Hinglish | Bengali | Banglish | Single | Two-step | Ask first | No action |
|---|---|---|---|---|---|---|---|---|---|
| Base | 45.1% | 47.7% | 58.0% | 47.1% | 49.2% | 61.4% | 1.2% | 48.7% | 91.2% |
| **Ours** | **98.6%** | **98.9%** | **93.2%** | **87.1%** | **77.8%** | **88.6%** | **85.5%** | **94.8%** | **100%** |

The gain holds on unseen wording, with a drop of about five points. English and Hindi barely move. Banglish,
Bengali and two-step tasks drop most: the model then disputes the wrong transaction more often or picks a different
block reason. Part of the Banglish drop may still be rewrite quality, since the judge is not strong at Banglish
either.

## Real customers, and the failure that matters

On 270 real, human-written English customer messages from BANKING77 (`scripts/eval_real_queries.py`, written and
run by K), which have no labels for our tools, we scored behaviour. The fine-tuned model asks for missing details
80% of the time (base 26%) and attempts far fewer unsafe actions (8.5% vs 28.1%). But its guesses are well-formed,
so 6.3% of messages end with an **executed** unsafe action, mostly blocking an invented card "1234" when a customer
reports a lost card without its number. The base model's guesses were malformed, and the bank rejected all of them.
Real customers trigger this far more often than our test set, because they almost never give the number.

**A grounding guard outside the model fixes most of it** (`src/desi_agent/eval/guard.py`, `scripts/eval_guard.py`):
a block, dispute or transfer may only execute if its identifiers appear in the conversation (card digits, UPI id and
amount in the customer's words; a transaction id there or in an earlier tool result). Replayed on the existing
traces, it blocks **none of the 170 correct state-changing calls** in the test set, and cuts executed unsafe actions
on real customers from **6.3% to 0.4%** (17 messages to 1). The one left is a dispute on a real transaction from the
customer's statement that they never singled out: grounded, but chosen without asking. A guard like this belongs in
any deployment, whatever the model.

## Reinforcement learning where there is signal

**A failure no test measured.** Trying the local demo, a native speaker (Ritankar) wrote one message in Banglish and
the next in Hinglish. The model took the right action but answered the second message in Banglish. Every training
and test conversation is single-turn, so this was never measured. `scripts/eval_multiturn.py` now measures it:
300 turn-2 conversations from the test split, each placed after a full conversation in another language (switch)
or the same language (control). A reply-language detector (`eval/language.py`) agrees with the gold replies'
language 97-100% of the time. After a switch, the fine-tuned model answered in the customer's new language only
**38.3%** of the time (98.3% without a switch); into English, 6.8%.

**Reward design, with a hacking audit.** We added an experimental reward, `reward_lang` (`eval/rewards.py`): the
contract reward, plus a bonus for answering in the customer's latest language, gated on task success, and a penalty
for answering in another one. Before optimising anything, `scripts/eval_reward_audit.py` scored scripted cheating
policies. It found two holes in the contract reward itself: echoing the customer's message back counted as full
success on "ask first" and "no action" episodes (0.77 of a possible 1.30), and so did any non-question reply.
`reward_lang` closes both: an echo gets no success credit (0.30), and an "ask first" reply must actually ask (a
question mark or a request word in that language; 361 of 362 gold clarify replies pass, 0 of 2,423 gold replies
are flagged as echoes).

**Signal first, then RL.** On 300 multi-turn language-switch prompts built only from the training split, we sampled
8 full conversations per prompt at temperature 1.0 and scored them with `reward_lang` (`scripts/eval_rl_signal.py`).
54.7% of prompts produced different rewards across samples (the single-turn probe: 12.7%), so there is something to
learn. With the time left we used rejection-sampling fine-tuning, the simplest RL method that uses this signal: keep
each prompt's best sample when it is fully correct under `reward_lang` (117 conversations), add them to the
original training set, and retrain with the unchanged pipeline (`scripts/data_rft_from_probe.py`, run `rft-a`,
[ToolW/desi-agent-tinyaya-rft-lora](https://huggingface.co/ToolW/desi-agent-tinyaya-rft-lora)). Validation loss:
0.271 (SFT: 0.273).

| | SFT | **SFT + RL (RFT)** |
|---|---|---|
| Frozen test, task success | 97.7% | **98.2%** |
| Frozen test, wrong or unrequested action | 1.8% | 1.7% |
| Reworded test (380 validated), task success | 91.8% | **93.7%** (Banglish 77.8% to 84.1%) |
| Language switch: reply in the customer's new language | 38.3% | **55.0%** |
| Language switch: right action | 84.3% | **94.0%** |
| Language switch: right action and right language | 34.0% | **52.7%** |
| Same language, multi-turn: right action and right language | 81.7% | **87.7%** |

Nothing got worse, and the targeted behaviour improved most. Two caveats. We did not train a control on the same
multi-turn conversations without reward selection, so part of the gain may come from seeing multi-turn
conversations at all rather than from the reward. And switching into English is still weak (15%), so this is a
partial fix. The reported headline model stays the SFT adapter (it is what every other table, the Lab and the demo
use); the RL adapter is published alongside it. The multi-turn evaluation, reward design, signal probe and RL run
were Ritankar's (R lane); the training pipeline they reuse is K's.

## What it costs to run (ROI)

Measured on our evaluation runs (`scripts/eval_roi.py`, `results/roi.json`): on **one T4** with 32 conversations in
flight, the fine-tuned model handles about **5,000 conversations per hour**. An average conversation takes 1.77 model
calls, about 2,640 prompt tokens (2,260 of them the fixed, cacheable tool prompt) and about 70 output tokens.
Training cost 2.2 hours on a free Kaggle T4 x2 (about $1.50 at the $0.35/hour on-demand T4 price).

Monthly cost, with prompt caching for the API models and 150 output tokens per conversation assumed for them.
Self-hosting is priced at $0.50/hour per T4 (the $0.35 Google Cloud on-demand GPU price plus a small VM) and
provisioned around the clock for twice the average load:

| Conversations per day | **Desi Agent (self-hosted)** | Claude Haiku 5.5 | Claude Sonnet 5.5 | Claude Opus 5.5 |
|---|---|---|---|---|
| 10,000 (pilot) | **$360** (1 T4) | $41 | $744 | $1,489 |
| 100,000 (mid-size bank) | **$720** (2 T4) | $406 | $7,443 | $14,887 |
| 1,000,000 (large bank) | **$6,120** (17 T4) | $4,061 | $74,433 | $148,867 |

Read plainly:

- Against frontier API models the self-hosted 3B agent is **about 10x cheaper than Sonnet and 20x cheaper than
  Opus** at realistic volume.
- Against the smallest current API model (Haiku 5.5) it is **not** cheaper. The case there is not cost: a bank's
  customer messages and account data never leave its own infrastructure (relevant under Indian data-localisation
  rules), there is no per-token bill, and the model and its behaviour are fixed and auditable.
- We have not measured any Claude model's accuracy on this task, so the table compares cost only. Our measured
  quality comparison is against the Qwen3-14B teacher (67.5% task success vs 97.7%).
- API prices are list prices for prompts under 100K tokens (Anthropic pricing table, November 2026); the T4 price is
  Google Cloud's US on-demand rate. Spot or committed-use GPUs ($0.16-0.22/hour) would cut the self-hosted column by
  more than half.

## Limits, stated plainly

- **The headline test set is in-distribution.** It comes from the same scenario templates and the same teacher as
  the training data. The checks above suggest the model learned the task rather than the generator, but the
  reworded score (91.8%) is the more conservative number, and code-mixed Bengali is the weakest spot.
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
