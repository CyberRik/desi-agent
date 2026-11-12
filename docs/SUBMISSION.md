# Submission (Round 1 form, due Nov 15 2026, 11:59 PM AoE)

Draft prepared by K (F's agent) for both humans to check and submit. Only the email is left blank on purpose.

| form field | value |
|---|---|
| 1. Registered email | *(fill in: the email we registered with)* |
| 2. Abstract (< 500 words) | below |
| 3. HF model collection | https://huggingface.co/collections/ToolW/desi-agent-models-6af5e4feb0b72f13ce2efec7 |
| 4. HF dataset collection (data, logs, traces) | https://huggingface.co/collections/ToolW/desi-agent-datasets-6af5e4ff1ceddc280ca38434 |
| 5. Write-up | https://github.com/CyberRik/desi-agent/blob/main/docs/ARTICLE.md *(or the HF Article link, if we publish one)* |
| 6. GitHub repo | https://github.com/CyberRik/desi-agent |
| 7a. Repos used | AgentTune (distillation), AuditKIT (agent evaluation), CuratorKIT (data curation) |
| 7b. Core model | CohereLabs/tiny-aya-global (teacher/judge: Qwen3-14B-AWQ; OOD rewriter: Aya Expanse 8B) |
| 7c. Track | Track 3: Agentic Fine-tuning |
| 7d. Problem statements | Reliable tool calling · Agentic distillation · Efficient trajectories |
| Demo | https://huggingface.co/spaces/ToolW/desi-agent-lab |

## Abstract

**Desi Agent: teaching a 3B model to bank in five Indian languages**

Indian banking customers write to support in English, Hindi, Bengali, and the romanized mixes Hinglish and
Banglish, and they often leave out the detail an action needs. A support agent must pick the right tool, fill its
arguments exactly, and ask instead of guessing, because a wrong card block or money transfer is a real harm. We
distilled this behaviour into CohereLabs/tiny-aya-global (3.35B) with AgentTune.

Data: a code-driven scenario sampler builds every gold tool call (8 banking tools against a deterministic mock
bank). A Qwen3-14B teacher only writes the customer message and the final reply in the five languages, so a
teacher error can make a sentence awkward but never a label wrong. About 12% of training cases deliberately omit
a required detail, where the correct action is to ask. Our own checks filtered script and language drift,
CuratorKIT cleaned and deduplicated the data, and an LLM checklist judge removed replies that skipped facts. This left 1,937 train, 99 val and 486 frozen, balanced test
conversations, with no customer message shared between splits.

Training: AgentTune `Project.distill` with LoRA (r=16) on Tiny Aya Global, loss on assistant tokens only, the
same prompt renderer as evaluation. It took 2.2 hours on a free Kaggle T4 x2. The checkpoint was chosen on
validation loss only. Profiling showed T4s emulate bf16 in software, and switching to fp16 made training 4.2x
faster.

Results on the 486 test conversations (fp16, greedy, every call executed against the mock bank): task success
rises from 45.9% to 97.7%, valid tool calls from 84.0% to 99.6%. Wrong
bank actions (blocking, disputing or sending money when it shouldn't) fall from 20.8% to 1.8%. Two-step tasks
go from 0% to 97.3%, and asking for missing details from 38.6% to 96.2%. Every language lands between 96% and
99%. The 3B student beats its 14B teacher (67.5%), because its labels come from code rather than from the
teacher. AuditKIT agrees (tool-call exact match 97.7%).

Robustness, reported plainly: success does not depend on how similar a test message is to training, and
replies never quote bank values the model did not retrieve, and with a mock bank returning shifted values,
success stays at 97.7%. On test messages reworded by a different model
(Aya Expanse 8B, meaning-validated), success drops from 97.1% to 91.8% (base: 46.6% to 49.7%), and most of the
drop is in Banglish (77.8%). On real, human-written English customer messages (BANKING77), [BANKING77 result
pending]. The main remaining failure is safety-relevant: asked to block a card with no number, the model
sometimes invents one.

We planned GRPO on top of SFT, but measured first: 95.7% of sampled answers were already correct and only 12.7%
of prompts produced any reward signal, so we kept the SFT model.

Everything is public: adapter, data, training logs, every evaluation trace, and the Desi Agent Lab, an
interactive tour and test explorer that replays the scored traces.

<!-- word count check: python -c "import re;t=open('docs/SUBMISSION.md').read().split('## Abstract')[1];print(len(re.findall(r'\S+',t)))" -->
