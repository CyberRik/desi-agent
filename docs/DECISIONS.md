# DECISIONS (append-only)

Format: `YYYY-MM-DD [R|F|S] decision. Why.` Never edit or delete someone else's entry.
Add a new entry that supersedes it instead.

- 2026-11-12 [R] Track 3 (Agentic Fine-tuning), banking/UPI tool-calling agent in en/hi/bn/hi-en/bn-en. Why: least crowded track, and multilingual tool calling plays to Tiny Aya's strength.
- 2026-11-12 [R] Student = `CohereLabs/tiny-aya-global`. Why: multilingual variant, ~3B, fits free GPUs with LoRA.
- 2026-11-12 [R] Own text tool-call format (`<tool_call>{json}</tool_call>`), tool results fed back as user turns. Why: template-agnostic, and one parser shared by train and eval (`fmt.py`).
- 2026-11-12 [R] Storage format = OpenAI chat messages. Why: AuditKIT `episode_from_openai_messages` ingests it directly, and it converts easily to AgentTune `EventLog`.
- 2026-11-12 [R] One tool call per assistant turn, max 6 steps. Why: simpler parsing and reward, and multi-step behaviour is still covered.
- 2026-11-12 [R] Coordination via GitHub Issues through `scripts/collab.py`, not a shared status file. Why: no merge conflicts between two agents.
- OPEN: teacher model for R2 (candidates: Cohere Command A via trial key, which fits the partner theme; or another strong API model). Decide by Nov 13 morning.
- OPEN: HF namespace (`HF_NS`) — org recommended so both can push.
- 2026-11-12 [F] Tiny Aya Global (Cohere2) chat template accepts our `system` role as-is (no folding needed); it also prepends Cohere's own safety preamble, which we keep. Assistant turns render as `<|START_RESPONSE|>…<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`. Why: verified in F1 on HF Jobs T4 (job 6af5767f9737d9a9e093da5b).
- 2026-11-12 [F] SFT = AgentTune `Project.distill` with our own `trainer_factory` (transformers `Trainer` + PEFT LoRA, loss on assistant tokens only), and AgentTune SFT rows are re-rendered through `fmt.to_chat` before tokenizing. Why: AgentTune's own row format (`json.dumps` tool calls, `tool` role) differs from MODEL_IO.md; training must see the exact text eval parses.
- 2026-11-12 [F] Policy generation stops at `</tool_call>`. Why: base model otherwise hallucinates its own `<tool_result>`; parse reads only the first call, so scores are unchanged and decoding is faster.
- 2026-11-12 [R] Compute: only R has HF credits ($20). Kaggle/Colab for SFT + eval. HF Jobs L4 under R's account only for GRPO (R launches F's script). Demo Space on free hardware. See docs/COMPUTE.md.
- 2026-11-12 [F] F3 runs as Kaggle script kernels on GPU T4 x2 with DDP (`torchrun`, one LoRA replica per T4), launched by `scripts/train_kaggle.py`. Checkpoints (adapter + optimizer + trainer state) mirror to `ToolW/desi-agent-training-logs/checkpoints/<run>/latest`, so a run resumes in a new 12 h session on either Kaggle account. Two runs in parallel: `sft-a` (2 epochs, F's account) and `sft-b` (1 epoch, R's account, early adapter for R7). The better one on val is published. Why: one T4 is ~13 s/episode, so 2 epochs of ~2k episodes do not fit in one Kaggle session.
