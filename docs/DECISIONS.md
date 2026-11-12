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
