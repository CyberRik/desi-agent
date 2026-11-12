# DATA_FORMAT — one JSONL row per episode

Validated by `python scripts/validate_data.py <file.jsonl>` (CI runs it on the seed file).

```json
{
  "id": "hi-block-000123",
  "lang": "hi",
  "category": "single_tool",
  "split": "train",
  "source": "teacher:command-a-03-2025",
  "messages": [
    {"role": "user", "content": "मेरा कार्ड खो गया है, आखिरी चार अंक 4821 हैं"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_0", "type": "function",
       "function": {"name": "block_card", "arguments": "{\"card_last4\": \"4821\", \"reason\": \"lost\"}"}}]},
    {"role": "tool", "tool_call_id": "call_0", "name": "block_card", "content": "{\"status\": \"blocked\", ...}"},
    {"role": "assistant", "content": "आपका कार्ड (****4821) ब्लॉक कर दिया गया है।"}
  ],
  "gold": {
    "tool_sequence": [{"name": "block_card", "arguments": {"card_last4": "4821", "reason": "lost"}}],
    "must_ask_clarification": false
  }
}
```

## Fields
| field | values |
|---|---|
| `lang` | `en`, `hi` (Devanagari), `bn` (Bengali script), `hi-en` (Hinglish, Latin script), `bn-en` (Banglish, Latin script) |
| `category` | `single_tool`, `multi_tool` (2–3 sequential calls), `clarify` (required info missing, so the correct move is to ask with no tool call), `no_tool` (chit-chat / out of scope) |
| `split` | `train`, `val`, `test`. **`test` rows live only in `data/test.jsonl` (frozen, R4).** |
| `source` | `human`, `teacher:<model-id>`, `template` |
| `messages` | OpenAI chat format (no system message, which `fmt.py` adds). `arguments` is a JSON **string**, as in the OpenAI API. Tool `content` must equal `tools.execute(...)` output exactly. |
| `gold.tool_sequence` | expected ordered calls with canonical args. Empty list for `clarify` / `no_tool`. |
| `gold.must_ask_clarification` | true only for `clarify` |

Notes
- No system message in the row: it's added at render time from `contracts/system_prompt.txt`.
- Only the first user message is "the task". Multi-turn user follow-ups are out of scope for v1.
- AuditKIT: `auditkit.agent_eval.importers.episode_from_openai_messages(row["messages"])` works on this directly.
