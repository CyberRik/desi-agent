# MODEL_IO — how episodes become model text and back

Implemented once in `src/desi_agent/fmt.py`. **Training and eval both import it.**
Don't re-implement any of this anywhere else.

## Why our own format
Tiny Aya's chat template may not support a `tools=` argument or a `tool` role (the model is
gated, so check this in F1). To stay template-agnostic we only use `system` / `user` /
`assistant` roles and encode tools as text.

## Rendering (`fmt.to_chat(messages)` → list of {role, content} for `tokenizer.apply_chat_template`)
- `system`: `contracts/system_prompt.txt` with `{TOOLS}` replaced by the compact JSON of `contracts/tools.json`.
- `user` → unchanged.
- `assistant` with `tool_calls` → content is exactly one line per call:
  `<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>`
  (JSON with `ensure_ascii=False`, keys sorted). **One tool call per assistant turn.**
- `tool` → becomes a `user` turn: `<tool_result name="<tool>">{json}</tool_result>`
- `assistant` without tool calls → plain final text (no tags).

## Parsing (`fmt.parse_assistant(text)` → `ParsedTurn`)
- If text contains `<tool_call>…</tool_call>`: parse the JSON of the **first** block.
  `ParsedTurn(kind="tool_call", name, arguments, valid_json=True/False, raw=text)`.
  Invalid JSON → `kind="tool_call", valid_json=False` (counts against schema_valid_rate).
- Otherwise → `ParsedTurn(kind="final", text=stripped_text)`.

## Agent loop (`fmt.run_episode(policy, user_messages, max_steps=6)`)
Shared, deterministic loop used by eval, by GRPO rollouts and by the Space demo:
1. Build the chat, call `policy(chat) -> str`.
2. Parse. If it's a valid tool call, execute it with `tools.execute(name, args)` and append the
   assistant `tool_calls` message + `tool` message. If it's final, append it and stop.
3. Stop after `max_steps` model calls (counts as failure, `truncated=True`).
Returns the full OpenAI-style `messages` list. This is the same format as `DATA_FORMAT.md`.
