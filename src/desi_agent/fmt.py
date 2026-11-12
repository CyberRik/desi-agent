"""Messages <-> model text, and the shared agent loop. SHARED + FROZEN (contracts/MODEL_IO.md)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import tools

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

Policy = Callable[[list[dict]], str]
INVALID = "__invalid__"  # tool name recorded for an unparseable tool call


def system_prompt() -> str:
    tpl = (tools.CONTRACTS / "system_prompt.txt").read_text(encoding="utf-8")
    compact = json.dumps([t["function"] for t in tools.tool_specs()], ensure_ascii=False, separators=(",", ":"))
    return tpl.replace("{TOOLS}", compact)


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def render_tool_call(name: str, arguments: dict) -> str:
    return f'<tool_call>{_dump({"name": name, "arguments": arguments})}</tool_call>'


def to_chat(messages: list[dict], *, with_system: bool = True) -> list[dict]:
    """OpenAI-style messages -> plain {role, content} chat (system/user/assistant only)."""
    chat = [{"role": "system", "content": system_prompt()}] if with_system else []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "assistant" and m.get("tool_calls"):
            lines = []
            for c in m["tool_calls"]:
                args = c["function"]["arguments"]
                if c["function"]["name"] == INVALID:
                    lines.append(str(args))  # unparseable call: replay exactly what the model wrote
                    continue
                args = json.loads(args) if isinstance(args, str) else args
                lines.append(render_tool_call(c["function"]["name"], args))
            chat.append({"role": "assistant", "content": "\n".join(lines)})
        elif role == "tool":
            chat.append({"role": "user", "content": f'<tool_result name="{m.get("name", "")}">{m["content"]}</tool_result>'})
        else:
            chat.append({"role": role, "content": m.get("content") or ""})
    return chat


@dataclass
class ParsedTurn:
    kind: str  # "tool_call" | "final"
    raw: str
    name: str | None = None
    arguments: dict | None = None
    valid_json: bool = True
    text: str = ""


def parse_assistant(text: str) -> ParsedTurn:
    m = _CALL_RE.search(text or "")
    if not m:
        return ParsedTurn(kind="final", raw=text, text=(text or "").strip())
    try:
        obj = json.loads(m.group(1))
        name, args = obj["name"], obj.get("arguments", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            raise ValueError
        return ParsedTurn(kind="tool_call", raw=text, name=name, arguments=args)
    except (ValueError, KeyError, TypeError):
        return ParsedTurn(kind="tool_call", raw=text, valid_json=False)


@dataclass
class EpisodeRun:
    messages: list[dict] = field(default_factory=list)
    truncated: bool = False
    raw_outputs: list[str] = field(default_factory=list)


def run_episode(policy: Policy, user_messages: list[dict], max_steps: int = 6) -> EpisodeRun:
    """The one agent loop used by eval, GRPO rollouts and the demo."""
    run = EpisodeRun(messages=[dict(m) for m in user_messages])
    for step in range(max_steps):
        out = policy(to_chat(run.messages))
        run.raw_outputs.append(out)
        turn = parse_assistant(out)
        if turn.kind == "final":
            run.messages.append({"role": "assistant", "content": turn.text})
            return run
        call_id = f"call_{step}"
        if not turn.valid_json:
            # Record the malformed call so metrics can see it, then feed back an error.
            run.messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": INVALID, "arguments": turn.raw}}]})
            run.messages.append({"role": "tool", "tool_call_id": call_id, "name": INVALID,
                                 "content": _dump({"error": "could not parse tool call JSON"})})
            continue
        run.messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": turn.name, "arguments": _dump(turn.arguments)}}]})
        result = tools.execute(turn.name, turn.arguments) if turn.name in tools.tool_names() \
            else _dump({"error": f"unknown tool: {turn.name}"})
        run.messages.append({"role": "tool", "tool_call_id": call_id, "name": turn.name, "content": result})
    run.truncated = True
    return run


def predicted_calls(messages: list[dict]) -> list[dict]:
    """Extract [{name, arguments(dict|None), raw}] from assistant tool_calls, in order."""
    out = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            raw = c["function"]["arguments"]
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError:
                args = None
            out.append({"name": c["function"]["name"], "arguments": args, "raw": raw})
    return out
