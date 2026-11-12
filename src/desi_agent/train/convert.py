"""[F] F2: DATA_FORMAT episodes <-> AgentTune full-tier EventLogs. See contracts/API.md.

    logs = episodes_to_eventlogs(load_split("train"))
    proj = Project(); [proj.add_trajectory(l) for l in logs]
    rows = proj.sft_dataset()                       # AgentTune SFT rows
    chats = [row_to_chat(r) for r in rows]          # == fmt.to_chat(episode["messages"])

AgentTune renders tool calls with plain `json.dumps` and tool results as a `tool` role. Our
contract (MODEL_IO.md) renders them through `fmt.py`. `row_to_chat` maps an AgentTune SFT row back
to the exact `fmt.to_chat` chat, so the student is trained on the same text that eval parses.
"""
from __future__ import annotations

import json

from .. import fmt


def episode_to_eventlog(ep: dict):
    from agenttune.agentic import Event, EventKind, EventLog

    log = EventLog(tier="full", id=ep["id"])
    for m in ep["messages"]:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            log.append(Event(EventKind.OBSERVATION, {"text": m["content"]}))
        elif role == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                args = c["function"]["arguments"]
                args = json.loads(args) if isinstance(args, str) else args
                log.append(Event(EventKind.TOOL_CALL, {"action": {"name": c["function"]["name"], "arguments": args}}))
        elif role == "assistant":
            log.append(Event(EventKind.TEXT, {"text": m.get("content") or ""}))
        elif role == "tool":
            log.append(Event(EventKind.TOOL_RESULT, {"output": m["content"], "name": m.get("name", "")}))
        else:
            raise ValueError(f"{ep['id']}: unknown role {role!r}")
    return log


def episodes_to_eventlogs(episodes: list[dict]) -> list:
    """DATA_FORMAT rows -> AgentTune full-tier EventLogs (log.id = episode id)."""
    return [episode_to_eventlog(ep) for ep in episodes]


def row_to_chat(row: dict, *, with_system: bool = True) -> list[dict]:
    """AgentTune SFT row ({"messages": [...]}) -> the `fmt.to_chat` chat for the same episode.

    Tool calls are re-rendered with `fmt.render_tool_call` (sorted keys, unescaped unicode) and
    `tool` turns become `<tool_result name=...>` user turns. One call per assistant turn means a
    result's tool name is the name of the call just before it.
    """
    msgs: list[dict] = []
    last_call = ""
    for m in row["messages"]:
        role, content = m["role"], m.get("content") or ""
        if role == "assistant" and content.startswith("<tool_call>"):
            turn = fmt.parse_assistant(content)
            if turn.kind != "tool_call" or not turn.valid_json:
                raise ValueError(f"unparseable tool call in SFT row: {content[:200]!r}")
            last_call = turn.name
            msgs.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": "c", "type": "function",
                "function": {"name": turn.name, "arguments": json.dumps(turn.arguments, ensure_ascii=False)}}]})
        elif role == "tool":
            msgs.append({"role": "tool", "name": last_call, "content": content})
        else:
            msgs.append({"role": role, "content": content})
    return fmt.to_chat(msgs, with_system=with_system)
