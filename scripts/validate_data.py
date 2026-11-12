"""Validate JSONL files against contracts/DATA_FORMAT.md. Exit 1 on any error.

    python scripts/validate_data.py data/seed/seed.jsonl [more.jsonl ...]
"""
import json
import sys

from desi_agent import tools

LANGS = {"en", "hi", "bn", "hi-en", "bn-en"}
CATS = {"single_tool", "multi_tool", "clarify", "no_tool"}
SPLITS = {"train", "val", "test", "seed"}


def check(row: dict) -> list[str]:
    e = []
    for k in ("id", "lang", "category", "split", "source", "messages", "gold"):
        if k not in row:
            e.append(f"missing {k}")
    if e:
        return e
    if row["lang"] not in LANGS: e.append(f"bad lang {row['lang']}")
    if row["category"] not in CATS: e.append(f"bad category {row['category']}")
    if row["split"] not in SPLITS: e.append(f"bad split {row['split']}")
    msgs = row["messages"]
    if not msgs or msgs[0]["role"] != "user": e.append("first message must be user")
    if msgs and (msgs[-1]["role"] != "assistant" or not msgs[-1].get("content")): e.append("last message must be a final assistant answer")
    calls = []
    for i, m in enumerate(msgs):
        if m["role"] == "system": e.append("no system message allowed in rows")
        if m["role"] == "assistant" and m.get("tool_calls"):
            if len(m["tool_calls"]) != 1: e.append(f"msg {i}: exactly one tool call per assistant turn")
            f = m["tool_calls"][0]["function"]
            if not isinstance(f["arguments"], str): e.append(f"msg {i}: arguments must be a JSON string")
            args = json.loads(f["arguments"])
            ok, err = tools.validate_call(f["name"], args)
            if not ok: e.append(f"msg {i}: {err}")
            calls.append((f["name"], args))
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if not nxt or nxt["role"] != "tool": e.append(f"msg {i}: tool call not followed by tool result")
            elif ok and nxt["content"] != tools.execute(f["name"], args): e.append(f"msg {i+1}: tool content != tools.execute output")
    gold = [(g["name"], g["arguments"]) for g in row["gold"].get("tool_sequence", [])]
    if gold != calls: e.append("gold.tool_sequence does not match tool calls in messages")
    if row["category"] in ("clarify", "no_tool") and calls: e.append(f"{row['category']} rows must not call tools")
    if row["gold"].get("must_ask_clarification") != (row["category"] == "clarify"): e.append("must_ask_clarification mismatch")
    return e


def main(paths):
    bad = 0
    n = 0
    seen = set()
    for p in paths:
        for ln, line in enumerate(open(p, encoding="utf-8"), 1):
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            errs = check(row)
            if row.get("id") in seen: errs.append("duplicate id")
            seen.add(row.get("id"))
            for err in errs:
                bad += 1
                print(f"{p}:{ln} [{row.get('id')}] {err}")
    print(f"{n} rows, {bad} errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
