"""[R] Per-episode metrics + RL reward (contracts/METRICS.md). Pure, no model, fast.

`reward()` is F's GRPO reward, so the eval metric and the training signal are the same function.
"""
from __future__ import annotations

import math

from .. import tools
from ..fmt import predicted_calls

DEFAULTS = {
    "get_recent_transactions": {"txn_type": "all"},
    "find_nearest_branch": {"kind": "branch"},
    "transfer_money": {"note": ""},
}
CASE_INSENSITIVE = {"account_type", "txn_type", "reason", "kind", "to_upi_id"}

KEYS = ("schema_valid", "tool_selection_correct", "args_exact", "task_success",
        "n_calls", "redundant_calls", "false_action", "truncated")


def _norm_args(name: str, args: dict | None) -> dict | None:
    if not isinstance(args, dict):
        return None
    out = {**DEFAULTS.get(name, {}), **args}
    return {k: (v.strip().lower() if k in CASE_INSENSITIVE and isinstance(v, str) else v) for k, v in out.items()}


def _val_eq(a, b) -> bool:
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


def call_eq(name_a: str, args_a: dict | None, name_b: str, args_b: dict | None) -> bool:
    if name_a != name_b:
        return False
    a, b = _norm_args(name_a, args_a), _norm_args(name_b, args_b)
    if a is None or b is None or set(a) != set(b):
        return False
    return all(_val_eq(a[k], b[k]) for k in a)


def _final_text(messages: list[dict]) -> str | None:
    if messages and messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls"):
        return (messages[-1].get("content") or "").strip()
    return None


def score_episode(pred_messages: list[dict], gold: dict) -> dict:
    preds = predicted_calls(pred_messages)
    gold_seq = gold.get("tool_sequence", [])
    final = _final_text(pred_messages)
    truncated = final is None

    schema_valid = all(isinstance(p["arguments"], dict) and tools.validate_call(p["name"], p["arguments"])[0]
                       for p in preds)
    names_ok = [p["name"] for p in preds] == [g["name"] for g in gold_seq]
    args_ok = names_ok and all(call_eq(p["name"], p["arguments"], g["name"], g["arguments"])
                               for p, g in zip(preds, gold_seq))
    if gold_seq:
        success = args_ok and bool(final)
    else:
        success = not preds and bool(final)
    false_action = any(
        p["name"] in tools.STATE_CHANGING
        and not any(call_eq(p["name"], p["arguments"], g["name"], g["arguments"]) for g in gold_seq)
        for p in preds)
    return {
        "schema_valid": float(schema_valid),
        "tool_selection_correct": float(names_ok),
        "args_exact": float(args_ok),
        "task_success": float(success and not truncated),
        "n_calls": float(len(preds)),
        "redundant_calls": float(max(0, len(preds) - len(gold_seq))),
        "false_action": float(false_action),
        "truncated": float(truncated),
    }


def reward(pred_messages: list[dict], gold: dict) -> float:
    s = score_episode(pred_messages, gold)
    return (1.0 * s["task_success"] + 0.3 * s["schema_valid"]
            - 0.5 * s["false_action"] - 0.1 * s["redundant_calls"])


def aggregate(rows: list[dict]) -> dict:
    """Mean of every metric over rows (each row = score dict)."""
    if not rows:
        return {k: None for k in KEYS} | {"n": 0}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in KEYS} | {"n": len(rows)}
