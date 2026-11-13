"""[R] Grounding guard for state-changing tool calls (a deployment safety layer outside the model).

BANKING77 showed the failure that matters in production: asked to block a card without the number, the fine-tuned
model sometimes invents a well-formed one ('1234'), and the bank executes it. The guard refuses any state-changing
call whose identifying arguments do not appear in the conversation so far (the customer's messages or earlier tool
results). Read-only tools are never blocked.

  block_card      card_last4 must appear in a customer message
  raise_dispute   transaction_id must appear in a customer message or an earlier tool result
  transfer_money  to_upi_id and amount must appear in a customer message (native digits normalised)
"""
from __future__ import annotations

import json
import re

from ..data.checks import to_ascii_digits

STATE_CHANGING = {"block_card", "raise_dispute", "transfer_money"}


def _norm(s: str) -> str:
    return re.sub(r"[,\s]", "", to_ascii_digits(s or "").lower())


def _amount_forms(x) -> set[str]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return {str(x)}
    forms = {f"{f:.2f}", f"{f:g}"}
    if f.is_integer():
        n = int(f)
        forms |= {str(n)}
        if n % 100000 == 0:
            forms.add(f"{n // 100000}lakh")
        if n % 1000 == 0:
            forms |= {f"{n // 1000}k", f"{n // 1000}thousand", f"{n // 1000}hazar"}
    return forms


def grounded(name: str, args: dict | None, user_text: str, tool_text: str) -> bool:
    """True if a call may execute. user_text / tool_text = everything said / returned before the call."""
    if name not in STATE_CHANGING:
        return True
    if not isinstance(args, dict):
        return False
    u, t = _norm(user_text), _norm(tool_text)
    if name == "block_card":
        return bool(args.get("card_last4")) and _norm(str(args["card_last4"])) in u
    if name == "raise_dispute":
        tid = _norm(str(args.get("transaction_id", "")))
        return bool(tid) and (tid in u or tid in t)
    if name == "transfer_money":
        upi = _norm(str(args.get("to_upi_id", "")))
        return bool(upi) and upi in u and any(_norm(a) in u for a in _amount_forms(args.get("amount")))
    return True


def audit_trace(messages: list[dict]) -> list[dict]:
    """For each state-changing call in a trace: was it grounded at the time it was made?"""
    user, tools, out = "", "", []
    for m in messages:
        if m["role"] == "user":
            user += " " + (m.get("content") or "")
        elif m["role"] == "tool":
            tools += " " + (m.get("content") or "")
        for c in m.get("tool_calls") or []:
            name = c["function"]["name"]
            if name in STATE_CHANGING:
                try:
                    args = json.loads(c["function"]["arguments"])
                except (ValueError, TypeError):
                    args = None
                out.append({"name": name, "args": args, "grounded": grounded(name, args, user, tools)})
    return out
