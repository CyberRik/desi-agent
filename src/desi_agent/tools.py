"""Deterministic mock banking backend. SHARED + FROZEN (see CLAUDE.md §2).

Same args always give the same result, so teacher data, training rollouts and eval all agree.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
STATE_CHANGING = {"block_card", "raise_dispute", "transfer_money"}


@lru_cache(maxsize=1)
def tool_specs() -> list[dict]:
    return json.loads((CONTRACTS / "tools.json").read_text(encoding="utf-8"))["tools"]


@lru_cache(maxsize=1)
def _schemas() -> dict[str, dict]:
    return {t["function"]["name"]: t["function"]["parameters"] for t in tool_specs()}


def tool_names() -> list[str]:
    return list(_schemas())


def validate_call(name: str, arguments: Any) -> tuple[bool, str]:
    """(ok, error). ok=False if unknown tool or args fail the JSON schema."""
    if name not in _schemas():
        return False, f"unknown tool: {name}"
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"
    try:
        jsonschema.validate(arguments, _schemas()[name])
    except jsonschema.ValidationError as e:
        return False, e.message
    return True, ""


def _h(*parts: Any) -> int:
    s = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def _get_account_balance(account_type: str) -> dict:
    if account_type == "credit_card":
        return {"account_type": "credit_card", "outstanding_inr": 12840.50, "available_limit_inr": 87159.50,
                "due_date": "2026-11-25", "as_of": "2026-11-12"}
    base = {"savings": 48250.75, "current": 213400.00}[account_type]
    return {"account_type": account_type, "balance_inr": base, "as_of": "2026-11-12"}


_DEBITS = ["Swiggy", "Amazon", "IRCTC", "Airtel recharge", "BigBasket", "Zomato", "Uber", "Electricity BESCOM",
           "UPI to ramesh@okaxis", "Netflix", "Apollo Pharmacy", "Petrol HPCL"]
_CREDITS = ["Salary ACME Ltd", "Refund Amazon", "UPI from priya@ybl", "Cashback Paytm"]


def _get_recent_transactions(count: int, txn_type: str = "all") -> dict:
    out = []
    i = 0
    while len(out) < count and i < 200:
        h = _h("txn", i)
        kind = "credit" if h % 6 == 0 else "debit"
        if txn_type in ("all", kind):
            names = _CREDITS if kind == "credit" else _DEBITS
            amount = 20000 + (h % 6000000) / 100 if names is _CREDITS and h % 4 == 0 else 50 + (h % 500000) / 100
            out.append({
                "transaction_id": f"TXN{100000 + (h % 900000):06d}",
                "date": f"2026-11-{12 - (i % 11):02d}",
                "type": kind,
                "amount_inr": round(amount, 2),
                "description": names[(h // 7) % len(names)],
            })
        i += 1
    return {"transactions": out}


def _block_card(card_last4: str, reason: str) -> dict:
    return {"status": "blocked", "card": f"****{card_last4}", "reason": reason,
            "replacement_eta_days": 5 + _h(card_last4) % 3,
            "reference": f"BLK{_h('blk', card_last4, reason) % 10**8:08d}"}


def _raise_dispute(transaction_id: str, reason: str) -> dict:
    return {"status": "dispute_raised", "transaction_id": transaction_id, "reason": reason,
            "dispute_id": f"DSP{_h('dsp', transaction_id) % 10**7:07d}", "resolution_days": 7}


def _check_upi_status(upi_ref: str) -> dict:
    status = ["success", "success", "success", "pending", "failed"][_h(upi_ref) % 5]
    res = {"upi_ref": upi_ref, "status": status}
    if status == "failed":
        res["note"] = "Amount will be auto-reversed within 48 hours."
    return res


def _transfer_money(to_upi_id: str, amount: float, note: str = "") -> dict:
    return {"status": "success", "to": to_upi_id, "amount_inr": amount, "note": note,
            "upi_ref": f"{_h('upi', to_upi_id, amount) % 10**12:012d}"}


def _find_nearest_branch(pincode: str, kind: str = "branch") -> dict:
    h = _h(pincode, kind)
    return {"kind": kind, "name": f"{['MG Road', 'Station Road', 'Main Bazaar', 'Sector 5'][h % 4]} {kind.upper()}",
            "pincode": pincode, "distance_km": round(0.3 + (h % 50) / 10, 1),
            "open_now": bool(h % 4)}


def _calculate_loan_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> dict:
    r = annual_rate_pct / 12 / 100
    emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    return {"emi_inr": round(emi, 2), "total_payable_inr": round(emi * tenure_months, 2),
            "total_interest_inr": round(emi * tenure_months - principal, 2)}


_IMPL = {
    "get_account_balance": _get_account_balance,
    "get_recent_transactions": _get_recent_transactions,
    "block_card": _block_card,
    "raise_dispute": _raise_dispute,
    "check_upi_status": _check_upi_status,
    "transfer_money": _transfer_money,
    "find_nearest_branch": _find_nearest_branch,
    "calculate_loan_emi": _calculate_loan_emi,
}


def execute(name: str, arguments: dict) -> str:
    """Run a tool and return its result as a canonical JSON string (what goes in the `tool` message).
    Invalid calls return an error payload instead of raising, as a real API would."""
    ok, err = validate_call(name, arguments)
    result = _IMPL[name](**arguments) if ok else {"error": err}
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
