"""[R] Build results/tour.json: the curated, caption-checked episode list for the Desi Agent Lab tour (and video).

Every caption is a claim about a real trace, so each one carries a check that runs against
results/<tag>/test.traces.jsonl. The script fails if any claim is not true. Output ids/captions only.

    python scripts/eval_tour.py --out results/tour.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE, FINAL = "tinyaya-global-base", "tinyaya-sft-a-final"
DEVANAGARI_LAST = "अंतिम"  # 'antim' = last
BENGALI_DIGIT_4, DEVANAGARI_DIGIT_4 = "৪", "४"


def load(tag):
    p = ROOT / "results" / tag / "test.traces.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def calls(t):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]) if c["function"]["name"] != "__invalid__" else None)
            for m in t["messages"] for c in (m.get("tool_calls") or [])]


def results(t):
    return [json.loads(m["content"]) for m in t["messages"] if m["role"] == "tool"]


def reply(t):
    return t["messages"][-1].get("content") or ""


def rejected(t):
    return any("error" in r for r in results(t))


def asks_last4(t):
    r = reply(t)
    return not calls(t) and ("4" in r or DEVANAGARI_DIGIT_4 in r or BENGALI_DIGIT_4 in r) and \
        (re.search(r"last|digit", r, re.I) is not None or DEVANAGARI_LAST in r)


# (id, base caption, base check, final caption, final check, note shown in the tour)
TOUR = [
    ("hi-en-clarify-upi_no_ref-02271",
     "Checks a UPI payment using a placeholder reference; the bank rejects it",
     lambda t: calls(t) and calls(t)[0][0] == "check_upi_status" and "YOUR" in calls(t)[0][1]["upi_ref"] and rejected(t),
     "Asks for the 12-digit UPI reference before doing anything",
     lambda t: not calls(t) and "12" in reply(t), None),
    ("hi-clarify-block_card_no_card-01040",
     "Invents card number 123456 and tries to block it; the bank rejects it",
     lambda t: calls(t) and calls(t)[0] == ("block_card", {"card_last4": "123456", "reason": "stolen"}) and rejected(t),
     "Asks for the last 4 digits and the reason first",
     asks_last4, "I need my card blocked immediately, please act as soon as possible."),
    ("bn-multi_tool-block_then_balance-01563",
     "Blocks the card as 'lost' (the customer said stolen) and never checks the outstanding",
     lambda t: [c[0] for c in calls(t)] == ["block_card"] and calls(t)[0][1]["reason"] == "lost" and not rejected(t),
     "Blocks it as stolen, then fetches and reports the outstanding",
     lambda t: [c[0] for c in calls(t)] == ["block_card", "get_account_balance"] and calls(t)[0][1]["reason"] == "stolen",
     "My credit card ending 8907 was stolen. Block it and tell me my current outstanding."),
    ("bn-en-multi_tool-txn_then_dispute-02723",
     "Tries to dispute transaction 'YOUR_TRANSACTION_ID'; the bank rejects it",
     lambda t: calls(t) and calls(t)[0][0] == "raise_dispute" and "YOUR" in calls(t)[0][1]["transaction_id"] and rejected(t),
     "Looks up the last 4 transactions, finds BigBasket and disputes it for the wrong amount",
     lambda t: [c[0] for c in calls(t)] == ["get_recent_transactions", "raise_dispute"]
     and calls(t)[1][1] == {"reason": "wrong_amount", "transaction_id": "TXN806078"}, None),
    ("en-single_tool-block_card-00097",
     "Blocks the card but records the reason as 'lost' instead of fraud",
     lambda t: calls(t) == [("block_card", {"card_last4": "6914", "reason": "lost"})],
     "Blocks the card with reason 'fraud'",
     lambda t: calls(t) == [("block_card", {"card_last4": "6914", "reason": "fraud"})], None),
    ("hi-en-multi_tool-balance_then_transfer-*",
     "Checks the balance and stops; the money is never sent",
     lambda t: [c[0] for c in calls(t)] == ["get_account_balance"],
     "Checks the balance, then sends the money to the right UPI id",
     lambda t: [c[0] for c in calls(t)] == ["get_account_balance", "transfer_money"], None),
    ("en-clarify-block_card_no_card-00441",
     "Acts without the card number",
     lambda t: any(c[0] == "block_card" for c in calls(t)),
     "Still fails: invents card 1234 and blocks it instead of asking (4 of our 11 test failures look like this)",
     lambda t: calls(t) and calls(t)[0] == ("block_card", {"card_last4": "1234", "reason": "lost"}), None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/tour.json")
    a = p.parse_args()
    B, F = load(BASE), load(FINAL)
    out = []
    for ep_id, bcap, bchk, fcap, fchk, gloss in TOUR:
        if ep_id.endswith("*"):  # first episode in that family where the claims hold
            pre = ep_id[:-1]
            ep_id = next(i for i in sorted(B) if i.startswith(pre) and bchk(B[i]) and fchk(F[i])
                         and B[i]["scores"]["task_success"] == 0 and F[i]["scores"]["task_success"] == 1)
        b, f = B[ep_id], F[ep_id]
        assert bchk(b), f"base caption not supported by trace: {ep_id}"
        assert fchk(f), f"final caption not supported by trace: {ep_id}"
        still_failing = f["scores"]["task_success"] == 0
        if not still_failing:
            assert b["scores"]["task_success"] == 0 and f["scores"]["task_success"] == 1, ep_id
        out.append({"id": ep_id, "lang": b["lang"], "category": b["category"], "gloss_en": gloss,
                    "base": {"tag": BASE, "caption": bcap, "task_success": b["scores"]["task_success"]},
                    "final": {"tag": FINAL, "caption": fcap, "task_success": f["scores"]["task_success"]},
                    "kind": "still_failing" if still_failing else "fixed_by_sft"})
    Path(a.out).write_text(json.dumps({"traces_repo": "ToolW/desi-agent-eval-traces", "episodes": out},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    for e in out:
        print(e["kind"], e["id"])


if __name__ == "__main__":
    main()
