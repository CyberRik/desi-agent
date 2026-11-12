"""[R] Build data/seed/seed.jsonl: small hand-written examples for smoke tests on both lanes.
Tool results are produced by tools.execute so they always match the mock backend.

    python scripts/data_make_seed.py
"""
import json
from pathlib import Path

from desi_agent import tools

# (id, lang, category, user, [(tool, args), ...], final_answer)
SEED = [
    ("en-bal-0001", "en", "single_tool", "What's my savings account balance?",
     [("get_account_balance", {"account_type": "savings"})],
     "Your savings account balance is ₹48,250.75."),
    ("hi-block-0001", "hi", "single_tool", "मेरा डेबिट कार्ड खो गया है, आखिरी चार अंक 4821 हैं। तुरंत ब्लॉक करो।",
     [("block_card", {"card_last4": "4821", "reason": "lost"})],
     "आपका कार्ड ****4821 ब्लॉक कर दिया गया है। नया कार्ड 5-7 दिनों में पहुँच जाएगा।"),
    ("bn-upi-0001", "bn", "single_tool", "আমার UPI পেমেন্টের স্ট্যাটাস দেখুন, রেফারেন্স নম্বর 402918375610",
     [("check_upi_status", {"upi_ref": "402918375610"})],
     None),
    ("hien-emi-0001", "hi-en", "single_tool", "bhai 5 lakh ka loan 9.5% pe 36 mahine ke liye, EMI kitna banega?",
     [("calculate_loan_emi", {"principal": 500000, "annual_rate_pct": 9.5, "tenure_months": 36})],
     None),
    ("bnen-atm-0001", "bn-en", "single_tool", "amar kache ekta ATM khuje dao, pincode 700019",
     [("find_nearest_branch", {"pincode": "700019", "kind": "atm"})],
     None),
    ("hien-txn-0001", "hi-en", "single_tool", "mere last 3 debit transactions dikhao",
     [("get_recent_transactions", {"count": 3, "txn_type": "debit"})],
     None),
    ("en-dispute-0001", "en", "multi_tool", "I see a duplicate charge in my last 2 transactions. Please dispute the duplicate one.",
     [("get_recent_transactions", {"count": 2, "txn_type": "all"}),
      ("raise_dispute", {"transaction_id": "__FROM_RESULT__", "reason": "duplicate"})],
     None),
    ("hi-pay-0001", "hi", "single_tool", "ramesh@okaxis को ₹1500 भेज दो, नोट में 'किराया' लिखना",
     [("transfer_money", {"to_upi_id": "ramesh@okaxis", "amount": 1500, "note": "किराया"})],
     None),
    ("hien-clar-0001", "hi-en", "clarify", "mera card block kar do jaldi",
     [], "Zaroor. Kripya card ke last 4 digits aur reason batayein (kho gaya, chori, fraud ya damaged)?"),
    ("bn-clar-0001", "bn", "clarify", "আমার ভাইকে টাকা পাঠাতে চাই",
     [], "অবশ্যই। আপনার ভাইয়ের UPI আইডি এবং কত টাকা পাঠাতে চান, জানাবেন?"),
    ("en-notool-0001", "en", "no_tool", "Hi! What can you help me with?",
     [], "Hi! I can check balances and transactions, block cards, raise disputes, check UPI payments, send money, find branches/ATMs and calculate loan EMIs."),
    ("hi-notool-0001", "hi", "no_tool", "आज क्रिकेट मैच कौन जीतेगा?",
     [], "माफ़ कीजिए, मैं केवल बैंकिंग से जुड़े सवालों में मदद कर सकता हूँ।"),
]


def build(row):
    rid, lang, cat, user, calls, final = row
    msgs = [{"role": "user", "content": user}]
    gold = []
    for i, (name, args) in enumerate(calls):
        args = dict(args)
        if args.get("transaction_id") == "__FROM_RESULT__":
            prev = json.loads(msgs[-1]["content"])["transactions"]
            args["transaction_id"] = prev[0]["transaction_id"]
        res = tools.execute(name, args)
        assert "error" not in json.loads(res), (rid, res)
        cid = f"call_{i}"
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": cid, "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, sort_keys=True)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "name": name, "content": res})
        gold.append({"name": name, "arguments": args})
    if final is None:
        final = f"Done. Result: {msgs[-1]['content']}"
    msgs.append({"role": "assistant", "content": final})
    return {"id": rid, "lang": lang, "category": cat, "split": "seed", "source": "human",
            "messages": msgs, "gold": {"tool_sequence": gold, "must_ask_clarification": cat == "clarify"}}


if __name__ == "__main__":
    out = Path("data/seed/seed.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in SEED:
            f.write(json.dumps(build(r), ensure_ascii=False) + "\n")
    print(f"wrote {len(SEED)} rows -> {out}")
