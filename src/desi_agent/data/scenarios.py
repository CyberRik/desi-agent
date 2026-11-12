"""[R] Programmatic scenario sampler.

The *decision* part of every episode (which tool, which args, in which order, or "ask"/"no tool")
is sampled here and executed against the mock bank, so gold labels are correct by construction.
The teacher LLM only writes the natural-language surface: the customer message and the final reply.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .. import tools

LANGS = ["en", "hi", "bn", "hi-en", "bn-en"]

STYLE = {
    "en": "English, as an Indian bank customer would write it.",
    "hi": "Hindi in Devanagari script. Banking words like UPI, ATM, EMI, PIN may stay in English letters.",
    "bn": "Bengali in Bengali script. Banking words like UPI, ATM, EMI, PIN may stay in English letters.",
    "hi-en": "Hinglish: Hindi written in Roman/Latin letters, freely mixed with English words, like a WhatsApp message. No Devanagari.",
    "bn-en": "Banglish: Bengali written in Roman/Latin letters, freely mixed with English words, like a WhatsApp message. No Bengali script.",
}
TONES = ["polite", "urgent and worried", "casual", "short and blunt", "slightly annoyed", "formal, elderly customer", "confused first-time user"]

UPI_IDS = ["ramesh@okaxis", "priya.sharma@ybl", "anil123@paytm", "sunita.das@oksbi", "rahul-k@ibl", "mou.roy@okhdfcbank",
           "kiranstore@upi", "dr.mehta@icici", "tanmoy.b@axl", "papa.gupta@okicici"]
PINCODES = ["700019", "110001", "400050", "560034", "600028", "700091", "226010", "800001", "411014", "302017", "713101", "122002"]
NOTES = {"en": ["rent", "groceries", "school fees", "dinner"], "hi": ["किराया", "दवाई", "फीस"], "bn": ["ভাড়া", "বাজার", "ফি"],
         "hi-en": ["rent", "kiraya", "dawai"], "bn-en": ["bhara", "bazar", "fees"]}


@dataclass
class Scenario:
    lang: str
    category: str
    pattern: str
    intent: str                      # what the customer wants, in English, for the teacher
    must_include: list[str]          # literal strings the customer message must contain
    calls: list[tuple[str, dict]]    # gold tool sequence (empty for clarify/no_tool)
    forbid: list[str] = field(default_factory=list)   # regexes that must NOT appear (clarify)
    missing: str = ""                # clarify: what the assistant must ask for
    native_digits: bool = False      # ask the teacher to write digits in Devanagari/Bengali numerals


def _amount(rng: random.Random) -> int:
    return rng.choice([200, 500, 750, 1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000, 12000, 25000])


def _single(rng: random.Random, lang: str, tool: str) -> Scenario:
    mk = lambda intent, inc, args: Scenario(lang, "single_tool", tool, intent, inc, [(tool, args)])
    if tool == "get_account_balance":
        t = rng.choice(["savings", "current", "credit_card"])
        word = {"savings": "savings account", "current": "current account", "credit_card": "credit card outstanding"}[t]
        return mk(f"wants to know their {word} balance", [], {"account_type": t})
    if tool == "get_recent_transactions":
        n = rng.randint(1, 10)
        ty = rng.choice(["all", "all", "debit", "credit"])
        desc = {"all": "transactions", "debit": "debits / money spent", "credit": "credits / money received"}[ty]
        args = {"count": n} if ty == "all" and rng.random() < 0.5 else {"count": n, "txn_type": ty}
        return mk(f"wants to see their last {n} {desc}", [], args)
    if tool == "block_card":
        last4 = f"{rng.randint(0, 9999):04d}"
        reason = rng.choice(["lost", "stolen", "fraud", "damaged"])
        story = {"lost": "lost their card", "stolen": "their card was stolen", "fraud": "sees fraudulent charges on their card",
                 "damaged": "their card is broken/damaged"}[reason]
        return mk(f"{story} and wants it blocked; card ends with {last4}", [last4], {"card_last4": last4, "reason": reason})
    if tool == "raise_dispute":
        txn = f"TXN{rng.randint(100000, 999999)}"
        reason = rng.choice(["not_received", "duplicate", "unauthorized", "wrong_amount"])
        story = {"not_received": "paid but never received the goods/service", "duplicate": "was charged twice",
                 "unauthorized": "did not make this payment", "wrong_amount": "was charged the wrong amount"}[reason]
        return mk(f"wants to dispute transaction {txn} because they {story}", [txn], {"transaction_id": txn, "reason": reason})
    if tool == "check_upi_status":
        ref = f"{rng.randint(10**11, 10**12 - 1)}"
        return mk(f"made a UPI payment and wants its status; UPI reference number {ref}", [ref], {"upi_ref": ref})
    if tool == "transfer_money":
        to, amt = rng.choice(UPI_IDS), _amount(rng)
        args = {"to_upi_id": to, "amount": amt}
        inc = [to]
        intent = f"wants to send Rs {amt} to UPI id {to}"
        if rng.random() < 0.35:
            note = rng.choice(NOTES[lang])
            args["note"] = note
            inc.append(note)
            intent += f" with the note '{note}'"
        return mk(intent, inc, args)
    if tool == "find_nearest_branch":
        pin = rng.choice(PINCODES)
        kind = rng.choice(["branch", "atm", "atm"])
        args = {"pincode": pin, "kind": kind} if kind == "atm" or rng.random() < 0.5 else {"pincode": pin}
        return mk(f"wants the nearest {'ATM' if kind == 'atm' else 'bank branch'}; their PIN code is {pin}", [pin], args)
    if tool == "calculate_loan_emi":
        lakh = rng.choice([1, 2, 3, 5, 8, 10, 15, 20, 25, 40, 50])
        rate = rng.choice([7.5, 8.25, 8.5, 9, 9.5, 10.5, 11, 12, 13.5, 14])
        years = rng.choice([1, 2, 3, 5, 7, 10, 15, 20])
        in_years = rng.random() < 0.6
        tenure = f"{years} years" if in_years else f"{years * 12} months"
        return mk(f"wants the monthly EMI for a loan of {lakh} lakh rupees at {rate}% yearly interest for {tenure}",
                  [], {"principal": lakh * 100000, "annual_rate_pct": rate, "tenure_months": years * 12})
    raise ValueError(tool)


def _multi(rng: random.Random, lang: str, pattern: str) -> Scenario:
    if pattern == "txn_then_dispute":
        n = rng.randint(3, 8)
        txns = json.loads(tools.execute("get_recent_transactions", {"count": n}))["transactions"]
        counts = {}
        for t in txns:
            counts[t["description"]] = counts.get(t["description"], 0) + 1
        uniq = [t for t in txns if counts[t["description"]] == 1 and t["type"] == "debit"]
        t = rng.choice(uniq or txns)
        reason = rng.choice(["not_received", "duplicate", "unauthorized", "wrong_amount"])
        return Scenario(lang, "multi_tool", pattern,
                        f"wants the assistant to look at their last {n} transactions, find the '{t['description']}' payment "
                        f"and raise a dispute on it (reason: {reason.replace('_', ' ')}). They do NOT know the transaction id.",
                        [], [("get_recent_transactions", {"count": n}),
                             ("raise_dispute", {"transaction_id": t["transaction_id"], "reason": reason})])
    if pattern == "block_then_balance":
        last4 = f"{rng.randint(0, 9999):04d}"
        reason = rng.choice(["lost", "stolen"])
        return Scenario(lang, "multi_tool", pattern,
                        f"{'lost' if reason == 'lost' else 'had stolen'} their credit card ending {last4}; wants it blocked and then "
                        f"wants to know their current credit card outstanding",
                        [last4], [("block_card", {"card_last4": last4, "reason": reason}),
                                  ("get_account_balance", {"account_type": "credit_card"})])
    if pattern == "balance_then_transfer":
        to, amt = rng.choice(UPI_IDS), _amount(rng)
        return Scenario(lang, "multi_tool", pattern,
                        f"wants to first check their savings balance and, if there is enough money, send Rs {amt} to {to}",
                        [to], [("get_account_balance", {"account_type": "savings"}),
                               ("transfer_money", {"to_upi_id": to, "amount": amt})])
    raise ValueError(pattern)


CLARIFY = {
    "block_card_no_card": ("wants their card blocked urgently but does NOT say which card / last 4 digits or why",
                           "the last 4 digits of the card and the reason (lost, stolen, fraud or damaged)", [r"\d{4}"]),
    "transfer_no_amount": ("wants to send money to someone (mentions the person by relation or name only, no UPI id, no amount)",
                           "the payee's UPI id and the amount", [r"@", r"\d"]),
    "dispute_no_txn": ("wants to complain about a wrong charge but gives no transaction id",
                       "the transaction id (or ask whether to look up recent transactions)", [r"TXN", r"txn\d"]),
    "upi_no_ref": ("says their UPI payment is stuck/failed and wants the status, but gives no reference number",
                   "the 12-digit UPI reference number", [r"\d{6,}"]),
    "emi_no_rate": ("wants an EMI calculation for a loan but does not give the interest rate or the tenure",
                    "the interest rate and the loan tenure", [r"%", r"percent", r"\d+\s*(year|month|saal|mahine|bochor|mash)"]),
}

NO_TOOL = [
    "greets the assistant and asks what it can do",
    "says thank you after their issue was solved",
    "asks who will win today's cricket match",
    "asks for a good biryani recipe",
    "asks what KYC means and why the bank needs it (general info, no account action)",
    "asks what the weather will be tomorrow",
    "asks the assistant to tell a joke",
    "asks whether it is safe to share their OTP with a caller claiming to be from the bank",
]

TOOLS8 = ["get_account_balance", "get_recent_transactions", "block_card", "raise_dispute", "check_upi_status",
          "transfer_money", "find_nearest_branch", "calculate_loan_emi"]
MULTI = ["txn_then_dispute", "block_then_balance", "balance_then_transfer"]

DEFAULT_COUNTS = {"single_tool": 45, "multi_tool": 25, "clarify": 18, "no_tool": 8}  # per tool / pattern / item, per lang


def sample_all(seed: int = 1234, counts: dict | None = None, langs: list[str] = LANGS) -> list[Scenario]:
    c = {**DEFAULT_COUNTS, **(counts or {})}
    rng = random.Random(seed)
    out: list[Scenario] = []
    for lang in langs:
        for tool in TOOLS8:
            out += [_single(rng, lang, tool) for _ in range(c["single_tool"])]
        for pat in MULTI:
            out += [_multi(rng, lang, pat) for _ in range(c["multi_tool"])]
        for key, (intent, missing, forbid) in CLARIFY.items():
            out += [Scenario(lang, "clarify", key, intent, [], [], forbid, missing) for _ in range(c["clarify"])]
        for intent in NO_TOOL:
            out += [Scenario(lang, "no_tool", "no_tool", intent, [], []) for _ in range(c["no_tool"])]
    for s in out:
        s.native_digits = s.lang in ("hi", "bn") and bool(s.must_include) and rng.random() < 0.2
    return out
