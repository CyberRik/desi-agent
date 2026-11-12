"""[R] Validate the reworded (OOD) test: keep only rewrites that ask for exactly the same thing as the original.

A review of the first OOD run showed the rewriter sometimes changed the request while keeping the numbers
("kho gaya" lost -> "khatma ho gaya" finished), dropped details ("last 5 transactions" -> "recent transactions")
or produced garbled text. Those make the gold calls wrong, so they must not count as model errors.

Two filters, both on (original, rewrite) pairs only, never on model outputs:
  1. numbers: the multiset of numbers (incl. single digits and English number words) must be identical
  2. meaning: a judge model answers a yes/no checklist (same action, same reason, same details, same missing
     info, natural text). Only pairs with every answer correct are kept.

    python scripts/eval_ood_validate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-14B-AWQ \
        --orig data/test.jsonl --ood data/test_ood.jsonl --out data/test_ood.valid.json
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from desi_agent.data.checks import to_ascii_digits  # noqa: E402

NUMW = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10"}
PROMPT = """Two messages that an Indian bank customer could send to a support chatbot. Message B is meant to be a
rewording of message A, possibly in romanized Hindi/Bengali mixed with English.

A: {a}
B: {b}

Answer each question with true or false:
- same_action: does B ask the bank to do exactly the same thing(s) as A (e.g. block a card, dispute a payment, check a balance, transfer money, find an ATM, calculate an EMI, or nothing)?
- same_reason: if A gives a reason or type (lost / stolen / fraud / damaged, duplicate / wrong amount / not received / unauthorized, savings / current / credit card, debit / credit), does B give the same one? true if A gives none and B gives none.
- same_details: are all details the same (card digits, counts like 'last 5', amounts, ids, UPI ids, merchant names, PIN codes, rates, durations)?
- same_missing: if A leaves out information the bank would need, does B leave out the same information (and not add it)?
- natural: is B a coherent, natural message a real customer could write (no garbled or invented words)?

Respond with JSON only: {{"same_action": true, "same_reason": true, "same_details": true, "same_missing": true, "natural": true}}"""
KEYS = ["same_action", "same_reason", "same_details", "same_missing", "natural"]


def nums(s: str) -> collections.Counter:
    s = to_ascii_digits(s).lower()
    for w, n in NUMW.items():
        s = re.sub(rf"\b{w}\b", n, s)
    return collections.Counter(x.replace(",", "").rstrip(".") for x in re.findall(r"\d[\d,.]*", s))


async def judge(pairs, base_url, model, concurrency):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=120)
    sem = asyncio.Semaphore(concurrency)

    async def one(i, a, b):
        async with sem:
            for _ in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model, temperature=0.0, max_tokens=120, response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": PROMPT.format(a=a, b=b)}],
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    d = json.loads(r.choices[0].message.content)
                    return i, {k: d.get(k) is True for k in KEYS}
                except Exception:  # noqa: BLE001
                    pass
        return i, None
    return dict(await asyncio.gather(*(one(i, a, b) for i, a, b in pairs)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--orig", default="data/test.jsonl")
    p.add_argument("--ood", default="data/test_ood.jsonl")
    p.add_argument("--out", default="data/test_ood.valid.json")
    p.add_argument("--concurrency", type=int, default=48)
    a = p.parse_args()
    orig = {json.loads(l)["id"]: json.loads(l) for l in Path(a.orig).read_text(encoding="utf-8").splitlines() if l.strip()}
    ood = [json.loads(l) for l in Path(a.ood).read_text(encoding="utf-8").splitlines() if l.strip()]
    num_ok = [r for r in ood if nums(orig[r["meta"]["orig_id"]]["messages"][0]["content"]) == nums(r["messages"][0]["content"])]
    pairs = [(r["id"], orig[r["meta"]["orig_id"]]["messages"][0]["content"], r["messages"][0]["content"]) for r in num_ok]
    verdicts = asyncio.run(judge(pairs, a.base_url, a.model, a.concurrency))
    valid = sorted(i for i, v in verdicts.items() if v and all(v.values()))
    fails = collections.Counter(k for v in verdicts.values() if v for k in KEYS if not v[k])
    rep = {"n_ood": len(ood), "after_numbers": len(num_ok), "valid": len(valid), "judge": a.model,
           "judge_errors": sum(v is None for v in verdicts.values()), "fail_counts": dict(fails),
           "valid_ids": valid, "verdicts": verdicts}
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_ood", "after_numbers", "valid", "judge_errors", "fail_counts")}))


if __name__ == "__main__":
    main()
