"""[R] Out-of-distribution test: the frozen test customer messages reworded by a DIFFERENT model family.

The test set was written by the same teacher (Qwen3-14B) as the training data, so a model can score well by
learning that teacher's style. Here each test customer message is rewritten by another model (default: Aya
Expanse 8B) in the same language and style. Gold calls are unchanged. A rewrite is kept only if:
  - every number / id / UPI id in the original appears in the rewrite, and no new ones appear,
  - it passes the same language/script gates as the training data (data/checks.py),
  - it really differs from the original (char 4-gram Jaccard < 0.6).
Rejected rewrites are logged; those episodes are left out of the OOD set (not replaced by the original).

    python scripts/data_ood_reword.py --base-url http://localhost:8000/v1 --model CohereLabs/aya-expanse-8b \
        --inp data/test.jsonl --out data/test_ood.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desi_agent.data.checks import (clean_teacher_text, lang_hygiene, romanized_ok, script_ok,  # noqa: E402
                                    to_ascii_digits)
from desi_agent.data.scenarios import STYLE  # noqa: E402

SEED = 1234
SYS = ("You rewrite messages that Indian bank customers send to a support chatbot. Keep the exact meaning, the request "
       "and every detail (numbers, ids, names, UPI ids), but use different words and sentence structure, the way a "
       "different person would write it. Output ONLY the rewritten message.")
VALUE_RX = re.compile(r"TXN\d+|[\w.\-]+@[\w]+|\d[\d,.]*")


def values(text: str) -> set[str]:
    t = to_ascii_digits(text)
    out = set()
    for v in VALUE_RX.findall(t):
        v = v.rstrip(".,").replace(",", "")
        if v and (not v[0].isdigit() or len(v.replace(".", "")) >= 2):
            out.add(v.lower())
    return out


def jaccard(a: str, b: str) -> float:
    g = lambda t: {t[i:i + 4] for i in range(len(t) - 3)}  # noqa: E731
    x, y = g(a.lower()), g(b.lower())
    return len(x & y) / max(1, len(x | y))


def problems(orig: str, new: str, lang: str) -> list[str]:
    p = []
    if not new or len(new) < 4 or "\n" in new.strip():
        p.append("empty_or_multiline")
    vo, vn = values(orig), values(new)
    if vo - vn:
        p.append("lost_values")
    if vn - vo:
        p.append("new_values")
    if not script_ok(new, lang, sorted(vo)):
        p.append("wrong_script")
    if not romanized_ok(new, lang, sorted(vo)):
        p.append("not_romanized")
    p += lang_hygiene(new, lang)
    if jaccard(orig, new) >= 0.6:
        p.append("too_similar")
    return p


async def run(a) -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=120)
    sem = asyncio.Semaphore(a.concurrency)
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]

    async def one(r):
        orig = r["messages"][0]["content"]
        prompt = (f"Language and style to keep: {STYLE[r['lang']]}\nOriginal message: {orig}\n"
                  "Rewrite it (one message, 1-2 sentences).")
        last = []
        async with sem:
            for k in range(4):
                out = await client.chat.completions.create(model=a.model, temperature=0.7 + 0.1 * k, max_tokens=160,
                                                           messages=[{"role": "system", "content": SYS},
                                                                     {"role": "user", "content": prompt}])
                new = clean_teacher_text(out.choices[0].message.content or "")
                last = problems(orig, new, r["lang"])
                if not last:
                    row = json.loads(json.dumps(r))
                    row["id"] = r["id"] + "-ood"
                    row["source"] = f"reword:{a.model}"
                    row["messages"] = [{"role": "user", "content": new}]  # eval only needs the customer turn + gold
                    row.setdefault("meta", {})["orig_id"] = r["id"]
                    return row, None
        return None, {"id": r["id"], "problems": last}

    res = await asyncio.gather(*(one(r) for r in rows))
    kept = [x for x, _ in res if x]
    rej = [y for _, y in res if y]
    out = Path(a.out)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    Path(f"{out}.rejected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rej), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = {"inp": a.inp, "out": a.out, "model": a.model, "seed": SEED, "git_sha": sha, "kept": len(kept),
           "rejected": len(rej)}
    Path(f"{out}.manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="CohereLabs/aya-expanse-8b")
    p.add_argument("--inp", default="data/test.jsonl")
    p.add_argument("--out", default="data/test_ood.jsonl")
    p.add_argument("--concurrency", type=int, default=32)
    random.seed(SEED)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
