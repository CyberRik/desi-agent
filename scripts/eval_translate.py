"""[R] English machine translations of every non-English text the Lab shows, so judges who don't read Hindi,
Bengali, Hinglish or Banglish can follow the examples.

Collects customer messages (test + OOD + multi-turn turn 2) and final replies of every evaluated model, skips
English, and translates each unique text once with an LLM served by vLLM (default Qwen3-14B-AWQ, thinking off).
Output: results/translations.json = {"model": ..., "items": {sha1(text): english}} keyed by sha1 of the exact text,
so the Lab can look a translation up without shipping the source text twice. Labelled as machine translation.

    python scripts/eval_translate.py --base-url http://localhost:8000/v1 --out results/translations.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from desi_agent.eval.language import lang_match  # noqa: E402

PROMPT = ("Translate this message from an Indian bank's customer-support chat into plain, natural English. It may be "
          "Hindi, Bengali, or romanized Hindi/Bengali mixed with English (Hinglish/Banglish). Keep every number, id, "
          "name and UPI id exactly as written. If it is garbled, translate what can be understood and keep the "
          "garbled part in quotes. Output ONLY the English translation.\n\nMessage: {text}")


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def collect() -> set[str]:
    texts = set()
    for f in glob.glob(str(ROOT / "results/*/test.traces.jsonl")) + glob.glob(str(ROOT / "results/*/multiturn.traces.jsonl")):
        for l in Path(f).read_text(encoding="utf-8").splitlines():
            t = json.loads(l)
            for m in t["messages"]:
                if m["role"] in ("user", "assistant") and not m.get("tool_calls") and m.get("content"):
                    texts.add(m["content"].strip())
    return {t for t in texts if t and not lang_match(t, "en")}


async def run(a):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=120)
    sem = asyncio.Semaphore(a.concurrency)
    if a.texts_file:
        texts = sorted(json.loads(l) for l in Path(a.texts_file).read_text(encoding="utf-8").splitlines() if l.strip())
    else:
        texts = sorted(collect())

    async def one(t):
        async with sem:
            for _ in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=a.model, temperature=0.0, max_tokens=400,
                        messages=[{"role": "user", "content": PROMPT.format(text=t)}],
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    return key(t), (r.choices[0].message.content or "").strip()
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(2)
        return key(t), None

    pairs = await asyncio.gather(*(one(t) for t in texts))
    items = {k: v for k, v in pairs if v}
    out = {"model": a.model, "note": "machine translation for readers who don't know the source language",
           "n": len(items), "items": items}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps({"texts": len(texts), "translated": len(items)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--out", default="results/translations.json")
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--count-only", action="store_true")
    p.add_argument("--dump", help="write the texts to translate (one JSON string per line) and exit")
    p.add_argument("--texts-file", help="translate texts from a --dump file instead of local traces")
    a = p.parse_args()
    if a.count_only:
        print(len(collect()))
        return
    if a.dump:
        Path(a.dump).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in sorted(collect())),
                                encoding="utf-8")
        print("dumped", len(collect()))
        return
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
