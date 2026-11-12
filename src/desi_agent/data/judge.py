"""[R] R3: LLM-judge quality scores for curated episodes, via CuratorKIT RewardGate.

The deterministic gates in checks.py catch wrong script / language drift / missing refs, but not garbled words
("48 barer moddhe"), replies that skip the asked fact, or invented reassurances. A judge model scores each episode
0-1; the threshold is picked afterwards from the score distribution (scores are stored, nothing is dropped here).

    # judge served by vLLM (same Qwen3-14B-AWQ teacher, on Kaggle T4x2)
    python -m desi_agent.data.judge --inp data/curated/curated.jsonl --out data/curated/judge_scores.jsonl \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-14B-AWQ
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LANG_NAME = {"en": "English", "hi": "Hindi in Devanagari script", "bn": "Bengali in Bengali script",
             "hi-en": "Hinglish (romanized Hindi mixed with English)",
             "bn-en": "Banglish (romanized Bengali mixed with English; NOT Hindi)"}
EXPECT = {"single_tool": "report the outcome using the facts in the tool results",
          "multi_tool": "report the outcome of every action using the facts in the tool results",
          "clarify": "NOT act, and ask the customer for the missing information",
          "no_tool": "answer briefly without any account action; off-topic requests are politely declined"}

JUDGE_PROMPT = """You review one conversation of an Indian bank's support chatbot. Be strict.

{instruction}

BOT FINAL REPLY: {response}

Give a LOW overall_score (0.0-0.4) if ANY of these is true:
- the reply or the customer message contains garbled, invented or misspelled words, or is not natural for a native speaker
- the reply is not in the required language/style
- the reply states a number, date, id or fact that is not in the tool results or the customer message, or gets one wrong
- the reply ignores part of what the customer asked, or does not do what the bot was expected to do
- the reply speaks as the customer instead of to the customer
- the customer message is nonsense or does not fit the expected bot behaviour
Give 0.8-1.0 only if both texts are natural and correct and the reply does exactly what is expected.

Respond in JSON format ONLY:
{{"overall_score": 0.XX, "dimension_scores": {{"coherence": 0.XX, "truthfulness": 0.XX, "instruction_following": 0.XX}}, "weaknesses": "one short sentence"}}
/no_think"""


def context(row: dict) -> str:
    m = row["messages"]
    calls = [(x["tool_calls"][0]["function"]["name"], x["tool_calls"][0]["function"]["arguments"])
             for x in m if x.get("tool_calls")]
    results = [x["content"] for x in m if x["role"] == "tool"]
    lines = [f"Required language/style: {LANG_NAME[row['lang']]}",
             f"Expected bot behaviour: {EXPECT[row['category']]}",
             f"CUSTOMER MESSAGE: {m[0]['content']}"]
    for (name, args), res in zip(calls, results):
        lines.append(f"TOOL CALL: {name}({args}) -> RESULT: {res}")
    if not calls:
        lines.append("(no tool was called)")
    return "\n".join(lines)


def judge(rows: list[dict], base_url: str = "", model: str = "", concurrency: int = 48, llm=None) -> dict[str, float | None]:
    from curatorkit.gates.reward import RewardGate
    from curatorkit.schema import DataSample

    if llm is None:
        from curatorkit.llm.litellm import LiteLLMBackend
        llm = LiteLLMBackend(model=f"openai/{model}", api_base=base_url, api_key="EMPTY", temperature=0.0,
                             max_tokens=200, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    gate = RewardGate(llm=llm, threshold=0.0, dimensions=["coherence", "truthfulness", "instruction_following"],
                      prompt_template=JUDGE_PROMPT, concurrency=concurrency, on_judge_error="pass")
    samples = [DataSample(id=r["id"], source_uri="desi-bank-toolcalls/curated", task_type="instruction_following",
                          instruction=context(r), output=r["messages"][-1]["content"]) for r in rows]
    passed, rejected = gate.run(samples)
    return {s.id: s.label for s in list(passed) + list(rejected)}


FLAGS = ["garbled", "wrong_language", "wrong_fact", "incomplete", "wrong_voice", "customer_nonsense"]
CHECKLIST_PROMPT = """You review one conversation of an Indian bank's support chatbot.

{context}

BOT FINAL REPLY: {reply}

Answer each question with true or false:
- garbled: does the reply OR the customer message contain a garbled, invented or clearly misspelled word?
- wrong_language: is the reply NOT written in the required language/style?
- wrong_fact: does the reply state a number, date, id or name that is missing from the tool results and the customer message, or contradict them?
- incomplete: does the reply fail to do the expected bot behaviour (e.g. leaves out a fact the customer asked for, or does not ask for the missing information)?
- wrong_voice: does the reply talk as if it were the customer instead of to the customer?
- customer_nonsense: does the customer message not make sense as a real customer message?

Respond with JSON only, for example:
{{"garbled": false, "wrong_language": false, "wrong_fact": false, "incomplete": false, "wrong_voice": false, "customer_nonsense": false, "note": "max 12 words"}}"""


async def _checklist_async(rows, base_url, model, concurrency):
    import asyncio

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=120)
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        prompt = CHECKLIST_PROMPT.format(context=context(r), reply=r["messages"][-1]["content"])
        err = ""
        async with sem:
            for _ in range(3):
                try:
                    out = await client.chat.completions.create(
                        model=model, temperature=0.0, max_tokens=150, messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    d = json.loads(out.choices[0].message.content)
                    flags = [f for f in FLAGS if d.get(f) is True]
                    return {"id": r["id"], "lang": r["lang"], "category": r["category"], "flags": flags,
                            "note": str(d.get("note", ""))[:120]}
                except Exception as e:  # noqa: BLE001
                    err = repr(e)[:120]
        return {"id": r["id"], "lang": r["lang"], "category": r["category"], "flags": None, "note": err}
    return await asyncio.gather(*(one(r) for r in rows))


def checklist(rows: list[dict], base_url: str, model: str, concurrency: int = 48) -> list[dict]:
    """Yes/no quality flags per episode (more reliable than a 0-1 score from a 14B judge)."""
    import asyncio
    return asyncio.run(_checklist_async(rows, base_url, model, concurrency))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/curated/curated.jsonl")
    p.add_argument("--out", default="data/curated/judge_scores.jsonl")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--mode", choices=["checklist", "score"], default="checklist")
    a = p.parse_args()
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.mode == "checklist":
        res = checklist(rows, a.base_url, a.model, a.concurrency)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res), encoding="utf-8")
        import collections
        c = collections.Counter()
        for x in res:
            c.update(x["flags"] if x["flags"] is not None else ["ERROR"])
        print(json.dumps({"n": len(res), "clean": sum(x["flags"] == [] for x in res), "flag_counts": dict(c)}))
        return
    scores = judge(rows, a.base_url, a.model, a.concurrency)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps({"id": r["id"], "lang": r["lang"], "category": r["category"],
                                               "score": scores.get(r["id"])}) + "\n" for r in rows), encoding="utf-8")
    vals = [v for v in scores.values() if v is not None]
    print(json.dumps({"n": len(rows), "scored": len(vals), "mean": round(sum(vals) / max(1, len(vals)), 3),
                      "lt_0.5": sum(v < 0.5 for v in vals), "lt_0.7": sum(v < 0.7 for v in vals)}))


if __name__ == "__main__":
    main()
