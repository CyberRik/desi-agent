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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/curated/curated.jsonl")
    p.add_argument("--out", default="data/curated/judge_scores.jsonl")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--concurrency", type=int, default=48)
    a = p.parse_args()
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = judge(rows, a.base_url, a.model, a.concurrency)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps({"id": r["id"], "lang": r["lang"], "category": r["category"],
                                               "score": scores.get(r["id"])}) + "\n" for r in rows), encoding="utf-8")
    vals = [v for v in scores.values() if v is not None]
    print(json.dumps({"n": len(rows), "scored": len(vals), "mean": round(sum(vals) / max(1, len(vals)), 3),
                      "lt_0.5": sum(v < 0.5 for v in vals), "lt_0.7": sum(v < 0.7 for v in vals)}))


if __name__ == "__main__":
    main()
