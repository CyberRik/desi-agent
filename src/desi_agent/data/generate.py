"""[R] R2: teacher generation. Scenarios (code) + surface text (teacher LLM) -> DATA_FORMAT rows.

Works with any OpenAI-compatible endpoint. Default plan: vLLM serving Qwen/Qwen3-14B-AWQ on Kaggle T4x2.

    python -m desi_agent.data.generate --base-url http://localhost:8000/v1 --model Qwen/Qwen3-14B-AWQ \
        --out data/raw/teacher.jsonl --concurrency 48

Resumable: rows already in --out (by id) are skipped. Rejected rows go to <out>.rejected.jsonl with reasons.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from .. import tools
from .checks import clean_teacher_text, final_problems, required_refs, user_msg_problems
from .scenarios import STYLE, TONES, Scenario, sample_all

USER_SYS = ("You write realistic chat messages that Indian bank customers send to their bank's support chatbot. "
            "Output ONLY the customer's message: no quotes, no labels, no explanations.")
FINAL_SYS = ("You are a friendly, concise support assistant of an Indian bank, replying TO a customer. Speak as the bank "
             "assistant and address the customer directly in the second person; never write as if you were the customer, "
             "never say 'my card' about the customer's card. Output ONLY your reply: no quotes, no labels, no explanations.")
# How to say "you" per language. bn-en MUST use Bengali forms: the teacher drifts into Hindi 'aap' otherwise.
ADDRESS = {"en": "you / your", "hi": "आप / आपका", "bn": "আপনি / আপনার", "hi-en": "aap / aapka (Hindi)",
           "bn-en": "apni / apnar (Bengali). NEVER use Hindi words like aap, aapka, hai, gaya, mein"}
# Banking terms the teacher mistranslates.
GLOSSARY = {"hi": "dispute = विवाद (or डिस्प्यूट); never बहस or बर्बरता.",
            "bn": "dispute = ডিসপিউট (or আপত্তি); never অপরাধ.",
            "hi-en": "Use only Hindi (not Bengali) words.", "bn-en": "Use only Bengali (not Hindi) words."}
REPLY_EXAMPLE = {
    "hi-en": "Aapka card ****1234 block kar diya gaya hai, reference number BLK12345678. Naya card 5 din mein aa jayega.",
    "bn-en": "Apnar card ****1234 block kore deoa hoyeche, reference number BLK12345678. Notun card 5 diner moddhe pouche jabe.",
}

NATIVE = {"hi": "Devanagari numerals (०-९)", "bn": "Bengali numerals (০-৯)"}


def user_prompt(s: Scenario, tone: str) -> str:
    lines = [f"Language and style: {STYLE[s.lang]}",
             f"Tone: {tone}.",
             f"The customer {s.intent.rstrip('.')}."]
    if s.must_include:
        lines.append("The message MUST contain these exactly as written: " + ", ".join(s.must_include) + ".")
        if s.native_digits:
            lines[-1] = ("The message MUST contain these values; write their digits using " + NATIVE[s.lang] + ": "
                         + ", ".join(s.must_include) + ".")
    if s.category == "clarify":
        lines.append("Do NOT include any card numbers, amounts, UPI ids, reference numbers, rates or durations.")
    if s.lang in GLOSSARY:
        lines.append(GLOSSARY[s.lang])
    lines.append("Write ONE message of 1-2 sentences (no alternatives, no line breaks). Vary the wording; don't start with a greeting every time.")
    return "\n".join(lines)


def final_prompt(s: Scenario, user_msg: str, results: list[tuple[str, dict, str]]) -> str:
    lines = [f"Reply in this language and style: {STYLE[s.lang]}", f"Customer message: {user_msg}",
             f"Address the customer as: {ADDRESS[s.lang]}. Do not copy their sentences."]
    if s.lang in GLOSSARY:
        lines.append(GLOSSARY[s.lang])
    if s.lang in REPLY_EXAMPLE:
        lines.append(f"Example of the reply style (different situation): {REPLY_EXAMPLE[s.lang]}")
    if s.category == "clarify":
        lines.append(f"You cannot act yet. Politely ask the customer for: {s.missing}. One or two sentences.")
    elif s.category == "no_tool":
        lines.append("No account action is needed. Answer briefly and helpfully. If the question is not about banking, "
                     "say politely that you can only help with banking (balances, transactions, cards, disputes, UPI, "
                     "transfers, branches/ATMs, EMI). Never ask for OTP/PIN/passwords.")
    else:
        lines.append("You already performed these actions; their results (JSON) are:")
        for name, args, res in results:
            lines.append(f"- {name}({json.dumps(args, ensure_ascii=False)}) -> {res}")
        lines.append("Tell the customer the outcome using the key facts from the results (amounts in ₹, reference "
                     "numbers). Do not invent anything not in the results. 1-3 sentences.")
    return "\n".join(lines)


def build_row(idx: int, s: Scenario, user_msg: str, final: str, results, teacher: str) -> dict:
    msgs = [{"role": "user", "content": user_msg}]
    for i, (name, args, res) in enumerate(results):
        cid = f"call_{i}"
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args, ensure_ascii=False, sort_keys=True)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "name": name, "content": res})
    msgs.append({"role": "assistant", "content": final})
    return {"id": f"{s.lang}-{s.category}-{s.pattern}-{idx:05d}", "lang": s.lang, "category": s.category,
            "split": "train", "source": f"teacher:{teacher}", "messages": msgs,
            "gold": {"tool_sequence": [{"name": n, "arguments": a} for n, a, _ in results],
                     "must_ask_clarification": s.category == "clarify"},
            "meta": {"pattern": s.pattern, "native_digits": s.native_digits}}


class Teacher:
    def __init__(self, base_url: str, model: str, api_key: str, concurrency: int):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in model.lower() else {}

    async def __call__(self, system: str, prompt: str, temperature: float, max_tokens: int = 200) -> str:
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model, temperature=temperature, max_tokens=max_tokens, extra_body=self.extra or None,
                        frequency_penalty=0.3, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
                    if r.choices[0].finish_reason == "length":
                        return ""  # cut off mid-sentence (often a repetition loop): reject and retry
                    return clean_teacher_text(r.choices[0].message.content or "")
                except Exception:  # noqa: BLE001 - network/rate-limit: back off and retry
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 ** attempt)
        return ""


def _latin_values(results) -> list[str]:
    """Strings from tool results (merchant names, ids, branch names) the reply may quote in Latin letters."""
    out: list[str] = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            out.append(v)
    for _, args, res in results:
        walk(args)
        walk(json.loads(res))
    return out


async def make_one(teacher: Teacher, idx: int, s: Scenario, rng: random.Random, tries: int = 3):
    tone = rng.choice(TONES)
    problems: list[str] = []
    for _ in range(tries):
        user_msg = await teacher(USER_SYS, user_prompt(s, tone), 0.9, 150)
        problems = user_msg_problems(user_msg, s.lang, s.must_include, s.forbid)
        if not problems:
            break
    else:
        return None, {"idx": idx, "stage": "user", "problems": problems, "text": user_msg, "scenario": s.__dict__}
    results = [(n, a, tools.execute(n, a)) for n, a in s.calls]
    final_tries = tries + 3 if s.lang in REPLY_EXAMPLE else tries  # code-mixed replies fail the gates more often
    for attempt in range(final_tries):
        final = await teacher(FINAL_SYS, final_prompt(s, user_msg, results), 0.4 + 0.1 * attempt, 300)
        problems = final_problems(final, s.lang, s.must_include + _latin_values(results), user_msg,
                                  required_refs([r for _, _, r in results]))
        if not problems:
            return build_row(idx, s, user_msg, final, results, teacher.model), None
    return None, {"idx": idx, "stage": "final", "problems": problems, "text": final, "scenario": s.__dict__}


async def run(args) -> None:
    scenarios = sample_all(seed=args.seed, counts=json.loads(args.counts) if args.counts else None,
                           langs=args.langs.split(","))
    if args.limit:
        random.Random(args.seed).shuffle(scenarios)
        scenarios = scenarios[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["meta"]["idx"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    teacher = Teacher(args.base_url, args.model, args.api_key, args.concurrency)
    todo = [(i, s) for i, s in enumerate(scenarios) if i not in done]
    print(f"{len(scenarios)} scenarios, {len(done)} already done, {len(todo)} to go")
    t0, ok, bad = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as fo, open(f"{out}.rejected.jsonl", "a", encoding="utf-8") as fr:
        tasks = [asyncio.create_task(make_one(teacher, i, s, random.Random(args.seed + i))) for i, s in todo]
        for fut in asyncio.as_completed(tasks):
            try:
                row, rej = await fut
            except Exception as e:  # noqa: BLE001
                row, rej = None, {"error": repr(e)}
            if row:
                row["meta"]["idx"] = int(row["id"].rsplit("-", 1)[1])
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += 1
            else:
                fr.write(json.dumps(rej, ensure_ascii=False, default=str) + "\n")
                bad += 1
            if (ok + bad) % 100 == 0:
                fo.flush()
                print(f"  {ok + bad}/{len(todo)}  kept={ok} rejected={bad}  {time.time() - t0:.0f}s", flush=True)
    manifest = {"out": str(out), "teacher": args.model, "seed": args.seed, "n_scenarios": len(scenarios),
                "kept_this_run": ok, "rejected_this_run": bad, "seconds": round(time.time() - t0)}
    Path(f"{out}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--out", default="data/raw/teacher.jsonl")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--langs", default="en,hi,bn,hi-en,bn-en")
    p.add_argument("--counts", help='JSON override, e.g. {"single_tool": 5}')
    p.add_argument("--limit", type=int, help="random subset of scenarios (smoke tests)")
    p.add_argument("--concurrency", type=int, default=32)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
