"""[R] Multi-turn language-switch eval: does the agent follow the customer into a new language on turn 2?

Found in the local playground: turn 1 in Banglish, turn 2 in Hinglish -> the fine-tuned model took the right action
(asked for the UPI id) but answered in Banglish. Training and the frozen test are single-turn, so this was never measured.

Episodes are built from the frozen test split (evaluation only, never training):
  prefix  = a full test conversation (customer turn, gold tool calls + results, gold reply) in language L1
  turn 2  = the customer message of another test conversation in language L2, scored against ITS gold calls
Two conditions, same targets: `switch` (L1 != L2) and `same` (L1 == L2, control).
Per episode we score turn 2 only: task success (contracts/METRICS.md, via score_episode on the turn-2 messages)
and `lang_match`: the final reply is in L2 by our data gates (script, romanization, no Hindi in Banglish / Bengali
in Hinglish, English not romanized Hindi/Bengali).

    python scripts/eval_multiturn.py --tag tinyaya-sft-a-final-multiturn --openai-base-url http://localhost:8001/v1 \
        --openai-model sft --n 300
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desi_agent import fmt  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402
from desi_agent.eval.language import lang_match  # noqa: E402
from desi_agent.eval.metrics import score_episode  # noqa: E402
from desi_agent.eval.runner import openai_policy  # noqa: E402

SEED = 1234
LANGS = ["en", "hi", "bn", "hi-en", "bn-en"]


def build(n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    test = load_split("test")
    by_lang = collections.defaultdict(list)
    for e in test:
        by_lang[e["lang"]].append(e)
    targets = rng.sample(test, min(n, len(test)))
    out = []
    for t in targets:
        for cond in ("switch", "same"):
            l1 = rng.choice([l for l in LANGS if l != t["lang"]]) if cond == "switch" else t["lang"]
            pool = [e for e in by_lang[l1] if e["id"] != t["id"]]
            pre = rng.choice(pool)
            out.append({"id": f"{cond}:{pre['id']}>{t['id']}", "condition": cond, "lang1": l1, "lang": t["lang"],
                        "category": t["category"], "prefix": pre["messages"], "turn2": t["messages"][0],
                        "gold": t["gold"]})
    return out


def run_one(policy, ep: dict) -> dict:
    start = [dict(m) for m in ep["prefix"]] + [dict(ep["turn2"])]
    run = fmt.run_episode(policy, start)
    turn2 = run.messages[len(ep["prefix"]):]  # customer turn 2 + everything the model did after it
    s = score_episode(turn2, ep["gold"])
    reply = turn2[-1].get("content") or "" if turn2[-1]["role"] == "assistant" else ""
    return {"id": ep["id"], "condition": ep["condition"], "lang1": ep["lang1"], "lang": ep["lang"],
            "category": ep["category"], "scores": s, "lang_match": float(lang_match(reply, ep["lang"])),
            "messages": turn2, "raw_outputs": run.raw_outputs}


def summarize(rows: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        return {"n": n, "task_success": round(sum(r["scores"]["task_success"] for r in rs) / max(1, n), 4),
                "lang_match": round(sum(r["lang_match"] for r in rs) / max(1, n), 4),
                "both": round(sum(r["scores"]["task_success"] * r["lang_match"] for r in rs) / max(1, n), 4),
                "false_action": round(sum(r["scores"]["false_action"] for r in rs) / max(1, n), 4)}
    out = {}
    for cond in ("switch", "same"):
        rs = [r for r in rows if r["condition"] == cond]
        g = collections.defaultdict(list)
        for r in rs:
            g[r["lang"]].append(r)
        out[cond] = {"overall": agg(rs), "by_target_lang": {k: agg(v) for k, v in sorted(g.items())}}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--openai-base-url", required=True)
    p.add_argument("--openai-model", required=True)
    p.add_argument("--n", type=int, default=300, help="target turn-2 conversations (each used in both conditions)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out-dir")
    a = p.parse_args()
    eps = build(a.n)
    policy = openai_policy(a.openai_base_url, a.openai_model)
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda e: run_one(policy, e), eps))
    out = Path(a.out_dir or ROOT / "results" / a.tag)
    out.mkdir(parents=True, exist_ok=True)
    (out / "multiturn.traces.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                 encoding="utf-8")
    res = {"model_tag": a.tag, "seed": SEED, "n_targets": a.n, **summarize(rows)}
    (out / "multiturn.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(a.tag, json.dumps({c: res[c]["overall"] for c in ("switch", "same")}))


if __name__ == "__main__":
    main()
