"""[R] RL signal probe for multi-turn language following, scored with rewards.reward_lang.

GRPO (and rejection-sampling fine-tuning) only learn from prompts whose sampled trajectories get DIFFERENT rewards.
K's probe on single-turn train prompts found almost none (95.7% of samples already right). Here the prompts are
multi-turn language switches built from the TRAIN split only (never test): prefix = a full train conversation in
language L1, turn 2 = another train conversation's customer message in L2 != L1, gold = its gold calls.

For each prompt we sample K full trajectories (temperature 1.0, the same agent loop + mock bank as eval) and report:
mean reward, share of prompts with reward variance, lang_match rate. All trajectories are saved with their reward,
so the best ones can be used for rejection-sampling fine-tuning if the signal is there.

    python scripts/eval_rl_signal.py --openai-base-url http://localhost:8001/v1 --openai-model sft --n 300 --k 8 \
        --out results/rl_signal
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desi_agent import fmt  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402
from desi_agent.eval.language import final_reply, lang_match  # noqa: E402
from desi_agent.eval.rewards import reward_lang  # noqa: E402
from desi_agent.eval.runner import openai_policy  # noqa: E402

SEED = 1234


def build_train_switch(n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    train = [e for e in load_split("train") if e["split"] == "train"]
    by_lang = collections.defaultdict(list)
    for e in train:
        by_lang[e["lang"]].append(e)
    out = []
    for t in rng.sample(train, n):
        l1 = rng.choice([l for l in by_lang if l != t["lang"]])
        pre = rng.choice(by_lang[l1])
        out.append({"id": f"switch:{pre['id']}>{t['id']}", "lang1": l1, "lang": t["lang"], "category": t["category"],
                    "prefix": pre["messages"], "turn2": t["messages"][0], "gold": t["gold"], "target": t["id"]})
    return out


def rollout(policy, ep):
    run = fmt.run_episode(policy, [dict(m) for m in ep["prefix"]] + [dict(ep["turn2"])])
    turn2 = run.messages[len(ep["prefix"]):]
    return {"messages": turn2, "reward": round(reward_lang(turn2, ep["gold"], ep["lang"]), 4),
            "lang_match": float(lang_match(final_reply(turn2), ep["lang"]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--openai-base-url", required=True)
    p.add_argument("--openai-model", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", default="results/rl_signal")
    a = p.parse_args()
    eps = build_train_switch(a.n)
    policy = openai_policy(a.openai_base_url, a.openai_model, temperature=a.temperature)
    jobs = [(e, j) for e in eps for j in range(a.k)]
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(lambda x: rollout(policy, x[0]), jobs))
    groups = collections.defaultdict(list)
    for (e, _), o in zip(jobs, outs):
        groups[e["id"]].append(o)
    rows = [{"id": e["id"], "lang": e["lang"], "lang1": e["lang1"], "category": e["category"], "target": e["target"],
             "prefix": e["prefix"], "gold": e["gold"], "samples": groups[e["id"]]} for e in eps]
    rs = [[s["reward"] for s in r["samples"]] for r in rows]
    summary = {"n_prompts": len(rows), "k": a.k, "temperature": a.temperature, "split": "train",
               "mean_reward": round(statistics.mean(x for g in rs for x in g), 4),
               "frac_prompts_with_variance": round(sum(statistics.pstdev(g) > 1e-9 for g in rs) / len(rs), 4),
               "mean_within_prompt_std": round(statistics.mean(statistics.pstdev(g) for g in rs), 4),
               "lang_match_rate": round(statistics.mean(s["lang_match"] for r in rows for s in r["samples"]), 4),
               "frac_prompts_best_beats_mean_by_0.3": round(sum(max(g) - statistics.mean(g) >= 0.3 for g in rs) / len(rs), 4)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
