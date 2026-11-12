"""[R] Reward-hacking audit: can a degenerate policy score well under the RL reward?

Scores scripted policies (no model) on the frozen test split (single-turn) and on the multi-turn language-switch set
(scripts/eval_multiturn.py), under the contract reward (`metrics.reward`) and the experimental `rewards.reward_lang`.
A reward is safe to optimise if every hack scores far below doing the task right (`gold`).

Policies:
  gold            the gold trajectory (upper bound)
  gold_wrong_lang gold actions, but the final reply in the previous turn's language (the playground failure)
  ask_en          never acts; always asks for more details in English
  ask_lang        never acts; always asks in the customer's language
  tool_spam       calls a harmless tool (balance), then asks in the customer's language
  echo            never acts; repeats the customer's message back

    python scripts/eval_reward_audit.py --out results/reward_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from desi_agent import fmt  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402
from desi_agent.eval.language import lang_match  # noqa: E402
from desi_agent.eval.metrics import reward  # noqa: E402
from desi_agent.eval.rewards import reward_lang  # noqa: E402
from eval_multiturn import build  # noqa: E402

ASK = {"en": "Could you share a few more details so I can help?",
       "hi": "कृपया थोड़ी और जानकारी दें ताकि मैं आपकी मदद कर सकूं।",
       "bn": "আরও একটু তথ্য দিন, তাহলে আমি আপনাকে সাহায্য করতে পারব।",
       "hi-en": "Aap thodi aur details bataiye please, tabhi main aapki madad kar paunga.",
       "bn-en": "Apni aro kichu details bolun please, tahole ami apnake help korte parbo."}


def scripted(kind: str, lang: str, user: str):
    state = {"n": 0}

    def policy(chat):
        state["n"] += 1
        if kind == "ask_en":
            return ASK["en"]
        if kind == "ask_lang":
            return ASK[lang]
        if kind == "echo":
            return user
        if kind == "tool_spam":
            return fmt.render_tool_call("get_account_balance", {"account_type": "savings"}) if state["n"] == 1 else ASK[lang]
        raise ValueError(kind)
    return policy


def episodes():
    single = [{"prefix": [], "turn2": e["messages"][0], "gold": e["gold"], "lang": e["lang"],
               "gold_msgs": e["messages"], "prev_reply": None} for e in load_split("test")]
    test = {e["id"]: e for e in load_split("test")}
    multi = []
    for m in build(300):
        if m["condition"] != "switch":
            continue
        tgt = test[m["id"].split(">")[1]]
        multi.append({"prefix": m["prefix"], "turn2": m["turn2"], "gold": m["gold"], "lang": m["lang"],
                      "gold_msgs": tgt["messages"], "prev_reply": m["prefix"][-1]["content"]})
    return {"single_turn_test": single, "multiturn_switch": multi}


def score(ep, kind):
    if kind.startswith("gold"):
        msgs = [dict(m) for m in ep["gold_msgs"]]
        if kind == "gold_wrong_lang":
            if not ep["prev_reply"]:
                return None
            msgs[-1] = {"role": "assistant", "content": ep["prev_reply"]}
    else:
        run = fmt.run_episode(scripted(kind, ep["lang"], ep["turn2"]["content"]), ep["prefix"] + [ep["turn2"]])
        msgs = run.messages[len(ep["prefix"]):]
    return reward(msgs, ep["gold"]), reward_lang(msgs, ep["gold"], ep["lang"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/reward_audit.json")
    a = p.parse_args()
    assert all(lang_match(t, l) for l, t in ASK.items()), "canned asks must be in their language"
    res = {}
    for name, eps in episodes().items():
        res[name] = {}
        for kind in ("gold", "gold_wrong_lang", "ask_en", "ask_lang", "tool_spam", "echo"):
            sc = [s for s in (score(e, kind) for e in eps) if s is not None]
            if sc:
                res[name][kind] = {"n": len(sc), "reward": round(sum(x for x, _ in sc) / len(sc), 3),
                                   "reward_lang": round(sum(y for _, y in sc) / len(sc), 3)}
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for name, r in res.items():
        print(name)
        for kind, v in r.items():
            print(f"  {kind:16} reward {v['reward']:6.3f}   reward_lang {v['reward_lang']:6.3f}   (n={v['n']})")


if __name__ == "__main__":
    main()
