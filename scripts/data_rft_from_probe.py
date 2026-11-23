"""[R] Rejection-sampling fine-tuning (RFT) data from the RL signal probe.

For every TRAIN-split prompt in the probe's samples.jsonl, keep the highest-reward sampled trajectory if it is fully
correct under rewards.reward_lang (task success, reply in the customer's latest language, not an echo, an 'ask first'
reply that asks). It becomes a training episode: prefix conversation (empty for single-turn grounding prompts) + the
customer turn + the model's own successful trajectory. These are appended to the base training set (so nothing is
forgotten), and rows whose rendered length would exceed the training max_len are dropped (they would be truncated).

    # round 1 (run rft-a): base = the original train split
    python scripts/data_rft_from_probe.py --samples results/rl_signal/samples.jsonl --out data/rl/train_rft.jsonl
    # round 2: probe with the rft-a adapter on data/round2 prompts; base = rft-a's own training file
    python scripts/data_rft_from_probe.py --samples results/rl_signal_round2/samples.jsonl \
        --base data/rl/train_rft.jsonl --id-prefix rft2- --source rft2:tinyaya-rft-a@reward_lang \
        --grounding-gold data/round2/grounding_train.jsonl --out data/round2/rft/train_rft2.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desi_agent import fmt  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402
from desi_agent.eval.language import final_reply  # noqa: E402

FULL = 1.0 + 0.3 + 0.3  # success + schema_valid + language bonus, no penalties

# Grounding 'missing' prompts: reward_lang only checks that the reply asks something. A kept reply must also name the
# missing identifier. Latin-script slot words per group (they match 100% of the gold clarify replies in en/hi-en/bn-en
# and some of hi/bn); for hi/bn we add the words that are distinctive for the group in that language's gold replies
# (in >= 30% of the group's gold replies, <= 15% of any other group's), learned from data/round2/grounding_train.jsonl.
SLOT = {
    "block": r"last\s*4|last four|4[ -]digit|\b4\b|card number|card ending",
    "dispute": r"transaction|txn|\bID\b|reference",
    "upi": r"UPI|\bref|reference|transaction|\bID\b|UTR",
    "transfer_no_amount": r"amount|how much|kitna|kitne|koto|rupee|rupaye|taka|₹|\brs\b",
    "transfer_no_upi": r"UPI|\bID\b|recipient|payee|whom|kisko|jinko|kake|jake|VPA|@",
}
TOKEN = re.compile(r"[\w@₹]+", re.U)


def slot_words(gold_path: str) -> dict:
    miss = [json.loads(l) for l in Path(gold_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    miss = [r for r in miss if r.get("meta", {}).get("variant") == "missing" and r["lang"] in ("hi", "bn")]
    df, n = collections.defaultdict(collections.Counter), collections.Counter()
    for r in miss:
        k = (r["lang"], r["meta"]["group"])
        n[k] += 1
        df[k].update({t.lower() for t in TOKEN.findall(r["messages"][-1]["content"] or "")})
    out = {}
    for k, c in df.items():
        others = [o for o in df if o[0] == k[0] and o != k]
        out[k] = {t for t, f in c.items() if f / n[k] >= 0.3 and max((df[o][t] / n[o] for o in others), default=0) <= 0.15}
    return out


def names_slot(reply: str, lang: str, group: str, words: dict) -> bool:
    if re.search(SLOT[group], reply or "", flags=re.I):
        return True
    return bool({t.lower() for t in TOKEN.findall(reply or "")} & words.get((lang, group), set()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="results/rl_signal/samples.jsonl")
    p.add_argument("--out", default="data/rl/train_rft.jsonl")
    p.add_argument("--base", default="split:train", help="'split:train' (original train split) or a JSONL file")
    p.add_argument("--id-prefix", default="rft-")
    p.add_argument("--source", default="rft:tinyaya-sft-a-final@reward_lang")
    p.add_argument("--max-tokens", type=int, default=1950)
    p.add_argument("--grounding-gold", default=None,
                   help="round-2 grounding train file: enables the 'reply names the missing identifier' check")
    a = p.parse_args()
    words = slot_words(a.grounding_gold) if a.grounding_gold else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    rows = [json.loads(l) for l in Path(a.samples).read_text(encoding="utf-8").splitlines() if l.strip()]
    test_ids = {e["id"] for e in load_split("test")}
    rft, stats = [], {"prompts": len(rows), "with_full_reward_sample": 0, "too_long": 0}
    kept = collections.Counter()
    for r in rows:
        assert r["target"] not in test_ids, "probe prompts must come from the train split"
        full = [s for s in r["samples"] if s["reward"] >= FULL - 1e-6]
        if not full:
            continue
        stats["with_full_reward_sample"] += 1
        pm = r.get("meta") or {}
        if words is not None and r.get("set") == "grounding" and pm.get("variant") == "missing":
            full = [s for s in full if names_slot(final_reply(s["messages"]), r["lang"], pm["group"], words)]
            if not full:
                stats["grounding_no_slot_named"] = stats.get("grounding_no_slot_named", 0) + 1
                continue
        best = full[0]
        msgs = [dict(m) for m in r["prefix"]] + best["messages"]
        text = tok.apply_chat_template(fmt.to_chat(msgs), tokenize=False)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > a.max_tokens:
            stats["too_long"] += 1
            continue
        # DATA_FORMAT describes the whole conversation: gold = every call in order (prefix gold + turn 2), category
        # derived from it. Turn 2's own gold/category are kept in meta (that is what reward_lang scored).
        calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                 for m in msgs for c in (m.get("tool_calls") or [])]
        cat = r["category"] if not calls else ("single_tool" if len(calls) == 1 else "multi_tool")
        meta = {"lang1": r["lang1"], "reward": best["reward"], "turn2_category": r["category"], "turn2_gold": r["gold"]}
        if "set" in r:
            meta |= {"probe_set": r["set"], "probe_meta": r.get("meta", {}), "prompt_id": r["id"],
                     "n_full_samples": sum(s["reward"] >= FULL - 1e-6 for s in r["samples"])}
        rft.append({"id": a.id_prefix + r["id"].replace(":", "-").replace(">", "-"), "lang": r["lang"],
                    "category": cat, "split": "train", "source": a.source,
                    "messages": msgs, "gold": {"tool_sequence": calls, "must_ask_clarification": cat == "clarify"},
                    "meta": meta})
        kept[(r.get("set", "langswitch"), r["lang"], r["category"])] += 1
    if a.base == "split:train":
        base = [e for e in load_split("train") if e["split"] == "train"]
    else:
        base = [json.loads(l) for l in Path(a.base).read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [e["id"] for e in base + rft]
    assert len(ids) == len(set(ids)), "duplicate ids between base and RFT rows"
    assert all(e["split"] == "train" for e in base + rft)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in base + rft), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    def by(i):
        c = collections.Counter()
        for k, n in kept.items():
            c[k[i]] += n
        return dict(sorted(c.items()))
    stats |= {"rft_added": len(rft), "base": a.base, "base_rows": len(base), "total": len(base) + len(rft),
              "git_sha": sha, "samples": a.samples, "id_prefix": a.id_prefix, "source": a.source,
              "kept_by_set": by(0), "kept_by_lang": by(1), "kept_by_turn2_category": by(2),
              "kept_by_set_lang_category": {"|".join(k): n for k, n in sorted(kept.items())}}
    Path(f"{out}.manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
