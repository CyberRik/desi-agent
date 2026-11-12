"""[R] Rejection-sampling fine-tuning (RFT) data from the RL signal probe.

For every TRAIN-split multi-turn prompt in results/rl_signal/samples.jsonl, keep the highest-reward sampled trajectory
if it is fully correct under rewards.reward_lang (task success, reply in the customer's latest language, not an echo,
an 'ask first' reply that asks). It becomes a multi-turn training episode: prefix conversation + turn 2 + the model's
own successful turn-2 trajectory. These are appended to the original train split (so nothing is forgotten), and rows
whose rendered length would exceed the training max_len are dropped (they would be truncated mid-reply).

    python scripts/data_rft_from_probe.py --samples results/rl_signal/samples.jsonl --out data/rl/train_rft.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desi_agent import fmt  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402

FULL = 1.0 + 0.3 + 0.3  # success + schema_valid + language bonus, no penalties


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="results/rl_signal/samples.jsonl")
    p.add_argument("--out", default="data/rl/train_rft.jsonl")
    p.add_argument("--max-tokens", type=int, default=1950)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    rows = [json.loads(l) for l in Path(a.samples).read_text(encoding="utf-8").splitlines() if l.strip()]
    test_ids = {e["id"] for e in load_split("test")}
    rft, stats = [], {"prompts": len(rows), "with_full_reward_sample": 0, "too_long": 0}
    for r in rows:
        assert r["target"] not in test_ids, "probe prompts must come from the train split"
        best = max(r["samples"], key=lambda s: s["reward"])
        if best["reward"] < FULL - 1e-6:
            continue
        stats["with_full_reward_sample"] += 1
        msgs = [dict(m) for m in r["prefix"]] + best["messages"]
        text = tok.apply_chat_template(fmt.to_chat(msgs), tokenize=False)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > a.max_tokens:
            stats["too_long"] += 1
            continue
        rft.append({"id": "rft-" + r["id"].replace(":", "-").replace(">", "-"), "lang": r["lang"],
                    "category": r["category"], "split": "train", "source": "rft:tinyaya-sft-a-final@reward_lang",
                    "messages": msgs, "gold": r["gold"], "meta": {"lang1": r["lang1"], "reward": best["reward"]}})
    base = [e for e in load_split("train") if e["split"] == "train"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in base + rft), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    stats |= {"rft_added": len(rft), "train_original": len(base), "total": len(base) + len(rft), "git_sha": sha,
              "samples": a.samples}
    Path(f"{out}.manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
