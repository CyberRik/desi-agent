"""[F] F5 go/no-go: does the SFT policy leave any GRPO signal on the train prompts that matter?

GRPO learns only from prompts whose sampled completions get *different* rewards. For the R7 failure
areas (clarify, block_card) we sample k first turns per **train** prompt at temperature T with the SFT
adapter, score each with `train.grpo.first_turn_reward` (= api.reward through the eval loop) and report
the share of prompts with reward variance, plus mean reward and greedy-equivalent stats.

    python scripts/train_grpo_probe.py --out outputs/grpo-probe [--k 8 --temp 1.0 --limit 0]
Runs one worker per visible GPU (sharded prompts). Details go to ToolW/desi-agent-training-logs runs/<out name>/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ADAPTER = "ToolW/desi-agent-tinyaya-sft-lora"


def pick(limit: int) -> list[dict]:
    from desi_agent.api import load_split
    from desi_agent.train.grpo import one_step
    eps = [e for e in load_split("train") if e["split"] == "train" and one_step(e)]
    keep = [e for e in eps if e["category"] == "clarify"
            or any(c["name"] == "block_card" for c in e["gold"]["tool_sequence"])]
    return keep[:limit] if limit else keep


def worker(a) -> None:
    import torch
    from desi_agent import fmt
    from desi_agent.train.chat import apply_template
    from desi_agent.train.grpo import first_turn_reward
    from desi_agent.train.policy import load_policy

    eps = pick(a.limit)[a.shard::a.nshards]
    pol = load_policy(adapter=ADAPTER)
    model, tok = pol.model, pol.tok
    rows, t0 = [], time.time()
    for i, e in enumerate(eps):
        first = [m for m in e["messages"] if m["role"] == "user"][:1]
        text = apply_template(tok, fmt.to_chat(first), add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=True, temperature=a.temp, top_p=1.0, num_return_sequences=a.k,
                                 max_new_tokens=a.max_new, stop_strings=["</tool_call>"], tokenizer=tok,
                                 pad_token_id=tok.pad_token_id)
        comps = [t.strip() for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        rs = [first_turn_reward(c, e) for c in comps]
        rows.append({"id": e["id"], "lang": e["lang"], "category": e["category"], "rewards": rs,
                     "samples": comps[:3]})
        if i % 20 == 0:
            print(f"[F] shard {a.shard}: {i + 1}/{len(eps)} {time.time() - t0:.0f}s", flush=True)
    Path(a.shard_out).write_text(json.dumps(rows, ensure_ascii=False))


def summarize(rows: list[dict], a) -> dict:
    from statistics import mean, pstdev

    def stats(rs):
        if not rs:
            return {"n": 0}
        var = [r for r in rs if pstdev(r["rewards"]) > 1e-9]
        return {"n": len(rs), "mean_reward": round(mean(mean(r["rewards"]) for r in rs), 4),
                "frac_prompts_with_signal": round(len(var) / len(rs), 4),
                "frac_samples_success": round(mean(sum(x >= 1.0 for x in r["rewards"]) / len(r["rewards"]) for r in rs), 4),
                "frac_prompts_all_wrong": round(sum(max(r["rewards"]) < 1.0 for r in rs) / len(rs), 4)}
    out = {"k": a.k, "temp": a.temp, "adapter": ADAPTER, "overall": stats(rows)}
    for key in ("category", "lang"):
        for v in sorted({r[key] for r in rows}):
            out[f"{key}={v}"] = stats([r for r in rows if r[key] == v])
    bc = [r for r in rows if r["category"] != "clarify"]
    out["block_card_single"] = stats(bc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard-out")
    a = ap.parse_args()
    if a.shard >= 0:
        return worker(a)

    import torch
    n = max(1, torch.cuda.device_count())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for s in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s))
        cmd = [sys.executable, __file__, "--out", a.out, "--k", str(a.k), "--temp", str(a.temp), "--max-new",
               str(a.max_new), "--limit", str(a.limit), "--shard", str(s), "--nshards", str(n),
               "--shard-out", str(out / f"shard{s}.json")]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker failed: {codes}")
    rows = [r for s in range(n) for r in json.loads((out / f"shard{s}.json").read_text())]
    summary = summarize(rows, a)
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    (out / "probe_rows.json").write_text(json.dumps(rows, ensure_ascii=False))
    print("[F] probe:", json.dumps(summary, indent=1), flush=True)
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(out), repo_id="ToolW/desi-agent-training-logs", repo_type="dataset",
                              path_in_repo=f"runs/{out.name}", allow_patterns=["manifest.json", "probe_rows.json"],
                              commit_message="[F] F5 GRPO signal probe")


if __name__ == "__main__":
    main()
