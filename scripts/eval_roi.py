"""[R] R8: ROI table, closed API vs self-hosted fine-tuned Tiny Aya. Writes results/roi.json + a markdown table.

Measured (results/tinyaya-sft-a-final/test.traces.jsonl, vLLM fp16 on ONE T4, 32 concurrent conversations):
  latency per conversation, model calls per conversation, prompt/output tokens per conversation.
Assumed (stated in the output): API prices, GPU price, volume, output tokens for API models.

    python scripts/eval_roi.py --out results/roi.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Claude API list prices, $ per 1M tokens (Anthropic pricing table, cached 2026-11-03).
API = {"Claude Haiku 5.5": {"in": 0.10, "out": 0.50, "cache_read": 0.01},
       "Claude Sonnet 5.5": {"in": 2.00, "out": 10.00, "cache_read": 0.10},
       "Claude Opus 5.5": {"in": 4.00, "out": 20.00, "cache_read": 0.20}}
T4_ON_DEMAND = 0.35       # $/GPU-hour, Google Cloud Compute Engine T4, US regions (on-demand list price)
T4_ALL_IN = 0.50          # $/hour incl. a small VM for CPU/RAM (conservative)
SYSTEM_TOKENS = 1278      # measured: fixed system prompt (tool schemas + chat-template preamble), cacheable
TOKENS_PER_CALL = 1491    # measured: mean prompt tokens per model call
API_OUT_TOKENS = 150      # assumed per conversation for API models (ours averages ~70; give the API room)
PEAK_FACTOR = 2.0         # provision self-hosted GPUs for 2x the average hourly load (traffic peaks)
VOLUMES = {"pilot (10k conversations/day)": 10_000, "mid-size bank (100k/day)": 100_000, "large bank (1M/day)": 1_000_000}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/roi.json")
    a = p.parse_args()
    T = [json.loads(l) for l in (ROOT / "results/tinyaya-sft-a-final/test.traces.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = statistics.mean(len(t["raw_outputs"]) for t in T)
    lat = statistics.mean(t["latency_s"] for t in T)
    conv_per_hour_t4 = 32 / lat * 3600                    # 32 concurrent conversations on one T4
    in_tok = calls * TOKENS_PER_CALL
    cached = calls * SYSTEM_TOKENS
    per_m = {}
    for name, pr in API.items():
        no_cache = (in_tok * pr["in"] + API_OUT_TOKENS * pr["out"])            # $ per 1M conversations
        cache = (cached * pr["cache_read"] + (in_tok - cached) * pr["in"] + API_OUT_TOKENS * pr["out"])
        per_m[name] = {"no_cache": round(no_cache, 2), "with_cache": round(cache, 2)}
    t4_hours_per_m = 1_000_000 / conv_per_hour_t4
    per_m["Desi Agent (Tiny Aya 3.35B + LoRA, self-hosted T4)"] = {
        "no_cache": round(t4_hours_per_m * T4_ALL_IN, 2), "with_cache": round(t4_hours_per_m * T4_ALL_IN, 2)}
    monthly = {}
    for vname, per_day in VOLUMES.items():
        conv_month = per_day * 30
        row = {k: round(v["with_cache"] * conv_month / 1e6) for k, v in per_m.items()}
        gpus = per_day / 24 * PEAK_FACTOR / conv_per_hour_t4
        row["Desi Agent (Tiny Aya 3.35B + LoRA, self-hosted T4)"] = round(max(1, -(-gpus // 1)) * 24 * 30 * T4_ALL_IN)
        row["_t4_gpus_needed_24x7"] = max(1, -(-gpus // 1))
        monthly[vname] = row
    res = {"measured": {"model_calls_per_conversation": round(calls, 2), "mean_latency_s_at_32_concurrent": round(lat, 2),
                        "conversations_per_hour_per_T4": round(conv_per_hour_t4), "prompt_tokens_per_conversation": round(in_tok),
                        "cacheable_tokens_per_conversation": round(cached)},
           "assumptions": {"api_prices_per_1M_tokens": API, "t4_on_demand_per_hour": T4_ON_DEMAND,
                           "t4_all_in_per_hour": T4_ALL_IN, "api_output_tokens_per_conversation": API_OUT_TOKENS,
                           "self_hosting_capacity": "provisioned 24x7 in whole T4s for 2x average load (no autoscaling)",
                           "training_cost": "2.2 h on free Kaggle T4 x2 (~$1.5 at $0.35/T4-h)"},
           "usd_per_1M_conversations": per_m, "usd_per_month": monthly}
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["measured"]))
    for k, v in per_m.items():
        print(f"{k:52} per 1M conversations: ${v['with_cache']:>9,.0f} (no cache ${v['no_cache']:,.0f})")
    for vname, row in monthly.items():
        print(vname, {k[:22]: v for k, v in row.items()})


if __name__ == "__main__":
    main()
