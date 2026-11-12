"""[R] evaluate(policy, split, model_tag) -> results dict (+ files). contracts/API.md, METRICS.md.

    from desi_agent.api import evaluate, load_policy
    evaluate(load_policy(), split="test", model_tag="tinyaya-global-base")

CLI (any OpenAI-compatible server, e.g. vLLM, for the teacher / reference runs):
    python -m desi_agent.eval.runner --split test --model-tag teacher \
        --openai-base-url http://localhost:8000/v1 --openai-model google/gemma-3-12b-it --workers 16
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import fmt
from ..data.splits import ROOT, load_split
from .metrics import aggregate, score_episode

SEED = 1234


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def _one(policy, ep: dict, max_steps: int) -> dict:
    t0 = time.time()
    run = fmt.run_episode(policy, [ep["messages"][0]], max_steps=max_steps)
    return {
        "id": ep["id"], "lang": ep["lang"], "category": ep["category"], "split": ep["split"],
        "source": ep.get("source"), "gold": ep["gold"], "messages": run.messages,
        "raw_outputs": run.raw_outputs, "scores": score_episode(run.messages, ep["gold"]),
        "latency_s": round(time.time() - t0, 3),
    }


def evaluate(policy, split: str = "test", model_tag: str = "unnamed", *, limit: int | None = None,
             episodes: list[dict] | None = None, max_steps: int = 6, workers: int = 1,
             out_dir: str | Path | None = None, model_id: str | None = None, adapter: str | None = None,
             decoding: dict | None = None, auditkit: bool = True) -> dict:
    """Run every episode through the shared agent loop, score it, write results + traces.

    workers>1 calls the policy from threads; use it only for thread-safe policies (HTTP/vLLM servers)."""
    eps = episodes if episodes is not None else load_split(split)
    if limit:
        eps = eps[:limit]
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            traces = list(ex.map(lambda e: _one(policy, e, max_steps), eps))
    else:
        traces = [_one(policy, e, max_steps) for e in eps]

    by_lang, by_cat = defaultdict(list), defaultdict(list)
    for t in traces:
        by_lang[t["lang"]].append(t["scores"])
        by_cat[t["category"]].append(t["scores"])

    out = Path(out_dir) if out_dir else ROOT / "results" / model_tag
    out.mkdir(parents=True, exist_ok=True)
    traces_path = out / f"{split}.traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    results = {
        "model_tag": model_tag, "model_id": model_id, "adapter": adapter, "split": split,
        "n": len(traces), "git_sha": _git_sha(), "seed": SEED,
        "decoding": decoding or {"temperature": 0.0, "max_new_tokens": 256}, "max_steps": max_steps,
        "overall": aggregate([t["scores"] for t in traces]),
        "by_lang": {k: aggregate(v) for k, v in sorted(by_lang.items())},
        "by_category": {k: aggregate(v) for k, v in sorted(by_cat.items())},
        "mean_latency_s": round(sum(t["latency_s"] for t in traces) / max(1, len(traces)), 3),
        "traces_path": str(traces_path.relative_to(ROOT)) if traces_path.is_relative_to(ROOT) else str(traces_path),
    }
    if auditkit:
        try:
            from .auditkit_report import auditkit_eval
            results["auditkit"] = auditkit_eval(traces, out / f"{split}.auditkit.json")
        except ImportError as e:  # auditkit not installed: our own metrics still stand
            results["auditkit"] = {"skipped": str(e)}
    (out / f"{split}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def openai_policy(base_url: str, model: str, api_key: str = "EMPTY", temperature: float = 0.0, max_tokens: int = 256,
                  stop_at_tool_call: bool = True, no_thinking: bool = False):
    """Policy backed by any OpenAI-compatible chat endpoint (vLLM, Gemini, Groq, ...). Thread-safe.

    stop_at_tool_call mirrors train.policy.DEFAULT_GEN (generation stops after `</tool_call>`, the tag is kept so
    fmt.parse_assistant sees a complete call). no_thinking turns off Qwen3-style reasoning via the chat template."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)
    extra: dict = {}
    if stop_at_tool_call:
        extra["include_stop_str_in_output"] = True  # vLLM extension; other servers ignore unknown fields or strip it
    if no_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}

    def policy(chat: list[dict]) -> str:
        r = client.chat.completions.create(model=model, messages=chat, temperature=temperature, max_tokens=max_tokens,
                                           stop=["</tool_call>"] if stop_at_tool_call else None,
                                           extra_body=extra or None)
        text = r.choices[0].message.content or ""
        if stop_at_tool_call and "<tool_call>" in text and "</tool_call>" not in text \
                and r.choices[0].finish_reason == "stop":
            text += "</tool_call>"  # server stripped the stop string
        return text.strip()
    return policy


def print_table(results: dict) -> None:
    cols = ("task_success", "tool_selection_correct", "args_exact", "schema_valid", "false_action", "redundant_calls", "n")
    print(f"{'slice':<14}" + "".join(f"{c[:12]:>13}" for c in cols))
    rows = [("overall", results["overall"])] + [(f"lang={k}", v) for k, v in results["by_lang"].items()] \
        + [(f"cat={k}", v) for k, v in results["by_category"].items()]
    for name, agg in rows:
        print(f"{name:<14}" + "".join(f"{agg[c]:>13}" for c in cols))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--model-tag", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--openai-base-url")
    p.add_argument("--openai-model")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--gold", action="store_true", help="sanity check: replay gold (should score ~1.0)")
    p.add_argument("--no-thinking", action="store_true", help="disable Qwen3-style thinking (teacher runs)")
    p.add_argument("--adapter", help="recorded in results (the LoRA name served by vLLM --lora-modules)")
    p.add_argument("--model-id", help="base model id recorded in results (defaults to --openai-model)")
    p.add_argument("--out-dir")
    a = p.parse_args()
    if a.gold:
        from ..api import DummyPolicy
        eps = load_split(a.split)[: a.limit] if a.limit else load_split(a.split)
        traces = [_one(DummyPolicy(e), e, 6) for e in eps]
        print(json.dumps(aggregate([t["scores"] for t in traces]), indent=2))
        return
    policy = openai_policy(a.openai_base_url, a.openai_model, a.api_key, no_thinking=a.no_thinking)
    res = evaluate(policy, a.split, a.model_tag, limit=a.limit, workers=a.workers,
                   model_id=a.model_id or a.openai_model, adapter=a.adapter, out_dir=a.out_dir,
                   decoding={"temperature": 0.0, "max_new_tokens": 256, "stop": ["</tool_call>"],
                             "no_thinking": a.no_thinking})
    print_table(res)


if __name__ == "__main__":
    main()
