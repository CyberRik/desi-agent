"""[R] Robustness evals beyond the frozen test: reworded customer messages (OOD) and a perturbed mock bank.

--episodes FILE   evaluate on another episode file (e.g. data/test_ood.jsonl from data_ood_reword.py)
--perturb         the mock bank returns shifted balances / transaction amounts and remapped bank-issued ids
                  (transaction ids in statements, block refs, dispute ids, transfer UPI refs). Gold dispute ids that
                  come from a statement are remapped the same way, so reading the receipt is still scored correct.
                  Reports `stale_value_rate`: final replies quoting an ORIGINAL bank value that this run never
                  returned, i.e. answering from memory of the training data instead of from the tool result.

    python scripts/eval_variants.py --tag tinyaya-sft-a-final-perturbed --perturb --openai-base-url http://localhost:8001/v1 \
        --openai-model sft --model-id CohereLabs/tiny-aya-global --adapter ToolW/desi-agent-tinyaya-sft-lora
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desi_agent import tools  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402
from desi_agent.eval.runner import evaluate, openai_policy  # noqa: E402

_ORIG_EXECUTE = tools.execute
ID_RX = re.compile(r"^(TXN|BLK|DSP)(\d+)$")
VAL_RX = re.compile(r"TXN\d+|BLK\d+|DSP\d+|\d[\d,]*\.\d+|\d{6,}")


def remap_id(s: str) -> str:
    m = ID_RX.match(s)
    if m:
        d = m.group(2)
        return m.group(1) + str((int(d) * 7 + 123457) % 10 ** len(d)).zfill(len(d))
    if s.isdigit() and len(s) == 12:  # bank-issued UPI ref of a transfer
        return str((int(s) * 7 + 123457) % 10 ** 12).zfill(12)
    return s


def shift_amount(x: float) -> float:
    return round(x * 1.37 + 11.11, 2)


def perturbed_execute(name: str, args: dict) -> str:
    out = _ORIG_EXECUTE(name, args)
    try:
        r = json.loads(out)
    except ValueError:
        return out
    if "error" in r:
        return out
    if name == "get_account_balance":
        for k in ("balance_inr", "outstanding_inr", "available_limit_inr"):
            if k in r:
                r[k] = shift_amount(r[k])
    elif name == "get_recent_transactions":
        for t in r.get("transactions", []):
            t["amount_inr"] = shift_amount(t["amount_inr"])
            t["transaction_id"] = remap_id(t["transaction_id"])
    elif name == "block_card":
        r["reference"] = remap_id(r["reference"])
    elif name == "raise_dispute":
        r["dispute_id"] = remap_id(r["dispute_id"])
    elif name == "transfer_money":
        r["upi_ref"] = remap_id(r["upi_ref"])
    return json.dumps(r, ensure_ascii=False, sort_keys=True)


def perturb_gold(ep: dict) -> dict:
    ep = copy.deepcopy(ep)
    seq = ep["gold"]["tool_sequence"]
    names = [g["name"] for g in seq]
    if "get_recent_transactions" in names:  # dispute target found in the statement -> its id is bank-issued
        for g in seq:
            if g["name"] == "raise_dispute":
                g["arguments"]["transaction_id"] = remap_id(g["arguments"]["transaction_id"])
    return ep


def norm_vals(s: str) -> set[str]:
    return {v.replace(",", "") for v in VAL_RX.findall(s)}


def stale_rate(traces_path: Path) -> float:
    """Share of final replies quoting an original (unperturbed) bank value that this episode's tool results lacked."""
    n = stale = 0
    for l in traces_path.read_text(encoding="utf-8").splitlines():
        t = json.loads(l)
        n += 1
        seen, orig = set(norm_vals(t["messages"][0]["content"])), set()
        for m in t["messages"]:
            if m["role"] == "tool":
                seen |= norm_vals(m["content"])
            for c in m.get("tool_calls") or []:
                try:
                    orig |= norm_vals(_ORIG_EXECUTE(c["function"]["name"], json.loads(c["function"]["arguments"])))
                except Exception:  # noqa: BLE001 - invalid calls
                    pass
        reply = norm_vals(t["messages"][-1].get("content") or "")
        stale += bool((reply & orig) - seen)
    return stale / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--episodes", help="JSONL episode file (default: frozen test split)")
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--openai-base-url", required=True)
    p.add_argument("--openai-model", required=True)
    p.add_argument("--model-id")
    p.add_argument("--adapter")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out-dir")
    a = p.parse_args()
    eps = ([json.loads(l) for l in Path(a.episodes).read_text(encoding="utf-8").splitlines() if l.strip()]
           if a.episodes else load_split("test"))
    if a.perturb:
        tools.execute = perturbed_execute
        eps = [perturb_gold(e) for e in eps]
    out_dir = Path(a.out_dir or ROOT / "results" / a.tag)
    res = evaluate(openai_policy(a.openai_base_url, a.openai_model), "test", a.tag, episodes=eps, workers=a.workers,
                   out_dir=out_dir, model_id=a.model_id or a.openai_model, adapter=a.adapter,
                   decoding={"temperature": 0.0, "max_new_tokens": 256, "stop": ["</tool_call>"]})
    res["variant"] = {"episodes": a.episodes or "data/test.jsonl", "perturbed_bank": a.perturb}
    if a.perturb:
        res["overall"]["stale_value_rate"] = round(stale_rate(out_dir / "test.traces.jsonl"), 4)
    (out_dir / "test.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(a.tag, json.dumps({k: res["overall"].get(k) for k in ("task_success", "false_action", "stale_value_rate")}))


if __name__ == "__main__":
    main()
