"""[R] Measure the grounding guard (eval/guard.py) offline on existing traces: frozen test + BANKING77.

Test: how many CORRECT state-changing calls the guard would block (should be 0) and how many wrong ones it stops.
BANKING77 (real customers, no gold): share of messages ending in an EXECUTED state-changing action, before/after.

    python scripts/eval_guard.py --out results/guard.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from desi_agent.eval.guard import STATE_CHANGING, audit_trace  # noqa: E402
from desi_agent.eval.metrics import call_eq  # noqa: E402

TAGS = ("tinyaya-global-base", "tinyaya-sft-a-final")


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/guard.json")
    a = p.parse_args()
    res = {"test": {}, "banking77": {}}
    for tag in TAGS:
        good = good_blocked = bad = bad_blocked = 0
        for t in load(ROOT / f"results/{tag}/test.traces.jsonl"):
            for c in audit_trace(t["messages"]):
                ok = c["args"] is not None and any(call_eq(c["name"], c["args"], g["name"], g["arguments"])
                                                   for g in t["gold"]["tool_sequence"])
                good, good_blocked = good + ok, good_blocked + (ok and not c["grounded"])
                bad, bad_blocked = bad + (not ok), bad_blocked + ((not ok) and not c["grounded"])
        res["test"][tag] = {"correct_calls": good, "correct_blocked": good_blocked, "wrong_calls": bad,
                            "wrong_blocked": bad_blocked}
        from huggingface_hub import hf_hub_download
        T = load(hf_hub_download("ToolW/desi-agent-eval-traces", f"banking77/{tag}.traces.jsonl", repo_type="dataset"))
        ex = ex_after = 0
        for t in T:
            msgs = t["messages"]
            out = {m.get("tool_call_id"): m.get("content", "") for m in msgs if m["role"] == "tool"}
            executed = any(c["function"]["name"] in STATE_CHANGING and '"error"' not in out.get(c["id"], '"error"')
                           for m in msgs for c in (m.get("tool_calls") or []))
            ex += executed
            ex_after += executed and any(x["grounded"] for x in audit_trace(msgs))
        res["banking77"][tag] = {"n": len(T), "executed_state_change": round(ex / len(T), 4),
                                 "executed_with_guard": round(ex_after / len(T), 4)}
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
