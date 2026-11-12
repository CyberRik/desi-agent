"""[R] Summarise the robustness evals into results/robustness.json (paired in-distribution vs OOD, perturbed bank).

Inputs: results/<tag>/test.traces.jsonl for the base/final in-distribution runs, the -ood and -perturbed runs from
scripts/eval_kaggle_robustness.py, and data/test_ood.valid.json from scripts/eval_ood_validate.py.

    python scripts/eval_ood_summary.py --out results/robustness.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE, FINAL = "tinyaya-global-base", "tinyaya-sft-a-final"


def traces(tag):
    p = ROOT / "results" / tag / "test.traces.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def agg(rows):
    n = len(rows)
    return {"n": n, "task_success": round(sum(r["scores"]["task_success"] for r in rows) / max(1, n), 4),
            "false_action": round(sum(r["scores"]["false_action"] for r in rows) / max(1, n), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid", default="data/test_ood.valid.json")
    p.add_argument("--out", default="results/robustness.json")
    a = p.parse_args()
    valid = set(json.loads((ROOT / a.valid).read_text(encoding="utf-8"))["valid_ids"])
    out = {"ood": {"rewriter": "CohereLabs/aya-expanse-8b", "validator": "Qwen/Qwen3-14B-AWQ (meaning checklist)",
                   "n_valid": len(valid)}, "perturbed_bank": {}}
    for tag in (BASE, FINAL):
        ind, ood = traces(tag), traces(f"{tag}-ood")
        ids = sorted(i for i in ood if i in valid)
        res = {"in_distribution_same_episodes": agg([ind[i[:-4]] for i in ids]), "ood": agg([ood[i] for i in ids])}
        for key in ("lang", "category"):
            g = collections.defaultdict(list)
            for i in ids:
                g[ood[i][key]].append(ood[i])
            res[f"ood_by_{key}"] = {k: agg(v) for k, v in sorted(g.items())}
        out["ood"][tag] = res
        pert = json.loads((ROOT / "results" / f"{tag}-perturbed" / "test.json").read_text(encoding="utf-8"))["overall"]
        out["perturbed_bank"][tag] = {k: pert.get(k) for k in ("task_success", "false_action", "stale_value_rate")}
    (ROOT / a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({t: out["ood"][t]["ood"] for t in (BASE, FINAL)} | {"perturbed": out["perturbed_bank"]}))


if __name__ == "__main__":
    main()
