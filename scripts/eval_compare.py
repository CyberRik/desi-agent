"""[R] One table across model tags for the Stage-2 deck (reads results/, no GPU).

    python scripts/eval_compare.py tinyaya-global-base tinyaya-sft-a-final tinyaya-rft-a s2-sft-a-check [--md]

Columns (contracts/METRICS.md; arrows say which way is better):
  test_ts ↑      task success on the frozen test (results/<tag>/test.json)
  wrong_act ↓    false_action on the frozen test (a state-changing call that is not in gold)
  ood_ts ↑       task success on the 380 judge-validated OOD rewrites (as in results/robustness.json). From
                 results/<tag>-ood/validated.json, else robustness.json, else computed from the -ood traces
                 (local, or downloaded from ToolW/desi-agent-eval-traces results/<tag>-ood/)
  pert_ts ↑      task success with the perturbed mock bank (results/<tag>-perturbed/test.json)
  mt_switch ↑    multi-turn language switch: right action AND right language (results/<tag>-multiturn, switch.both)
  b77_exec ↓     BANKING77: share of real customer messages ending in an EXECUTED block/dispute/transfer
                 (unsafe_executed; results/<tag>-banking77/manifest.json, else results/banking77.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATASET = "ToolW/desi-agent-eval-traces"
COLS = ("test_ts", "wrong_act", "ood_ts", "pert_ts", "mt_switch", "b77_exec")


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ood_ts(tag: str, download: bool) -> float | None:
    v = _j(RES / f"{tag}-ood" / "validated.json")
    if v:
        return v["ood"]["task_success"]
    rob = _j(RES / "robustness.json") or {}
    if tag in rob.get("ood", {}):
        return rob["ood"][tag]["ood"]["task_success"]
    tr = RES / f"{tag}-ood" / "test.traces.jsonl"
    if not tr.exists() and download:
        try:
            from huggingface_hub import hf_hub_download
            tr = Path(hf_hub_download(DATASET, f"results/{tag}-ood/test.traces.jsonl", repo_type="dataset"))
        except Exception:  # noqa: BLE001 - no such run
            return None
    if not tr.exists():
        return None
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_hf_job import ood_validated
    return ood_validated(tr)["ood"]["task_success"]


def row(tag: str, download: bool = True) -> dict:
    r: dict = {"tag": tag}
    t = _j(RES / tag / "test.json")
    if t:
        r["test_ts"], r["wrong_act"], r["n"] = t["overall"]["task_success"], t["overall"]["false_action"], t["n"]
    r["ood_ts"] = ood_ts(tag, download)
    p = _j(RES / f"{tag}-perturbed" / "test.json")
    r["pert_ts"] = p["overall"]["task_success"] if p else None
    m = _j(RES / f"{tag}-multiturn" / "multiturn.json")
    r["mt_switch"] = m["switch"]["overall"]["both"] if m else None
    b = _j(RES / f"{tag}-banking77" / "manifest.json") or _j(RES / "banking77.json") or {}
    r["b77_exec"] = b[tag]["unsafe_executed"] if tag in b else None
    return r


def fmt(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--md", action="store_true", help="markdown table")
    p.add_argument("--no-download", action="store_true", help="never fetch legacy OOD traces from the Hub")
    a = p.parse_args()
    rows = [row(t, not a.no_download) for t in a.tags]
    heads = ("tag",) + COLS
    if a.md:
        print("| " + " | ".join(heads) + " |\n|" + "---|" * len(heads))
        for r in rows:
            print("| " + " | ".join([r["tag"]] + [fmt(r.get(c)) for c in COLS]) + " |")
    else:
        w = max(len(h) for h in [*a.tags, "tag"]) + 2
        print(f"{'tag':<{w}}" + "".join(f"{c:>11}" for c in COLS))
        for r in rows:
            print(f"{r['tag']:<{w}}" + "".join(f"{fmt(r.get(c)):>11}" for c in COLS))
    print("(%; test_ts/ood_ts/pert_ts/mt_switch higher is better, wrong_act/b77_exec lower is better)")


if __name__ == "__main__":
    main()
