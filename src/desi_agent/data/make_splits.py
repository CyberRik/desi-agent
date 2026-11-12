"""[R] R4: freeze splits from curated episodes.

test: balanced, `--per-cell` episodes for every (lang, category) cell (default 30 -> 5 x 4 x 30 = 600).
val:  `--val-frac` of the rest, stratified the same way. train: everything else.
Ids listed in `--exclude` (one per line, e.g. rejected during the human spot check) are dropped first.

    python -m desi_agent.data.make_splits --inp data/curated/curated.jsonl --out data/ --exclude data/spotcheck_rejected.txt

Refuses to overwrite an existing data/test.jsonl (frozen) unless --force.
Also writes data/spotcheck.md: a few test episodes per cell for a human to read before freezing.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
from pathlib import Path

SEED = 1234


def make_splits(rows: list[dict], per_cell: int = 30, val_frac: float = 0.05, seed: int = SEED):
    rng = random.Random(seed)
    cells: dict[tuple, list] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        cells[(r["lang"], r["category"])].append(r)
    test, val, train, short = [], [], [], {}
    for key in sorted(cells):
        grp = cells[key][:]
        rng.shuffle(grp)
        n_test = min(per_cell, len(grp) // 3)  # never let test eat a whole small cell
        if n_test < per_cell:
            short[f"{key[0]}/{key[1]}"] = n_test
        rest = grp[n_test:]
        n_val = max(1, round(len(rest) * val_frac)) if len(rest) > 10 else 0
        test += grp[:n_test]
        val += rest[:n_val]
        train += rest[n_val:]
    for split, rs in (("test", test), ("val", val), ("train", train)):
        for r in rs:
            r["split"] = split
    return train, val, test, short


def _render(r: dict, header: str = "") -> list[str]:
    out = [f"\n**{r['id']}** {header}\n"]
    for m in r["messages"]:
        if m["role"] == "user":
            out.append(f"- USER: {m['content']}")
        elif m.get("tool_calls"):
            f = m["tool_calls"][0]["function"]
            out.append(f"- CALL: `{f['name']}({f['arguments']})`")
        elif m["role"] == "tool":
            out.append(f"- RESULT: `{m['content'][:200]}`")
        else:
            out.append(f"- BOT: {m['content']}")
    return out


def _spotcheck_md(test: list[dict], k: int, seed: int, flagged: dict | None = None) -> str:
    flagged = flagged or {}
    rng = random.Random(seed)
    out = ["# Test-set spot check (R4)", "",
           "Scores only use the tool calls, so judge the CUSTOMER message and the CALL lines: does the message make "
           "sense, and are the calls what a good bank agent should do? Bad BOT wording matters less (it is not scored).",
           "To reject an episode, add its id to `data/spotcheck_rejected.txt` (one per line). Rejected ids are removed "
           "from the test set without changing the others.", ""]
    flag_rows = [r for r in test if flagged.get(r["id"])]
    if flag_rows:
        out.append(f"## 1. Flagged by the LLM judge ({len(flag_rows)}): please decide each one")
        for r in flag_rows:
            out += _render(r, f"`{', '.join(flagged[r['id']]['flags'])}`: {flagged[r['id']]['note']}")
        out.append("")
    out.append(f"## 2. Random sample, {k} per (language, category), not flagged")
    cells: dict[tuple, list] = collections.defaultdict(list)
    for r in test:
        if not flagged.get(r["id"]):
            cells[(r["lang"], r["category"])].append(r)
    for key in sorted(cells):
        out.append(f"\n### {key[0]} / {key[1]}")
        for r in rng.sample(cells[key], min(k, len(cells[key]))):
            out += _render(r)
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/curated/curated.jsonl")
    p.add_argument("--out", default="data")
    p.add_argument("--exclude")
    p.add_argument("--per-cell", type=int, default=30)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--spotcheck-per-cell", type=int, default=3)
    p.add_argument("--force", action="store_true", help="overwrite a frozen test.jsonl")
    p.add_argument("--judge-flags", help="judge_flags.jsonl: flagged episodes leave train/val, and are listed first in the spot check")
    a = p.parse_args()
    out = Path(a.out)
    if (out / "test.jsonl").exists() and not a.force:
        raise SystemExit(f"{out / 'test.jsonl'} exists and is frozen; pass --force only before R4 is merged")
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    excl = set()
    if a.exclude and Path(a.exclude).exists():
        excl = {l.strip() for l in Path(a.exclude).read_text(encoding="utf-8").splitlines() if l.strip()}
    train, val, test, short = make_splits(rows, a.per_cell, a.val_frac)
    # exclusions after the split, so rejecting one test episode never reshuffles the others
    train, val, test = ([r for r in rs if r["id"] not in excl] for rs in (train, val, test))
    flagged: dict = {}
    if a.judge_flags:
        for l in Path(a.judge_flags).read_text(encoding="utf-8").splitlines():
            if l.strip():
                x = json.loads(l)
                if x.get("flags"):
                    flagged[x["id"]] = x
    n_before = len(train) + len(val)
    train = [r for r in train if r["id"] not in flagged]
    val = [r for r in val if r["id"] not in flagged]
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in (("train", train), ("val", val), ("test", test)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    (out / "spotcheck.md").write_text(_spotcheck_md(test, a.spotcheck_per_cell, SEED, flagged), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def table(rs):
        return dict(sorted(collections.Counter(f"{r['lang']}/{r['category']}" for r in rs).items()))
    manifest = {"inp": a.inp, "git_sha": sha, "seed": SEED, "excluded": len(excl), "judge_dropped_train_val": n_before - len(train) - len(val),
                "judge_flagged_test": sum(r["id"] in flagged for r in test), "per_cell": a.per_cell,
                "val_frac": a.val_frac, "counts": {"train": len(train), "val": len(val), "test": len(test)},
                "short_test_cells": short, "test_by_cell": table(test), "train_by_cell": table(train)}
    (out / "splits.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "short_test_cells")}, indent=2))


if __name__ == "__main__":
    main()
