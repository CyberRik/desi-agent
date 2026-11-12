"""[R] R3: curate raw teacher episodes.

Stages (each drop is logged with a reason):
  1. contract   - scripts/validate_data.check (schema, tool args valid, results match the mock tools)
  2. quality    - our language-ID / script / romanization / voice / repetition gates (data/checks.py),
                  re-run on the user message and the final reply
  3. curatorkit - CuratorKIT TextCleaner (unicode + whitespace), SecretsGate, ExactDeduplicator and
                  MinHashDeduplicator, run per language on (user message, final reply). Digits are masked
                  for dedup so "block card 1234" and "block card 5678" with the same wording count as dups.

    python -m desi_agent.data.curate --inp data/raw/teacher.jsonl --out data/curated/curated.jsonl

Writes <out>, <out>.rejected.jsonl and <out>.report.json (kept/dropped per stage, language, category, reason).
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

from .checks import final_problems, required_refs, user_msg_problems

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_data import check as contract_check  # noqa: E402

DIGITS = [r"[0-9०-९০-৯]+"]


def _latin_allowed(row: dict) -> list[str]:
    out: list[str] = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif v is not None:
            out.append(str(v))
    for m in row["messages"]:
        if m["role"] == "tool":
            try:
                walk(json.loads(m["content"]))
            except (json.JSONDecodeError, TypeError):
                pass
    for g in row["gold"]["tool_sequence"]:
        walk(g["arguments"])
    return out


def quality_problems(row: dict) -> list[str]:
    user, final = row["messages"][0]["content"], row["messages"][-1]["content"]
    allowed = _latin_allowed(row)
    p = [f"user:{x}" for x in user_msg_problems(user, row["lang"], [], []) if x != "markup" or "<tool_call>" in user]
    refs = required_refs([m["content"] for m in row["messages"] if m["role"] == "tool"])
    p += [f"final:{x}" for x in final_problems(final, row["lang"], allowed, user, refs)]
    return p


def _curatorkit(rows: list[dict], minhash_threshold: float) -> tuple[list[dict], list[tuple[dict, str]]]:
    from curatorkit.hygiene.secrets import SecretsGate
    from curatorkit.normalizers.clean import TextCleaner
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.schema import DataSample

    by_id = {r["id"]: r for r in rows}
    samples = [DataSample(id=r["id"], source_uri="desi-bank-toolcalls/raw", task_type="instruction_following",
                          instruction=r["messages"][0]["content"], output=r["messages"][-1]["content"],
                          metadata={"lang": r["lang"], "category": r["category"]}) for r in rows]
    samples = TextCleaner(transforms={"strip_html": False}, fields=["instruction", "output"]).run(samples)
    for s in samples:  # write cleaned text back into the episode
        by_id[s.id]["messages"][0]["content"] = s.instruction
        by_id[s.id]["messages"][-1]["content"] = s.output

    dropped: list[tuple[dict, str]] = []
    passed, rejected = SecretsGate(fields=["instruction", "output"]).run(samples)
    dropped += [(by_id[r.id], "curatorkit:secrets") for r in rejected]

    kept: list = []
    groups: dict[tuple, list] = collections.defaultdict(list)
    for s in passed:  # dedup inside (lang, category): same wording in another language/category is not a dup
        groups[(s.metadata["lang"], s.metadata["category"])].append(s)
    for _, grp in sorted(groups.items()):
        after_exact = ExactDeduplicator(ignore_for_dedup=DIGITS).run(grp)
        after_mh = MinHashDeduplicator(threshold=minhash_threshold, ignore_for_dedup=DIGITS).run(after_exact)
        ids_exact, ids_mh = {s.id for s in after_exact}, {s.id for s in after_mh}
        dropped += [(by_id[s.id], "curatorkit:exact_dup") for s in grp if s.id not in ids_exact]
        dropped += [(by_id[s.id], "curatorkit:near_dup") for s in after_exact if s.id not in ids_mh]
        kept += after_mh
    keep_ids = {s.id for s in kept}
    return [r for r in rows if r["id"] in keep_ids], dropped


def curate(rows: list[dict], minhash_threshold: float = 0.85) -> tuple[list[dict], list[dict], dict]:
    rejected: list[dict] = []
    stage_counts: dict[str, int] = {"input": len(rows)}

    ok = []
    for r in rows:
        errs = contract_check(r)
        if errs:
            rejected.append({"id": r.get("id"), "stage": "contract", "reasons": errs, "row": r})
        else:
            ok.append(r)
    stage_counts["after_contract"] = len(ok)

    ok2 = []
    for r in ok:
        p = quality_problems(r)
        if p:
            rejected.append({"id": r["id"], "stage": "quality", "reasons": p, "row": r})
        else:
            ok2.append(r)
    stage_counts["after_quality"] = len(ok2)

    kept, dropped = _curatorkit(ok2, minhash_threshold)
    rejected += [{"id": r["id"], "stage": "curatorkit", "reasons": [why], "row": r} for r, why in dropped]
    stage_counts["after_curatorkit"] = len(kept)
    kept.sort(key=lambda r: r["id"])

    reasons = collections.Counter("contract" if rej["stage"] == "contract" else ":".join(x.split(":")[:2])
                                  for rej in rejected for x in rej["reasons"])

    def table(rs):
        t: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        for r in rs:
            t[r["lang"]][r["category"]] += 1
        return {k: dict(v) for k, v in sorted(t.items())}
    report = {"stages": stage_counts, "kept": len(kept), "dropped": len(rejected),
              "drop_reasons": dict(reasons.most_common()), "kept_by_lang_category": table(kept),
              "dropped_by_lang_category": table([x["row"] for x in rejected])}
    return kept, rejected, report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/raw/teacher.jsonl")
    p.add_argument("--out", default="data/curated/curated.jsonl")
    p.add_argument("--minhash-threshold", type=float, default=0.85)
    a = p.parse_args()
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, rejected, report = curate(rows, a.minhash_threshold)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    Path(f"{out}.rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    import curatorkit
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        sha = ""
    report["manifest"] = {"inp": a.inp, "out": a.out, "git_sha": sha, "seed": 1234,
                          "curatorkit": curatorkit.__version__, "minhash_threshold": a.minhash_threshold}
    Path(f"{out}.report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stages", "drop_reasons")}, indent=2))


if __name__ == "__main__":
    main()
