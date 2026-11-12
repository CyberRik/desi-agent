import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("curatorkit")
pytest.importorskip("detect_secrets")

from desi_agent.data.curate import curate  # noqa: E402

SEED = Path(__file__).resolve().parents[1] / "data/seed/seed.jsonl"


def test_curate_dedups_and_keeps_clean_rows():
    rows = [json.loads(l) for l in SEED.read_text(encoding="utf-8").splitlines() if l.strip()]
    dup = copy.deepcopy(rows[0])
    dup["id"] += "-dup"
    # same wording, different digits -> must be caught as a duplicate (digits are masked for dedup)
    dup["messages"][0]["content"] = "".join("7" if c.isdigit() else c for c in dup["messages"][0]["content"])
    kept, rejected, report = curate(rows + [dup])
    assert report["stages"]["input"] == len(rows) + 1
    assert any(r["id"] == dup["id"] and r["stage"] == "curatorkit" for r in rejected) or \
        any(r["id"] == rows[0]["id"] for r in rejected)
    assert len(kept) + len(rejected) == len(rows) + 1
    assert len({r["id"] for r in kept}) == len(kept)
