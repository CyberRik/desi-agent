"""[R] Round-2 sets (scripts/data_round2.py): contract-valid, gold gets full reward, never touches test."""
import importlib.util
import json
from pathlib import Path

import pytest

from desi_agent.data.splits import load_split
from desi_agent.eval.rewards import reward_lang

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "round2"


def _rows(name):
    p = D / name
    if not p.exists():
        pytest.skip(f"{p} not built")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _validator():
    spec = importlib.util.spec_from_file_location("validate_data", ROOT / "scripts" / "validate_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.check


@pytest.mark.parametrize("split", ["train", "val"])
def test_grounding_rows(split):
    check = _validator()
    rows = _rows(f"grounding_{split}.jsonl")
    assert rows and all(r["split"] == split for r in rows)
    for r in rows:
        assert check(r) == [], r["id"]
        assert abs(reward_lang(r["messages"], r["gold"], r["lang"]) - 1.6) < 1e-6, r["id"]
        if r["meta"]["variant"] == "missing":
            assert r["category"] == "clarify" and r["gold"]["tool_sequence"] == []
    variants = [r["meta"]["variant"] for r in rows]
    assert variants.count("control") > 0.8 * variants.count("missing")  # positives balance the "ask" rows


def test_langswitch_rows_are_train_only_and_scorable():
    test_ids = {e["id"] for e in load_split("test")}
    for split in ("train", "val"):
        for r in _rows(f"langswitch_{split}.jsonl"):
            pre, tgt = r["id"].split(":", 1)[1].split(">")
            assert pre not in test_ids and tgt not in test_ids and r["target"] == tgt
            assert r["messages"][: len(r["prefix"])] == r["prefix"] and r["messages"][len(r["prefix"])] == r["turn2"]
            tail = r["messages"][len(r["prefix"]):]
            assert abs(reward_lang(tail, r["gold"], r["lang"]) - 1.6) < 1e-6, r["id"]
            assert (r["lang1"] != r["lang"]) == (r["condition"] == "switch")


def test_no_customer_message_overlap_with_test():
    test_msgs = {m["content"].strip() for e in load_split("test") for m in e["messages"] if m["role"] == "user"}
    for name in ("grounding_train.jsonl", "grounding_val.jsonl", "langswitch_train.jsonl", "langswitch_val.jsonl"):
        for r in _rows(name):
            for m in r["messages"]:
                if m["role"] == "user":
                    assert m["content"].strip() not in test_msgs, r["id"]
