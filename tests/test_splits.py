from desi_agent.data.make_splits import make_splits


def _rows(n_per_cell):
    return [{"id": f"{l}-{c}-{i:04d}", "lang": l, "category": c, "split": "train"}
            for l in ("en", "hi", "bn", "hi-en", "bn-en") for c in ("single_tool", "multi_tool", "clarify", "no_tool")
            for i in range(n_per_cell)]


def test_balanced_disjoint_deterministic():
    train, val, test, short = make_splits(_rows(200), per_cell=30)
    assert len(test) == 600 and not short
    ids = [r["id"] for r in train + val + test]
    assert len(ids) == len(set(ids)) == 4000
    assert all(r["split"] == "test" for r in test) and all(r["split"] == "val" for r in val)
    assert [r["id"] for r in make_splits(_rows(200), per_cell=30)[2]] == [r["id"] for r in test]


def test_small_cell_not_emptied():
    train, val, test, short = make_splits(_rows(12), per_cell=30)
    assert short and all(v == 4 for v in short.values())
    assert len(train) > len(test)
