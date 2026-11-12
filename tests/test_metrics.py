from desi_agent.api import DummyPolicy, evaluate, reward, score_episode
from desi_agent.data.splits import load_split
from desi_agent.eval.metrics import call_eq


def test_gold_scores_perfect():
    for ep in load_split("seed"):
        s = score_episode(ep["messages"], ep["gold"])
        assert s["task_success"] == 1.0 and s["false_action"] == 0.0, (ep["id"], s)


def test_defaults_and_tolerance():
    assert call_eq("get_recent_transactions", {"count": 3}, "get_recent_transactions", {"count": 3, "txn_type": "all"})
    assert call_eq("calculate_loan_emi", {"principal": 500000, "annual_rate_pct": 9.5, "tenure_months": 36},
                   "calculate_loan_emi", {"principal": 500000.0, "annual_rate_pct": 9.5, "tenure_months": 36})
    assert not call_eq("block_card", {"card_last4": "4821", "reason": "lost"}, "block_card", {"card_last4": "4812", "reason": "lost"})


def test_false_action_and_clarify():
    ep = next(e for e in load_split("seed") if e["id"] == "hien-clar-0001")
    bad = [ep["messages"][0],
           {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {
               "name": "block_card", "arguments": '{"card_last4": "0000", "reason": "lost"}'}}]},
           {"role": "tool", "tool_call_id": "c", "name": "block_card", "content": "{}"},
           {"role": "assistant", "content": "done"}]
    s = score_episode(bad, ep["gold"])
    assert s["false_action"] == 1.0 and s["task_success"] == 0.0
    assert reward(bad, ep["gold"]) < reward(ep["messages"], ep["gold"])


def test_evaluate_writes_results(tmp_path):
    eps = load_split("seed")
    never_calls = DummyPolicy()  # always answers with text, never calls tools
    res = evaluate(never_calls, split="seed", model_tag="t", episodes=eps, out_dir=tmp_path, auditkit=False)
    assert res["n"] == len(eps)
    # only clarify/no_tool episodes can succeed without tools
    expected = sum(e["category"] in ("clarify", "no_tool") for e in eps) / len(eps)
    assert abs(res["overall"]["task_success"] - expected) < 1e-3
    assert (tmp_path / "seed.json").exists() and (tmp_path / "seed.traces.jsonl").exists()
