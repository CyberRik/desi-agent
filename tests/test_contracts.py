"""Guards for the shared contracts. Both lanes run these before every PR."""
import json

from desi_agent import fmt, tools
from desi_agent.api import DummyPolicy
from desi_agent.data.splits import load_split


def test_tools_deterministic_and_valid():
    assert len(tools.tool_names()) == 8
    a = tools.execute("get_recent_transactions", {"count": 3})
    assert a == tools.execute("get_recent_transactions", {"count": 3})
    assert len(json.loads(a)["transactions"]) == 3


def test_invalid_call_returns_error_payload():
    assert "error" in json.loads(tools.execute("block_card", {"card_last4": "12", "reason": "lost"}))
    ok, _ = tools.validate_call("nope", {})
    assert not ok


def test_parse_roundtrip():
    s = fmt.render_tool_call("block_card", {"card_last4": "4821", "reason": "lost"})
    t = fmt.parse_assistant(s)
    assert (t.kind, t.name, t.arguments) == ("tool_call", "block_card", {"card_last4": "4821", "reason": "lost"})
    assert fmt.parse_assistant("<tool_call>{bad json</tool_call>").valid_json is False
    assert fmt.parse_assistant("आपका कार्ड ब्लॉक हो गया").kind == "final"


def test_render_has_only_three_roles():
    for ep in load_split("seed"):
        chat = fmt.to_chat(ep["messages"])
        assert {m["role"] for m in chat} <= {"system", "user", "assistant"}
        assert "{TOOLS}" not in chat[0]["content"]


def test_gold_replay_reproduces_seed_episodes():
    """A perfect policy run through the shared loop must reproduce each seed episode exactly."""
    for ep in load_split("seed"):
        run = fmt.run_episode(DummyPolicy(ep), [ep["messages"][0]])
        assert not run.truncated
        assert run.messages == ep["messages"], ep["id"]
