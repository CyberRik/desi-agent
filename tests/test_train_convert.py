"""[F] episodes -> AgentTune EventLogs -> SFT rows -> fmt.to_chat must round-trip exactly."""
import pytest

pytest.importorskip("agenttune")

from agenttune.agentic import EventKind, Project  # noqa: E402

from desi_agent import fmt  # noqa: E402
from desi_agent.api import episodes_to_eventlogs, load_split  # noqa: E402
from desi_agent.train.convert import row_to_chat  # noqa: E402


def test_eventlogs_are_full_tier_with_ids():
    eps = load_split("seed")
    logs = episodes_to_eventlogs(eps)
    assert [l.id for l in logs] == [e["id"] for e in eps]
    assert all(l.tier == "full" for l in logs)
    kinds = [e.kind for e in logs[0]]
    assert kinds == [EventKind.OBSERVATION, EventKind.TOOL_CALL, EventKind.TOOL_RESULT, EventKind.TEXT]


def test_sft_rows_roundtrip_to_fmt_chat():
    eps = load_split("seed")
    proj = Project()
    for log in episodes_to_eventlogs(eps):
        proj.add_trajectory(log)
    rows = proj.sft_dataset()
    assert len(rows) == len(eps)
    for ep, row in zip(eps, rows):
        assert row_to_chat(row) == fmt.to_chat(ep["messages"]), ep["id"]
