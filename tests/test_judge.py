import json
from pathlib import Path

import pytest

pytest.importorskip("curatorkit")

from curatorkit.llm.base import BaseLLM, LLMResponse  # noqa: E402

from desi_agent.data.judge import judge  # noqa: E402

SEED = Path(__file__).resolve().parents[1] / "data/seed/seed.jsonl"


class FakeJudge(BaseLLM):
    """Scores 0.2 when the reply is a 'Done. Result:' placeholder, else 0.9."""
    def __init__(self):
        super().__init__(model="fake")

    def _call(self, messages, **kw):
        bad = "BOT FINAL REPLY: Done." in messages[-1]["content"]
        s = 0.2 if bad else 0.9
        return LLMResponse(text=json.dumps({"overall_score": s, "dimension_scores": {
            "coherence": s, "truthfulness": s, "instruction_following": s}, "weaknesses": ""}))


def test_judge_scores_every_row():
    rows = [json.loads(l) for l in SEED.read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = judge(rows, llm=FakeJudge(), concurrency=4)
    assert set(scores) == {r["id"] for r in rows}
    for r in rows:
        want = 0.2 if r["messages"][-1]["content"].startswith("Done.") else 0.9
        assert scores[r["id"]] == pytest.approx(want), r["id"]


def test_checklist_flags(monkeypatch):
    import types
    import openai
    from desi_agent.data import judge as J

    class FakeClient:
        def __init__(self, **kw):
            async def create(**kw):
                bad = "BOT FINAL REPLY: Done." in kw["messages"][0]["content"]
                d = {f: False for f in J.FLAGS} | {"garbled": bad, "note": "x"}
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(d)))])
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    rows = [json.loads(l) for l in SEED.read_text(encoding="utf-8").splitlines() if l.strip()]
    res = {x["id"]: x["flags"] for x in J.checklist(rows, "http://x", "m", 4)}
    for r in rows:
        assert res[r["id"]] == (["garbled"] if r["messages"][-1]["content"].startswith("Done.") else [])
