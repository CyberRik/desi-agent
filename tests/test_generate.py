import asyncio
import random
import re
import sys
from pathlib import Path

from desi_agent import tools
from desi_agent.data import generate
from desi_agent.data.checks import script_ok, to_ascii_digits, user_msg_problems
from desi_agent.data.scenarios import sample_all
from desi_agent.eval.metrics import score_episode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_data import check  # noqa: E402

FILLER = {"en": "please help", "hi": "कृपया मदद करें", "bn": "দয়া করে সাহায্য করুন", "hi-en": "please madad karo", "bn-en": "please help koro"}


class FakeTeacher:
    """Echoes back required values in the right script, like a well-behaved teacher would."""
    model = "fake"

    def __init__(self, lang):
        self.lang = lang

    async def __call__(self, system, prompt, temperature):
        m = re.search(r"exactly as written: (.*)\.$", prompt, flags=re.M) or re.search(r"values;.*?: (.*)\.$", prompt, flags=re.M)
        vals = m.group(1) if m else ""
        return f"{FILLER[self.lang]} {vals}".strip()


def test_all_scenarios_have_valid_gold_calls():
    scen = sample_all(counts={"single_tool": 3, "multi_tool": 3, "clarify": 1, "no_tool": 1})
    assert {s.category for s in scen} == {"single_tool", "multi_tool", "clarify", "no_tool"}
    for s in scen:
        for name, args in s.calls:
            ok, err = tools.validate_call(name, args)
            assert ok, (s.pattern, args, err)


def test_pipeline_with_fake_teacher_produces_valid_rows():
    scen = sample_all(counts={"single_tool": 2, "multi_tool": 2, "clarify": 1, "no_tool": 1})
    rows = []
    for i, s in enumerate(scen):
        if s.native_digits:
            s.native_digits = False  # the fake teacher only writes ASCII digits
        row, rej = asyncio.run(generate.make_one(FakeTeacher(s.lang), i, s, random.Random(i)))
        assert row, rej
        rows.append(row)
    for r in rows:
        assert check(r) == [], (r["id"], check(r))
        assert score_episode(r["messages"], r["gold"])["task_success"] == 1.0


def test_checks():
    assert to_ascii_digits("४८२१ ৪৮২১") == "4821 4821"
    assert script_ok("मेरा कार्ड ब्लॉक करो UPI", "hi") and not script_ok("mera card block karo", "hi")
    assert not script_ok("আমার কার্ড", "hi-en")
    assert user_msg_problems("mera card block karo", "hi-en", ["4821"], []) == ["missing:4821"]
    assert user_msg_problems("card 4821 block", "hi-en", [], [r"\d{4}"]) == [r"forbidden:\d{4}"]
