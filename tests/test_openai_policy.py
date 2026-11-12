import types

import pytest

openai = pytest.importorskip("openai")

from desi_agent.eval.runner import openai_policy  # noqa: E402


def _fake(content, finish="stop"):
    seen = {}

    class C:
        def __init__(self, **kw):
            def create(**kw):
                seen.update(kw)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content), finish_reason=finish)])
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    return C, seen


def test_stop_tag_restored_and_sent(monkeypatch):
    C, seen = _fake('<tool_call>{"name": "x", "arguments": {}}')  # server stripped the stop string
    monkeypatch.setattr(openai, "OpenAI", C)
    out = openai_policy("http://x", "m", no_thinking=True)([{"role": "user", "content": "hi"}])
    assert out.endswith("</tool_call>")
    assert seen["stop"] == ["</tool_call>"] and seen["temperature"] == 0.0
    assert seen["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_final_answer_untouched(monkeypatch):
    C, _ = _fake("  Your balance is 100.  ")
    monkeypatch.setattr(openai, "OpenAI", C)
    assert openai_policy("http://x", "m")([{"role": "user", "content": "hi"}]) == "Your balance is 100."
