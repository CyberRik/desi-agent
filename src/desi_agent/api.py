"""Cross-lane API (contracts/API.md). SHARED + FROZEN.

Always import the other lane's code through here:
    from desi_agent.api import load_split, evaluate, score_episode, reward   # R's side
    from desi_agent.api import load_policy, train_sft, train_grpo            # F's side
"""
from __future__ import annotations

import importlib
import json
from typing import Any, Callable

from .fmt import Policy, parse_assistant, render_tool_call, run_episode, to_chat  # noqa: F401

Episode = dict
Scores = dict


def _lane(module: str, fn: str, owner: str, task: str) -> Callable:
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            impl = getattr(importlib.import_module(f"desi_agent.{module}"), fn)
        except (ImportError, AttributeError) as e:
            raise NotImplementedError(f"[{owner}] {task} not merged yet: desi_agent.{module}.{fn}") from e
        return impl(*args, **kwargs)
    call.__name__ = fn
    call.__doc__ = f"Provided by lane {owner} ({task}). See contracts/API.md."
    return call


# ---- R provides ---------------------------------------------------------------
load_split: Callable[..., list[Episode]] = _lane("data.splits", "load_split", "R", "R1")
score_episode: Callable[[list[dict], dict], Scores] = _lane("eval.metrics", "score_episode", "R", "R5")
reward: Callable[[list[dict], dict], float] = _lane("eval.metrics", "reward", "R", "R5")
evaluate: Callable[..., dict] = _lane("eval.runner", "evaluate", "R", "R5")

# ---- F provides ---------------------------------------------------------------
load_policy: Callable[..., Policy] = _lane("train.policy", "load_policy", "F", "F1")
train_sft: Callable[[dict], dict] = _lane("train.sft", "train_sft", "F", "F3")
train_grpo: Callable[[dict], dict] = _lane("train.grpo", "train_grpo", "F", "F5")
episodes_to_eventlogs: Callable[[list[Episode]], list] = _lane("train.convert", "episodes_to_eventlogs", "F", "F2")


class DummyPolicy:
    """Test double. With `script`, replays assistant turns from a gold episode
    (perfect policy); otherwise always answers with fixed text (never calls tools)."""

    def __init__(self, episode: Episode | None = None, fixed: str = "Sorry, I can't help with that."):
        self.fixed = fixed
        self.script = []
        if episode:
            for m in episode["messages"]:
                if m["role"] != "assistant":
                    continue
                if m.get("tool_calls"):
                    c = m["tool_calls"][0]["function"]
                    self.script.append(render_tool_call(c["name"], json.loads(c["arguments"])))
                else:
                    self.script.append(m["content"])
        self._i = 0

    def __call__(self, chat: list[dict]) -> str:
        if self._i < len(self.script):
            self._i += 1
            return self.script[self._i - 1]
        return self.fixed
