"""[F] F5: GRPO on top of SFT with `api.reward` (contracts/API.md).

Reward = first-turn reward through the eval loop. A sampled completion is the model's first assistant turn.
We run it through `fmt.run_episode` with a policy that returns the completion first and a fixed non-empty
final reply afterwards, then score the resulting messages with R's `reward(messages, gold)`. Tool execution,
schema checks, false_action and arg matching are therefore exactly as in eval. The only assumption is that
the model writes *some* final reply after a successful call; the metric only requires it to be non-empty,
and the SFT model never truncates (0/486 on test).

Valid only for one-step episodes (gold has <= 1 tool call: clarify, no_tool, single_tool), which is
where the SFT errors are (R7: invented card numbers instead of asking; wrong block_card reason).
"""
from __future__ import annotations

from .. import fmt

FINAL_STUB = "Done."


def one_step(episode: dict) -> bool:
    """True if a first-turn reward is a faithful stand-in for the episode reward."""
    users = [m for m in episode["messages"] if m["role"] == "user"]
    return len(users) == 1 and len(episode["gold"].get("tool_sequence", [])) <= 1


def first_turn_messages(completion: str, episode: dict) -> list[dict]:
    """Episode messages produced when the model's first turn is `completion` (then a stub final reply)."""
    first = [m for m in episode["messages"] if m["role"] == "user"][:1]
    outs = iter([completion])
    run = fmt.run_episode(lambda chat: next(outs, FINAL_STUB), first, max_steps=3)
    return run.messages


def first_turn_reward(completion: str, episode: dict) -> float:
    from ..api import reward
    return reward(first_turn_messages(completion, episode), episode["gold"])
