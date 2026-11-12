"""[R] Experimental reward for the multi-turn language-following GRPO experiment.

The contract reward (`metrics.reward`, contracts/API.md) is unchanged. `reward_lang` adds a language term on top of
it, designed against the hacks found by scripts/eval_reward_audit.py:

  reward_lang = reward + 0.3 * task_success * lang_match - 0.3 * (1 - lang_match)

- The +0.3 bonus is gated on task success, so "always ask a polite question in the right language" earns no bonus
  on episodes that needed an action (that policy already scores 0 task success there).
- The -0.3 penalty applies whenever the final reply is not in the customer's latest language, so getting the action
  right but answering in the previous turn's language (the playground failure) is no longer a full-score episode.
- Echoing the customer's message is already caught by task success (no action / no answer) and by the data gates.
"""
from __future__ import annotations

import re

from .language import final_reply, lang_match
from .metrics import score_episode

LANG_BONUS, LANG_PENALTY = 0.3, 0.3
# Request words per language: most gold clarify replies are polite requests without a '?' (Hindi: 6 of 75 use one).
ASK_WORDS = {
    "en": r"\?|please|could you|can you|provide|share|tell me|let me know|need (the|your)|which|what",
    "hi": r"\?|कृपया|बताएं|बताएँ|बताइए|बताइये|दें|दीजिए|प्रदान|साझा|जानकारी",
    "bn": r"\?|দয়া করে|জানান|দিন|বলুন|প্রদান|শেয়ার|জানাতে",
    "hi-en": r"\?|please|kripya|bataiye|batao|bata do|bataye|dijiye|de do|dein|share|chahiye",
    "bn-en": r"\?|please|bolun|bolo|din\b|dite|deowa|dewa|janan|janao|dao|share|lagbe",
}


def asks(text: str, lang: str) -> bool:
    return bool(re.search(ASK_WORDS[lang], text or "", flags=re.I))


def _jaccard4(a: str, b: str) -> float:
    g = lambda t: {t[i:i + 4] for i in range(len(t) - 3)}  # noqa: E731
    x, y = g((a or "").lower()), g((b or "").lower())
    return len(x & y) / max(1, len(x | y))


def reward_lang(pred_messages: list[dict], gold: dict, lang: str) -> float:
    """pred_messages start at the customer turn being scored (turn 2 in multi-turn episodes)."""
    s = score_episode(pred_messages, gold)
    reply = final_reply(pred_messages)
    user = next((m.get("content") or "" for m in pred_messages if m["role"] == "user"), "")
    success = s["task_success"]
    if _jaccard4(reply, user) >= 0.6:  # echoing the customer is not an answer
        success = 0.0
    if gold.get("must_ask_clarification") and not asks(reply, lang):  # 'ask first' must actually ask
        success = 0.0
    lm = float(lang_match(reply, lang))
    return (1.0 * success + 0.3 * s["schema_valid"] - 0.5 * s["false_action"] - 0.1 * s["redundant_calls"]
            + LANG_BONUS * success * lm - LANG_PENALTY * (1.0 - lm))
