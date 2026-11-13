"""[R] Reply-language detection for the 5 supported languages (used by the multi-turn eval and reward_lang).

Calibrated on the frozen test split: the gold final replies match their own language 97-100% of the time
(scripts/eval_multiturn.py). Romanized Hindi vs romanized Bengali is decided by which language's characteristic
words dominate, because the shared marker lists in data/checks.py overlap (ki, na, ta, ...).
"""
from __future__ import annotations

from ..data.checks import lang_hygiene, romanized_ok, script_ok

HI_ONLY = {"hai", "hain", "kar", "karo", "karna", "karke", "kardo", "kijiye", "dijiye", "bataiye", "batao", "chahiye",
           "mein", "ka", "ke", "aap", "aapka", "aapke", "aapki", "aapko", "kya", "nahi", "nahin", "bhejne", "bhej", "hoga",
           "gaya", "gayi", "raha", "rahi", "mujhe", "mera", "meri", "unka", "unki", "liye", "aur", "lekin", "toh", "wala"}
BN_ONLY = {"apni", "apnar", "apnake", "ami", "amar", "amake", "kothai", "kothay", "deowa", "dewa", "jete", "pare",
           "parben", "korbo", "korun", "korte", "kore", "koro", "hobe", "ache", "achhe", "theke", "diye", "dite", "bolun",
           "bolo", "jodi", "ekhon", "kichu", "kono", "taka", "dao", "chai", "hoyeche", "geche", "ar", "kintu", "keno"}


def _hits(text: str, words: set[str]) -> int:
    return sum(t.strip(".,!?:;()'\"").lower() in words for t in text.split())


def lang_match(text: str, lang: str) -> bool:
    """True if `text` is written in `lang` (en | hi | bn | hi-en | bn-en)."""
    if not text or not text.strip():
        return False
    ok = script_ok(text, lang) and romanized_ok(text, lang) and not [p for p in lang_hygiene(text, lang) if p != "bad_term"]
    hi, bn = _hits(text, HI_ONLY), _hits(text, BN_ONLY)
    if lang == "hi-en":
        ok = ok and hi > bn
    elif lang == "bn-en":
        ok = ok and bn > hi
    elif lang == "en":  # English must not be romanized Hindi/Bengali
        ok = ok and hi + bn <= 1
    return ok


def final_reply(messages: list[dict]) -> str:
    last = messages[-1] if messages else {}
    return (last.get("content") or "") if last.get("role") == "assistant" and not last.get("tool_calls") else ""
