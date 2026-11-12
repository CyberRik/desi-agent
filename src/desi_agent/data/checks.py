"""[R] Cheap, deterministic quality gates for teacher text (used in R2, reused in R3)."""
from __future__ import annotations

import re
import unicodedata

_NATIVE_DIGITS = str.maketrans("०१२३४५६७८९০১২৩৪৫৬৭৮৯", "01234567890123456789")


def to_ascii_digits(s: str) -> str:
    return s.translate(_NATIVE_DIGITS)


def script_counts(text: str) -> dict[str, int]:
    c = {"devanagari": 0, "bengali": 0, "latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            c["devanagari"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            c["bengali"] += 1
        elif "LATIN" in unicodedata.name(ch, ""):
            c["latin"] += 1
    return c


LATIN_OK = {"upi", "atm", "emi", "pin", "otp", "kyc", "id", "ok", "rs", "inr", "sms", "app", "neft", "imps", "cvv", "txn"}


def _strip_allowed(text: str, allowed: list[str]) -> str:
    for a in sorted(allowed, key=len, reverse=True):
        if a:
            text = re.sub(re.escape(a), " ", text, flags=re.I)
    # tokens with digits or @ (ids, amounts, references) and common banking acronyms are script-neutral
    return " ".join(t for t in text.split() if not re.search(r"[\d@]", t) and t.strip(".,!?:;()").lower() not in LATIN_OK)


def script_ok(text: str, lang: str, allowed_latin: list[str] | None = None) -> bool:
    if lang in ("hi", "bn"):
        text = _strip_allowed(text, allowed_latin or [])
    c = script_counts(text)
    total = sum(c.values()) or 1
    if lang == "hi":
        return c["devanagari"] / total >= 0.7 and c["bengali"] == 0
    if lang == "bn":
        return c["bengali"] / total >= 0.7 and c["devanagari"] == 0
    return c["devanagari"] == 0 and c["bengali"] == 0 and c["latin"] > 0  # en / hi-en / bn-en


def clean_teacher_text(text: str) -> str:
    """Strip reasoning blocks, wrapping quotes and labels the teacher sometimes adds."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^(customer|user|message|reply|assistant)\s*:\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text.strip("“”").strip()


def user_msg_problems(text: str, lang: str, must_include: list[str], forbid: list[str]) -> list[str]:
    p = []
    if not text or len(text) < 4:
        p.append("empty")
    if len(text) > 400:
        p.append("too_long")
    if "<tool_call>" in text or "{" in text:
        p.append("markup")
    if not script_ok(text, lang, must_include):
        p.append("wrong_script")
    norm = to_ascii_digits(text).lower()
    for s in must_include:
        if s.lower() not in norm:
            p.append(f"missing:{s}")
    for rx in forbid:
        if re.search(rx, norm, flags=re.I):
            p.append(f"forbidden:{rx}")
    return p


def final_problems(text: str, lang: str, allowed_latin: list[str] | None = None) -> list[str]:
    p = []
    if not text or len(text) < 2:
        p.append("empty")
    if len(text) > 700:
        p.append("too_long")
    if "<tool_call>" in text or "<tool_result" in text:
        p.append("markup")
    if not script_ok(text, lang, allowed_latin):
        p.append("wrong_script")
    return p
