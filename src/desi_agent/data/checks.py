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


# Common function words of romanized Hindi / Bengali. Pure English text contains almost none of them.
HI_EN_MARKERS = {"hai", "hain", "ho", "hoga", "mera", "meri", "mere", "mujhe", "maine", "aap", "aapka", "aapke", "aapki", "karo",
                 "kar", "karna", "karke", "kardo", "kya", "kyun", "kaise", "nahi", "nahin", "ka", "ki", "ke", "ko", "se", "mein", "me",
                 "chahiye", "tha", "thi", "gaya", "gayi", "diya", "bhai", "yaar", "jaldi", "abhi", "kitna", "kitne", "batao",
                 "bataiye", "dikhao", "bhejo", "bhej", "wala", "wali", "aur", "lekin", "toh", "bhi", "raha", "rahi", "hua", "kab", "kahan"}
BN_EN_MARKERS = {"ami", "aami", "amar", "amake", "apni", "apnar", "apnake", "ache", "ase", "achhe", "koro", "korte", "korun",
                 "kore", "korechi", "korbo", "kintu", "ki", "na", "nai", "ta", "ti", "theke", "hobe", "hoye", "hoyeche", "chai",
                 "chaai", "ekta", "dao", "din", "bolun", "bolo", "kothay", "koto", "keno", "kibhabe", "ar", "aar", "tumi", "tomar",
                 "dekhao", "dekhte", "pathao", "pathate", "jodi", "tahole", "ekhon", "taka", "gelo", "geche", "hoy", "plz"}


def romanized_ok(text: str, lang: str, allowed_latin: list[str] | None = None) -> bool:
    """hi-en / bn-en must really be romanized Hindi / Bengali, not plain English.
    Values quoted from tool results (merchant names etc.) don't count against the ratio."""
    if lang not in ("hi-en", "bn-en"):
        return True
    if allowed_latin:
        text = _strip_allowed(text, allowed_latin)
    toks = [t.strip(".,!?:;()'\"").lower() for t in text.split()]
    toks = [t for t in toks if t.isalpha()]
    markers = HI_EN_MARKERS if lang == "hi-en" else BN_EN_MARKERS
    hits = sum(t in markers for t in toks)
    return hits >= 2 and hits / max(1, len(toks)) >= 0.15


# The assistant must talk TO the customer (second person), never as the customer.
SECOND_PERSON = {
    "en": {"you", "your", "you're", "you've", "yours"},
    "hi": {"आप", "आपका", "आपके", "आपकी", "आपको", "आपने"},
    "bn": {"আপনার", "আপনি", "আপনাকে", "আপনারা"},
    "hi-en": {"aap", "aapka", "aapke", "aapki", "aapko", "aapne", "apka", "apke", "apki", "apko", "aapse"},
    "bn-en": {"apnar", "apni", "apnake", "apnara", "tomar", "tumi", "tomake"},
}
# Possessives only ("my card", "amar card"): the bot may say "I can help" / "mujhe batayein", but never "my card".
FIRST_PERSON_CUSTOMER = {
    "en": {"my"},
    "hi": {"मेरा", "मेरी", "मेरे"},
    "bn": {"আমার"},
    "hi-en": {"mera", "meri", "mere"},
    "bn-en": {"amar"},
}


def addresses_customer(text: str, lang: str) -> bool:
    """False when the reply talks as the customer ("my card was blocked, I'll get a new one") instead of to them.
    Impersonal replies ("The EMI is ...") are fine."""
    toks = {t.strip(".,!?:;()'\"।").lower() for t in text.split()}
    return bool(toks & SECOND_PERSON[lang]) or not (toks & FIRST_PERSON_CUSTOMER[lang])


def repetitive(text: str, n: int = 3, max_repeats: int = 2) -> bool:
    toks = text.split()
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return any(grams.count(g) > max_repeats for g in set(grams))


def _overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


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
    if "\n" in text.strip():
        p.append("multiline")
    if not script_ok(text, lang, must_include):
        p.append("wrong_script")
    if not romanized_ok(text, lang):
        p.append("not_romanized_" + lang)
    if repetitive(text):
        p.append("repetitive")
    norm = to_ascii_digits(text).lower()
    for s in must_include:
        if s.lower() not in norm:
            p.append(f"missing:{s}")
    for rx in forbid:
        if re.search(rx, norm, flags=re.I):
            p.append(f"forbidden:{rx}")
    return p


def final_problems(text: str, lang: str, allowed_latin: list[str] | None = None, user_msg: str = "") -> list[str]:
    p = []
    if not text or len(text) < 2:
        p.append("empty")
    if len(text) > 700:
        p.append("too_long")
    if "<tool_call>" in text or "<tool_result" in text:
        p.append("markup")
    if not script_ok(text, lang, allowed_latin):
        p.append("wrong_script")
    if not romanized_ok(text, lang, allowed_latin):
        p.append("not_romanized_" + lang)
    if repetitive(text):
        p.append("repetitive")
    if user_msg and _overlap(text, user_msg) > 0.8:
        p.append("echoes_user")
    if not addresses_customer(text, lang):
        p.append("not_addressing_customer")
    return p
