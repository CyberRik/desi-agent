"""[F] Chat-template helpers shared by inference (policy.py) and SFT (sft.py).

Training and inference must turn a `fmt.to_chat` chat into tokens the *same* way, so both go
through `apply_template` here. If the tokenizer's template rejects the `system` role, the system
prompt is folded into the first user turn (decided once per tokenizer, see `supports_system`).
"""
from __future__ import annotations

from functools import lru_cache


def fold_system(chat: list[dict]) -> list[dict]:
    """[system, user, ...] -> [user(system + user), ...]. No-op if there is no system turn."""
    if not chat or chat[0]["role"] != "system":
        return chat
    sys, rest = chat[0]["content"], chat[1:]
    if rest and rest[0]["role"] == "user":
        return [{"role": "user", "content": f"{sys}\n\n{rest[0]['content']}"}, *rest[1:]]
    return [{"role": "user", "content": sys}, *rest]


@lru_cache(maxsize=8)
def _supports_system(tok_id: int, tok) -> bool:  # tok_id keeps the cache key hashable/stable
    try:
        tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                                tokenize=False, add_generation_prompt=True)
        return True
    except Exception:  # noqa: BLE001 - jinja raises TemplateError, any failure means "no"
        return False


def supports_system(tok) -> bool:
    return _supports_system(id(tok), tok)


def apply_template(tok, chat: list[dict], *, add_generation_prompt: bool, tokenize: bool = False):
    """`tokenizer.apply_chat_template` with the system-role fallback applied."""
    if not supports_system(tok):
        chat = fold_system(chat)
    return tok.apply_chat_template(chat, tokenize=tokenize, add_generation_prompt=add_generation_prompt)


def tokenize_sft(tok, chat: list[dict], max_len: int = 2048) -> dict:
    """Tokenize a full chat for SFT with loss only on assistant turns.

    Uses the prefix-difference trick: the tokens of assistant turn i are
    render(chat[:i+1]) minus render(chat[:i], add_generation_prompt=True). Works for any chat
    template without needing `{% generation %}` markers. Returns {input_ids, attention_mask, labels}.
    Rows longer than max_len are truncated from the right (labels too).
    """
    ids: list[int] = []
    labels: list[int] = []
    prev_text = ""
    for i, m in enumerate(chat):
        if m["role"] == "assistant":
            prompt_text = apply_template(tok, chat[:i], add_generation_prompt=True)
            full_text = apply_template(tok, chat[: i + 1], add_generation_prompt=False)
            if not full_text.startswith(prompt_text) or not prompt_text.startswith(prev_text):
                raise ValueError("chat template is not prefix-stable; cannot mask assistant turns")
            ctx = tok(prompt_text[len(prev_text):], add_special_tokens=False)["input_ids"]
            ans = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
            ids += ctx + ans
            labels += [-100] * len(ctx) + ans
            prev_text = full_text
    ids, labels = ids[:max_len], labels[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
