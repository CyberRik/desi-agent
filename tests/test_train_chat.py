"""[F] Tests for train/chat.py with a fake tokenizer (no torch / no model download)."""
from desi_agent import fmt
from desi_agent.data.splits import load_split
from desi_agent.train.chat import apply_template, fold_system, tokenize_sft


class FakeTok:
    """Char-level tokenizer with a simple prefix-stable chat template."""

    def __init__(self, system_ok: bool = True):
        self.system_ok = system_ok

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False):
        if not self.system_ok and any(m["role"] == "system" for m in chat):
            raise ValueError("System role not supported")
        s = "<s>" + "".join(f"<{m['role']}>{m['content']}</t>" for m in chat)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _chat():
    return fmt.to_chat(load_split("seed")[0]["messages"])


def test_fold_system():
    chat = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert fold_system(chat) == [{"role": "user", "content": "S\n\nU"}]
    assert fold_system(chat[1:]) == chat[1:]


def test_apply_template_falls_back_without_system_role():
    out = apply_template(FakeTok(system_ok=False), _chat(), add_generation_prompt=True)
    assert out.startswith("<s><user>") and "<system>" not in out


def test_tokenize_sft_masks_everything_but_assistant():
    tok, chat = FakeTok(), _chat()
    row = tokenize_sft(tok, chat, max_len=100_000)
    full = apply_template(tok, chat, add_generation_prompt=False)
    assert "".join(map(chr, row["input_ids"])) == full
    trained = "".join(chr(i) for i, l in zip(row["input_ids"], row["labels"]) if l != -100)
    want = "".join(m["content"] + "</t>" for m in chat if m["role"] == "assistant")
    assert trained == want
    assert '<tool_call>{"arguments"' in trained
