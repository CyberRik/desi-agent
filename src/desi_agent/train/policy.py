"""[F] F1: load_policy(model_id, adapter, **gen_kw) -> Policy. See contracts/API.md.

    from desi_agent.api import load_policy, evaluate
    pol = load_policy()                                    # base Tiny Aya Global
    pol = load_policy(adapter="<HF_NS>/desi-agent-tinyaya-sft-lora")
    evaluate(pol, split="test", model_tag="tinyaya-global-base")

The returned policy takes a *rendered* chat (output of `fmt.to_chat`) and returns raw assistant
text. Greedy by default and stateless across calls. `policy.batch(chats)` runs left-padded
batched generation.
"""
from __future__ import annotations

from .chat import apply_template

DEFAULT_MODEL = "CohereLabs/tiny-aya-global"
# Stop right after a tool call: the model otherwise goes on to hallucinate its own <tool_result>.
# fmt.parse_assistant only reads the first call, so this changes speed, not scores.
DEFAULT_GEN = {"max_new_tokens": 256, "do_sample": False, "stop_strings": ["</tool_call>"]}


class HFPolicy:
    def __init__(self, model, tok, gen_kw: dict, *, model_id: str, adapter: str | None):
        self.model, self.tok, self.gen_kw = model, tok, gen_kw
        self.model_id, self.adapter = model_id, adapter

    def _encode(self, chats: list[list[dict]]):
        texts = [apply_template(self.tok, c, add_generation_prompt=True) for c in chats]
        # The template already inserts BOS (if any) as text, so don't add special tokens again.
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)

    def batch(self, chats: list[list[dict]]) -> list[str]:
        import torch

        enc = self._encode(chats)
        with torch.inference_mode():
            out = self.model.generate(**enc, pad_token_id=self.tok.pad_token_id, tokenizer=self.tok, **self.gen_kw)
        new = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]

    def __call__(self, chat: list[dict]) -> str:
        return self.batch([chat])[0]


def native_bf16() -> bool:
    """True only on GPUs with bf16 hardware (compute capability >= 8, Ampere+).

    `torch.cuda.is_bf16_supported()` is True on a T4 too (emulated), and emulated bf16 matmuls
    (magma_sgemmEx) made F3 training ~4.5x slower on Kaggle T4s than fp16.
    """
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def load_policy(model_id: str = DEFAULT_MODEL, adapter: str | None = None, *, load_in_4bit: bool = False,
                dtype: str = "auto", device_map: str | dict | None = "auto", **gen_kw) -> HFPolicy:
    """HF transformers policy. `adapter` = local path or HF repo of a PEFT LoRA (applied on top)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw: dict = {"device_map": device_map}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        # T4 has no bf16 hardware; fall back to fp16 there.
        if dtype == "auto":
            kw["dtype"] = torch.bfloat16 if native_bf16() \
                else (torch.float16 if torch.cuda.is_available() else torch.float32)
        else:
            kw["dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return HFPolicy(model, tok, {**DEFAULT_GEN, **gen_kw}, model_id=model_id, adapter=adapter)
