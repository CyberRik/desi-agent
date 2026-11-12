"""[F] Where does Tiny Aya's time go on a T4? Times forward / forward+backward on one real-length row.

    python scripts/train_profile.py --out outputs/profile
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def timed(fn, n=3):
    import torch
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return round((time.time() - t) / n, 3)


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--tokens", type=int, default=1450)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res: dict = {"gpu": torch.cuda.get_device_name(0), "tokens": a.tokens}
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = torch.randint(1000, 20000, (1, a.tokens), device="cuda")
    labels = torch.full_like(ids, -100)
    labels[:, -90:] = ids[:, -90:]  # ~6% trained tokens, like our rows
    res["vocab"] = len(tok)

    for attn in ("sdpa", "eager"):
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map={"": 0}, attn_implementation=attn)
        m.eval()
        res[f"{attn}_attn_impl_used"] = m.config._attn_implementation
        res[f"{attn}_param_dtypes"] = sorted({str(p.dtype) for p in m.parameters()})
        with torch.no_grad():
            res[f"{attn}_fwd_nograd_s"] = timed(lambda: m(input_ids=ids))
            res[f"{attn}_fwd_nograd_keep90_s"] = timed(lambda: m(input_ids=ids, logits_to_keep=90))
        if attn == "sdpa":
            cfg = m.config
            res["config"] = {k: getattr(cfg, k, None) for k in ("hidden_size", "num_hidden_layers", "sliding_window",
                                                                 "tie_word_embeddings", "logit_scale", "torch_dtype")}
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
            pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                                 "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
            pm.train()
            res["lora_param_dtypes"] = sorted({str(p.dtype) for p in pm.parameters() if p.requires_grad})

            def step():
                with torch.autocast("cuda", dtype=torch.float16):
                    out = pm(input_ids=ids, labels=labels)
                out.loss.backward()
                pm.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            res["train_fwd_bwd_gc_s"] = timed(step, n=2)
            res["train_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            del pm
        del m
        torch.cuda.empty_cache()
        print("[F] profile so far:", json.dumps(res), flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(res, indent=2))
    print("[F] profile:", json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
