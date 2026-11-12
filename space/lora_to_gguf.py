"""[F] F4: PEFT LoRA adapter (Tiny Aya / Cohere2) -> GGUF LoRA adapter for llama.cpp.

Small, dependency-light replacement for llama.cpp's convert_lora_to_gguf.py, written against the
`gguf` package from PyPI. Tensor names follow gguf's Cohere2 map; Cohere2 conversion does not
permute q/k, so LoRA A/B are copied as-is (A: [r, in], B: [out, r], like the official converter).

    python space/lora_to_gguf.py <adapter_dir> out.gguf [--outtype f16|f32]
    # publish: HF adapter repo -> gguf/desi-agent-sft-lora-f16.gguf in the same repo (what the Space loads)
    python space/lora_to_gguf.py --publish ToolW/desi-agent-tinyaya-sft-lora
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HF_TO_GGUF = {  # module name in HF Cohere2 -> gguf tensor name
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}
KEY_RE = re.compile(r"(?:base_model\.model\.)?model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def convert(adapter_dir: str | Path, out: str | Path, outtype: str = "f16") -> dict:
    import gguf
    from safetensors.numpy import load_file

    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("peft_type", "LORA") != "LORA" or cfg.get("use_dora") or cfg.get("use_rslora"):
        raise ValueError(f"only plain LoRA is supported, got {cfg.get('peft_type')} dora={cfg.get('use_dora')}")
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    dtype = np.float16 if outtype == "f16" else np.float32

    pairs: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for k, v in tensors.items():
        m = KEY_RE.match(k)
        if not m:
            raise ValueError(f"unexpected adapter tensor {k}")
        layer, module, ab = int(m[1]), m[2], m[3]
        if module not in HF_TO_GGUF:
            raise ValueError(f"no gguf mapping for module {module}")
        pairs.setdefault((layer, module), {})[ab] = v

    w = gguf.GGUFWriter(str(out), arch="cohere2")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(cfg["lora_alpha"]))
    ranks = set()
    for (layer, module), ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise ValueError(f"layer {layer} {module}: missing lora_A or lora_B")
        a, b = ab["A"], ab["B"]
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"layer {layer} {module}: rank mismatch {a.shape} vs {b.shape}")
        ranks.add(a.shape[0])
        name = f"blk.{layer}.{HF_TO_GGUF[module]}.weight"
        w.add_tensor(f"{name}.lora_a", a.astype(dtype))
        w.add_tensor(f"{name}.lora_b", b.astype(dtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"pairs": len(pairs), "ranks": sorted(ranks), "alpha": cfg["lora_alpha"], "out": str(out)}


def publish(repo: str, outtype: str = "f16") -> dict:
    """Convert the adapter stored in HF model repo `repo` and upload the GGUF to gguf/ in the same repo."""
    import tempfile
    from huggingface_hub import HfApi, snapshot_download

    src = snapshot_download(repo, allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
    with tempfile.TemporaryDirectory() as d:
        name = f"desi-agent-sft-lora-{outtype}.gguf"
        info = convert(src, Path(d) / name, outtype)
        HfApi().upload_file(path_or_fileobj=str(Path(d) / name), path_in_repo=f"gguf/{name}", repo_id=repo,
                            commit_message="[F] F4: GGUF LoRA for llama.cpp (demo Space)")
    return {**info, "out": f"{repo}/gguf/{name}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--outtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--publish", metavar="HF_REPO", help="convert the adapter in this HF repo and upload it there")
    a = ap.parse_args()
    if a.publish:
        print(json.dumps(publish(a.publish, a.outtype)))
    elif a.adapter_dir and a.out:
        print(json.dumps(convert(a.adapter_dir, a.out, a.outtype)))
    else:
        ap.error("give <adapter_dir> <out> or --publish HF_REPO")


if __name__ == "__main__":
    main()
