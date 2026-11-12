"""[F] F4: PEFT LoRA -> GGUF converter round trip on a tiny fake adapter (skipped without gguf/safetensors)."""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
st = pytest.importorskip("safetensors.numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))
from lora_to_gguf import convert  # noqa: E402


def test_round_trip(tmp_path):
    r, d_in, d_out = 4, 8, 6
    t = {}
    for layer in range(2):
        for mod in ("self_attn.q_proj", "mlp.down_proj"):
            t[f"base_model.model.model.layers.{layer}.{mod}.lora_A.weight"] = np.random.rand(r, d_in).astype(np.float32)
            t[f"base_model.model.model.layers.{layer}.{mod}.lora_B.weight"] = np.random.rand(d_out, r).astype(np.float32)
    st.save_file(t, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "lora_alpha": 32, "r": r}))
    info = convert(tmp_path, tmp_path / "out.gguf", "f32")
    assert info["pairs"] == 4 and info["ranks"] == [r]

    rd = gguf.GGUFReader(str(tmp_path / "out.gguf"))
    names = {x.name: x for x in rd.tensors}
    assert set(names) == {f"blk.{i}.{m}.weight.lora_{ab}" for i in range(2) for m in ("attn_q", "ffn_down") for ab in "ab"}
    a = names["blk.1.attn_q.weight.lora_a"]
    assert np.allclose(np.asarray(a.data).reshape(r, d_in), t["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"])
    kv = {f.name: f for f in rd.fields.values()}
    assert bytes(kv["general.type"].parts[-1]).decode() == "adapter"
    assert bytes(kv["adapter.type"].parts[-1]).decode() == "lora"
    assert float(kv["adapter.lora.alpha"].parts[-1][0]) == 32.0


def test_rejects_unknown_module(tmp_path):
    st.save_file({"base_model.model.model.layers.0.self_attn.rotary.lora_A.weight": np.zeros((2, 2), np.float32)},
                 str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "lora_alpha": 8}))
    with pytest.raises(ValueError):
        convert(tmp_path, tmp_path / "x.gguf")
