"""[F] F2/F3: train_sft(cfg) -> {"adapter_path", "hf_repo", "metrics"}. See contracts/API.md.

AgentTune distillation: episodes -> full-tier EventLogs -> `Project.distill(student, trainer_factory=...)`.
Our trainer_factory maps each AgentTune SFT row back to the `fmt.to_chat` chat (`convert.row_to_chat`),
tokenizes it with the student's chat template, and trains a LoRA only on assistant tokens
(`chat.tokenize_sft`). Trainer = plain transformers `Trainer` + PEFT (boring, version-proof).

    from desi_agent.api import train_sft
    train_sft({"split": "seed", "max_steps": 20, "out_dir": "outputs/sft_smoke"})
"""
from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

from .chat import tokenize_sft
from .convert import episodes_to_eventlogs, row_to_chat

SEED = 1234
ROOT = Path(__file__).resolve().parents[3]

DEFAULTS: dict = {
    "model_id": "CohereLabs/tiny-aya-global",
    "split": "train",            # training split (never "test")
    "val_split": "val",          # None to skip eval; skipped automatically if the split file is missing
    "limit": None,               # cap #train episodes (smoke runs)
    "max_len": 2048,
    "load_in_4bit": False,       # QLoRA fallback for small GPUs
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "epochs": 2, "max_steps": -1,
    "lr": 2e-4, "batch_size": 1, "grad_accum": 8, "warmup_ratio": 0.05,
    "gradient_checkpointing": True,
    "logging_steps": 5, "eval_steps": 50,
    "out_dir": "outputs/sft",
    "push_to": None,             # e.g. "ToolW/desi-agent-tinyaya-sft-lora"
    "seed": SEED,
}


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        import torch

        n = max(len(f["input_ids"]) for f in feats)
        pad = lambda xs, v: xs + [v] * (n - len(xs))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_id) for f in feats]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in feats]),
            "labels": torch.tensor([pad(f["labels"], -100) for f in feats]),
        }


class TinyAyaSFTFactory:
    """AgentTune `trainer_factory`: called by `Project.distill(model=..., train_dataset=rows)`."""

    def __init__(self, cfg: dict, val_rows: list[dict] | None = None):
        self.cfg, self.val_rows = cfg, val_rows
        self.model = self.tok = self.trainer = None
        self.data_stats: dict = {}

    def _tokenize(self, rows: list[dict]) -> list[dict]:
        out = [tokenize_sft(self.tok, row_to_chat(r), max_len=self.cfg["max_len"]) for r in rows]
        return [o for o in out if any(l != -100 for l in o["labels"])]

    def __call__(self, *, model: str, train_dataset: list[dict], **_):
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

        c = self.cfg
        cuda = torch.cuda.is_available()
        bf16 = cuda and torch.cuda.is_bf16_supported()
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw: dict = {"device_map": {"": 0} if cuda else None}
        if c["load_in_4bit"]:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        else:
            kw["dtype"] = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        if c["load_in_4bit"]:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=c["gradient_checkpointing"])
        elif c["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        names = {n.split(".")[-1] for n, _ in self.model.named_modules()}
        targets = [t for t in c["lora_targets"] if t in names]
        self.model = get_peft_model(self.model, LoraConfig(
            r=c["lora_r"], lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
            target_modules=targets, task_type="CAUSAL_LM"))

        train = self._tokenize(train_dataset)
        val = self._tokenize(self.val_rows) if self.val_rows else None
        lens = [len(t["input_ids"]) for t in train]
        self.data_stats = {"n_train_rows": len(train), "n_val_rows": len(val or []),
                           "max_tokens": max(lens), "mean_tokens": round(sum(lens) / len(lens), 1),
                           "n_truncated": sum(l >= c["max_len"] for l in lens), "lora_targets": targets}
        print("[F] sft data:", self.data_stats)

        args = TrainingArguments(
            output_dir=c["out_dir"], num_train_epochs=c["epochs"], max_steps=c["max_steps"],
            per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=c["batch_size"],
            gradient_accumulation_steps=c["grad_accum"], learning_rate=c["lr"], lr_scheduler_type="cosine",
            warmup_steps=c["warmup_ratio"], logging_steps=c["logging_steps"],
            eval_strategy="steps" if val else "no", eval_steps=c["eval_steps"],
            save_strategy="no", report_to=[], seed=c["seed"], data_seed=c["seed"],
            bf16=bf16, fp16=cuda and not bf16, use_cpu=not cuda,
            gradient_checkpointing=False,  # enabled on the model above
            remove_unused_columns=False)
        self.trainer = Trainer(model=self.model, args=args, train_dataset=train, eval_dataset=val,
                               data_collator=Collator(self.tok.pad_token_id))
        return self.trainer


def _load(name: str | None, limit: int | None = None) -> list[dict] | None:
    from ..api import load_split
    if not name:
        return None
    if name == "test":
        raise ValueError("never train on the test split (CLAUDE.md §7)")
    try:
        eps = load_split(name)
    except FileNotFoundError:
        print(f"[F] split {name!r} not found, skipping")
        return None
    return eps[:limit] if limit else eps


def train_sft(cfg: dict) -> dict:
    from agenttune.agentic import Project

    c = {**DEFAULTS, **(cfg or {})}
    random.seed(c["seed"])
    import torch
    torch.manual_seed(c["seed"])
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    train_eps = _load(c["split"], c["limit"])
    if not train_eps:
        raise FileNotFoundError(f"no training data for split {c['split']!r}")
    val_eps = _load(c["val_split"])

    proj = Project()
    for log in episodes_to_eventlogs(train_eps):
        proj.add_trajectory(log)
    val_rows = None
    if val_eps:
        vp = Project()
        for log in episodes_to_eventlogs(val_eps):
            vp.add_trajectory(log)
        val_rows = vp.sft_dataset()

    factory = TinyAyaSFTFactory(c, val_rows)
    t0 = time.time()
    result = proj.distill(c["model_id"], trainer_factory=factory)
    runtime = time.time() - t0

    adapter_path = out / "adapter"
    factory.model.save_pretrained(adapter_path)
    factory.tok.save_pretrained(adapter_path)
    metrics = {"train_loss": getattr(result, "training_loss", None), "runtime_s": round(runtime, 1),
               **{k: v for k, v in (getattr(result, "metrics", None) or {}).items() if isinstance(v, (int, float))}}
    if val_rows:
        metrics.update(factory.trainer.evaluate())
    log_history = factory.trainer.state.log_history

    manifest = {"task": "train_sft", "git_sha": _git_sha(), "model_id": c["model_id"], "seed": c["seed"],
                "config": c, "data": {"train_split": c["split"], "n_train_episodes": len(train_eps),
                                      "val_split": c["val_split"] if val_eps else None,
                                      "n_val_episodes": len(val_eps or []), **factory.data_stats},
                "metrics": metrics}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "log_history.json").write_text(json.dumps(log_history, indent=2, default=str))
    (adapter_path / "desi_agent_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    hf_repo = None
    if c["push_to"]:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(c["push_to"], exist_ok=True, private=False)
        api.upload_folder(folder_path=str(adapter_path), repo_id=c["push_to"],
                          commit_message=f"[F] SFT adapter ({_git_sha()})")
        hf_repo = c["push_to"]
    print("[F] sft done:", json.dumps(metrics, default=str))
    return {"adapter_path": str(adapter_path), "hf_repo": hf_repo, "metrics": metrics}
