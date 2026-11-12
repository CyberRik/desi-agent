"""[F] F2/F3: train_sft(cfg) -> {"adapter_path", "hf_repo", "metrics"}. See contracts/API.md.

AgentTune distillation: episodes -> full-tier EventLogs -> `Project.distill(student, trainer_factory=...)`.
Our trainer_factory maps each AgentTune SFT row back to the `fmt.to_chat` chat (`convert.row_to_chat`),
tokenizes it with the student's chat template, and trains a LoRA only on assistant tokens
(`chat.tokenize_sft`). Trainer = plain transformers `Trainer` + PEFT (boring, version-proof).

    from desi_agent.api import train_sft
    train_sft({"split": "seed", "max_steps": 20, "out_dir": "outputs/sft_smoke"})

F3 on Kaggle T4x2: launched with `torchrun --nproc_per_node 2` (DDP, one full LoRA replica per GPU).
Checkpoints (adapter + optimizer + trainer state) are uploaded to `ckpt_repo` under
`checkpoints/<run_name>/latest/` every `save_steps`, and `resume=True` continues from there, so a
run can outlive Kaggle's 12 h session limit by resuming in a new session on either account.
`max_hours` stops cleanly (save + upload) before the session is killed.
"""
from __future__ import annotations

import json
import os
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
    "attn_implementation": "sdpa",
    "group_by_length": True,     # less padding when batch_size > 1
    "out_dir": "outputs/sft",
    "push_to": None,             # e.g. "ToolW/desi-agent-tinyaya-sft-lora" (only after a complete run)
    "run_name": "sft",           # checkpoints/<run_name>/ in ckpt_repo
    "ckpt_repo": None,           # e.g. "ToolW/desi-agent-training-logs" (HF dataset); None = local only
    "save_steps": 50,
    "resume": False,             # continue from checkpoints/<run_name>/latest in ckpt_repo
    "max_hours": None,           # stop + checkpoint after this many hours of training (Kaggle limit is 12)
    "seed": SEED,
}


def _rank() -> tuple[int, int, bool]:
    """(local_rank, world_size, is_main) from torchrun env vars; (0, 1, True) without torchrun."""
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return local, world, int(os.environ.get("RANK", 0)) == 0


def _ckpt_path(run_name: str) -> str:
    return f"checkpoints/{run_name}/latest"


def _download_checkpoint(repo: str, run_name: str, dest: Path) -> Path | None:
    """Fetch checkpoints/<run_name>/latest from the HF dataset `repo`; None if there is none."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    sub = _ckpt_path(run_name)
    try:
        local = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{sub}/*"], local_dir=str(dest))
    except RepositoryNotFoundError:
        return None
    path = Path(local) / sub
    return path if (path / "trainer_state.json").exists() else None


def _callbacks(c: dict):
    import torch
    from transformers import TrainerCallback

    class HubCheckpoint(TrainerCallback):
        """After each save, rank 0 mirrors the newest checkpoint-N folder to ckpt_repo (replacing the old one)."""

        def on_save(self, args, state, control, **kw):
            if not (c["ckpt_repo"] and state.is_world_process_zero):
                return
            from huggingface_hub import HfApi
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not folder.exists():
                return
            api = HfApi()
            api.create_repo(c["ckpt_repo"], repo_type="dataset", exist_ok=True, private=False)
            api.upload_folder(folder_path=str(folder), repo_id=c["ckpt_repo"], repo_type="dataset",
                              path_in_repo=_ckpt_path(c["run_name"]), delete_patterns=["*"],
                              commit_message=f"[F] {c['run_name']} checkpoint step {state.global_step}")
            print(f"[F] checkpoint step {state.global_step} -> {c['ckpt_repo']}/{_ckpt_path(c['run_name'])}", flush=True)

    class TimeBudget(TrainerCallback):
        """Stop (and save) once max_hours have passed. All ranks agree via all_reduce, so DDP never hangs."""

        def __init__(self):
            self.t0 = time.time()
            self.hit = False

        def on_step_end(self, args, state, control, **kw):
            if not c["max_hours"]:
                return
            over = torch.tensor([float(time.time() - self.t0 > c["max_hours"] * 3600)],
                                device="cuda" if torch.cuda.is_available() else "cpu")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(over, op=torch.distributed.ReduceOp.MAX)
            if over.item() > 0:
                self.hit = True
                control.should_training_stop = True
                control.should_save = True
                if state.is_world_process_zero:
                    print(f"[F] time budget {c['max_hours']} h reached at step {state.global_step}, stopping", flush=True)

    return [HubCheckpoint(), TimeBudget()]


class _ResumingTrainer:
    """What the factory hands to AgentTune: `distill` calls `.train()`, we add `resume_from_checkpoint`."""

    def __init__(self, trainer, resume_from: str | None):
        self.trainer, self.resume_from = trainer, resume_from

    def train(self):
        return self.trainer.train(resume_from_checkpoint=self.resume_from)


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

    def __init__(self, cfg: dict, val_rows: list[dict] | None = None, resume_from: str | None = None):
        self.cfg, self.val_rows, self.resume_from = cfg, val_rows, resume_from
        self.model = self.tok = self.trainer = None
        self.callbacks: list = []
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

        local_rank, world, is_main = _rank()
        kw: dict = {"device_map": {"": local_rank} if cuda else None}
        if c["attn_implementation"]:
            kw["attn_implementation"] = c["attn_implementation"]
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
        if is_main:
            print("[F] sft data:", self.data_stats, "world_size:", world, flush=True)

        args = TrainingArguments(
            output_dir=c["out_dir"], num_train_epochs=c["epochs"], max_steps=c["max_steps"],
            per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=c["batch_size"],
            gradient_accumulation_steps=c["grad_accum"], learning_rate=c["lr"], lr_scheduler_type="cosine",
            warmup_steps=c["warmup_ratio"], logging_steps=c["logging_steps"],
            eval_strategy="steps" if val else "no", eval_steps=c["eval_steps"],
            save_strategy="steps", save_steps=c["save_steps"], save_total_limit=1, save_only_model=False,
            train_sampling_strategy="group_by_length" if c["group_by_length"] and c["batch_size"] > 1 else "random",
            ddp_find_unused_parameters=False if world > 1 else None,
            report_to=[], seed=c["seed"], data_seed=c["seed"],
            bf16=bf16, fp16=cuda and not bf16, use_cpu=not cuda,
            gradient_checkpointing=False,  # enabled on the model above
            remove_unused_columns=False)
        self.callbacks = _callbacks(c)
        self.trainer = Trainer(model=self.model, args=args, train_dataset=train, eval_dataset=val,
                               data_collator=Collator(self.tok.pad_token_id), callbacks=self.callbacks)
        if self.resume_from and is_main:
            print(f"[F] resuming from {self.resume_from}", flush=True)
        return _ResumingTrainer(self.trainer, self.resume_from)


def _load(name: str | None, limit: int | None = None) -> list[dict] | None:
    from ..api import load_split
    if not name:
        return None
    if name == "test":
        raise ValueError("never train on the test split (CLAUDE.md §7)")
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    try:
        eps = load_split(name)
    except (FileNotFoundError, EntryNotFoundError, RepositoryNotFoundError):
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

    resume_from = None
    if c["resume"]:
        if not c["ckpt_repo"]:
            raise ValueError("resume=True needs ckpt_repo")
        found = _download_checkpoint(c["ckpt_repo"], c["run_name"], out / "resume")
        resume_from = str(found) if found else None
        print(f"[F] resume: {'from ' + resume_from if found else 'no checkpoint yet, starting fresh'}", flush=True)

    factory = TinyAyaSFTFactory(c, val_rows, resume_from)
    t0 = time.time()
    result = proj.distill(c["model_id"], trainer_factory=factory)
    runtime = time.time() - t0

    state = factory.trainer.state
    stopped_early = any(getattr(cb, "hit", False) for cb in factory.callbacks)
    if val_rows:
        eval_metrics = factory.trainer.evaluate()  # collective under DDP: every rank runs it
    if not _rank()[2]:
        return {"adapter_path": None, "hf_repo": None, "metrics": {}}  # non-main DDP ranks stop here

    adapter_path = out / "adapter"
    factory.model.save_pretrained(adapter_path)
    factory.tok.save_pretrained(adapter_path)
    metrics = {"train_loss": getattr(result, "training_loss", None), "runtime_s": round(runtime, 1),
               **{k: v for k, v in (getattr(result, "metrics", None) or {}).items() if isinstance(v, (int, float))}}
    metrics.update({"global_step": state.global_step, "max_steps": state.max_steps,
                    "complete": not stopped_early, "resumed_from": resume_from})
    if val_rows:
        metrics.update(eval_metrics)
    log_history = state.log_history

    manifest = {"task": "train_sft", "git_sha": _git_sha(), "model_id": c["model_id"], "seed": c["seed"],
                "config": c, "data": {"train_split": c["split"], "n_train_episodes": len(train_eps),
                                      "val_split": c["val_split"] if val_eps else None,
                                      "n_val_episodes": len(val_eps or []), **factory.data_stats},
                "metrics": metrics}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "log_history.json").write_text(json.dumps(log_history, indent=2, default=str))
    (adapter_path / "desi_agent_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if c["ckpt_repo"]:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(out), repo_id=c["ckpt_repo"], repo_type="dataset",
                              path_in_repo=f"runs/{c['run_name']}", allow_patterns=["manifest.json", "log_history.json"],
                              commit_message=f"[F] {c['run_name']} manifest + log history")
    hf_repo = None
    if c["push_to"] and stopped_early:
        print(f"[F] run stopped by time budget: not pushing to {c['push_to']}; resume with --set resume=true")
    elif c["push_to"]:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(c["push_to"], exist_ok=True, private=False)
        api.upload_folder(folder_path=str(adapter_path), repo_id=c["push_to"],
                          commit_message=f"[F] SFT adapter ({_git_sha()})")
        hf_repo = c["push_to"]
    print("[F] sft done:", json.dumps(metrics, default=str))
    return {"adapter_path": str(adapter_path), "hf_repo": hf_repo, "metrics": metrics}
