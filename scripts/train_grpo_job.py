"""[F] Stage 2: multi-turn GRPO signal check from rft2, in ONE Hugging Face Job (src/desi_agent/train/grpo_mt.py).

Online GRPO on data/round2/langswitch_train.jsonl (train only), reward = rewards.reward_lang (R's audited reward), full
turn-2 episodes through the shared agent loop + mock bank. Starts from the rft2 adapter and pushes the result to a NEW
repo. Every rollout (with reward parts: task success, language, echo, 'ask first' check, reply length) plus the per-step
log go to ToolW/desi-agent-training-logs round2/<run>/ for R's hacking audit; adapters every --save-every steps go to
runs/<run>/step-<n>/.

    # launch (prints the job URL)
    python scripts/train_grpo_job.py --run grpo-mt --flavor a100-large --timeout 3h
    # CPU smoke test of the training step on a tiny random model with Tiny Aya's tokenizer (no GPU, no upload)
    python scripts/train_grpo_job.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GITHUB = "CyberRik/desi-agent"
BASE = "CohereLabs/tiny-aya-global"
START = "ToolW/desi-agent-tinyaya-rft2-lora"
LOGS = "ToolW/desi-agent-training-logs"
IMAGE = "vllm/vllm-openai:v0.33.0"  # same image as the other jobs; training runs in its own py3.12 venv
PROTECTED = {"ToolW/desi-agent-tinyaya-sft-lora", "ToolW/desi-agent-tinyaya-rft-lora",
             "ToolW/desi-agent-tinyaya-rft2-lora"}
KEYS = ("group", "prompts_per_step", "steps", "lr", "temperature", "gen_batch", "save_every", "seed")


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_trainable(adapter_path: str | None, model_id: str = BASE, model=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from desi_agent.train.policy import DEFAULT_GEN, HFPolicy, native_bf16
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model is None:
        dtype = torch.bfloat16 if native_bf16() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True) if adapter_path else model
    model.eval()
    return HFPolicy(model, tok, dict(DEFAULT_GEN), model_id=model_id, adapter=adapter_path)


def inside(a) -> None:
    from huggingface_hub import HfApi, snapshot_download
    from desi_agent.train.grpo_mt import train_grpo_mt
    api = HfApi()
    out = Path("/out") / a.run
    remote = f"round2/{a.run}"

    def up(path: Path, dest: str, folder: bool = False) -> None:
        for attempt in range(3):
            try:
                if folder:
                    api.upload_folder(folder_path=str(path), repo_id=LOGS, repo_type="dataset", path_in_repo=dest,
                                      commit_message=f"[F] {a.run}: {dest}")
                else:
                    api.upload_file(path_or_fileobj=str(path), repo_id=LOGS, repo_type="dataset", path_in_repo=dest,
                                    commit_message=f"[F] {a.run}: {dest}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[F] upload failed ({e}), retry {attempt + 1}", flush=True)
                time.sleep(20)

    repo, _, rev = a.start.partition("@")
    pol = load_trainable(snapshot_download(repo, revision=rev or None))
    rows = jl(ROOT / "data/round2/langswitch_train.jsonl")
    cfg = {k: getattr(a, k) for k in KEYS}

    def on_save(step: int) -> None:
        d = out / f"step-{step}"
        pol.model.save_pretrained(d)
        pol.tok.save_pretrained(d)
        up(d, f"runs/{a.run}/step-{step}", folder=True)
        for f in ("log.json", "rollouts.jsonl"):
            up(out / f, f"{remote}/{f}")

    t0 = time.time()
    log = train_grpo_mt(pol, rows, cfg, out, on_save=on_save)
    manifest = {"task": "grpo_mt", "git_sha": os.environ.get("DESI_GIT_SHA"), "base": BASE, "start_adapter": a.start,
                "data": "data/round2/langswitch_train.jsonl", "reward": "desi_agent.eval.rewards.reward_lang",
                "config": cfg, "minutes": round((time.time() - t0) / 60, 1), "first": log[0], "last": log[-1]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    up(out / "manifest.json", f"{remote}/manifest.json")
    final = out / f"step-{a.steps}"
    api.create_repo(a.push_to, exist_ok=False, private=False)
    api.upload_folder(folder_path=str(final), repo_id=a.push_to, commit_message=f"[F] {a.run}: GRPO from {a.start}")
    print(f"[F] done in {manifest['minutes']} min -> {a.push_to}", flush=True)


def smoke() -> None:
    """Tiny random Cohere model + Tiny Aya tokenizer on CPU: rollouts, masking and one update run end to end."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    from desi_agent.train import grpo_mt
    cfg = AutoConfig.from_pretrained(BASE)
    for k, v in {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 2,
                 "num_key_value_heads": 1, "head_dim": 32}.items():
        setattr(cfg, k, v)
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = cfg.layer_types[:2]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    pol = load_trainable(None, model=model)
    rows = jl(ROOT / "data/round2/langswitch_train.jsonl")[:2]
    # masking: the trained tokens must be exactly the gold turn-2 assistant text
    row = rows[0]
    turn2 = row["messages"][len(row["prefix"]):]
    s = grpo_mt.sequence(pol.tok, row, turn2, 10**6)
    trained = pol.tok.decode([t for t, l in zip(s["input_ids"], s["labels"]) if l != -100])
    want = [m for m in fmt_chat(turn2) if m["role"] == "assistant"]
    assert all(w["content"] in trained for w in want), (trained[:300], want)
    print("[F] smoke: masking ok,", sum(l != -100 for l in s["labels"]), "trained tokens of", len(s["labels"]))
    # one full step with a forced reward spread (a random model scores 0 everywhere)
    real = grpo_mt.score
    import itertools
    tick = itertools.count()
    grpo_mt.score = lambda t2, r: {**real(t2, r), "reward": float(next(tick) % 3)}
    log = grpo_mt.train_grpo_mt(pol, rows, {"group": 3, "prompts_per_step": 2, "steps": 1, "max_new_tokens": 12,
                                            "gen_batch": 6}, ROOT / "outputs/grpo_smoke")
    assert log[0]["n_trained_seqs"] > 0 and log[0]["grad_norm"] > 0, log
    print("[F] smoke: one GRPO step ok", json.dumps(log[0]))


def fmt_chat(msgs):
    from desi_agent import fmt
    return fmt.to_chat(msgs, with_system=False)


def load_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    from huggingface_hub import get_token
    tok = get_token()
    if not tok:
        sys.exit("[F] no HF token")
    return tok


def launch(a) -> None:
    from huggingface_hub import HfApi
    if a.push_to in PROTECTED:
        sys.exit(f"[F] {a.push_to} is protected; push to a new repo")
    token = load_token()
    api = HfApi(token=token)
    if api.repo_exists(a.push_to):
        sys.exit(f"[F] {a.push_to} already exists; pick a new --push-to")
    if any(f.startswith(f"round2/{a.run}/") for f in api.list_repo_files(LOGS, repo_type="dataset")):
        sys.exit(f"[F] round2/{a.run}/ already exists on {LOGS}; pick a new --run")
    repo, _, rev = a.start.partition("@")
    start = f"{repo}@{api.model_info(repo, revision=rev or None).sha}"
    sha = subprocess.run(["git", "rev-parse", a.git_sha or "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    inner = ["/tmp/venv/bin/python", "scripts/train_grpo_job.py", "--inside", "--run", a.run, "--start", start,
             "--push-to", a.push_to] + [x for k in KEYS for x in (f"--{k.replace('_', '-')}", str(getattr(a, k)))]
    script = " && ".join([
        "set -e", "mkdir -p /w", "cd /w",
        f"python3 -c \"import urllib.request,tarfile,io; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen("
        f"'https://codeload.github.com/{GITHUB}/tar.gz/{sha}').read())).extractall('/w')\"",
        "cd /w/desi-agent-*",
        "(command -v git || (apt-get update -qq && apt-get install -y -qq git >/dev/null))",
        "python3 -m pip install -q --root-user-action=ignore uv",
        "env -u UV_SYSTEM_PYTHON -u UV_PYTHON uv venv -q -p 3.12 --python-preference managed /tmp/venv",
        "uv pip install --python /tmp/venv/bin/python -q --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu126 torch==2.11.0 -e '.[train,eval]'",
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
        " ".join(inner)])
    job = api.run_job(image=IMAGE, command=["bash", "-c", script], flavor=a.flavor, timeout=a.timeout,
                      secrets={"HF_TOKEN": token},
                      env={"DESI_GIT_SHA": sha, "PYTHONUNBUFFERED": "1", "PIP_BREAK_SYSTEM_PACKAGES": "1",
                           "TOKENIZERS_PARALLELISM": "false"},
                      labels={"project": "desi-agent", "run": a.run[:60]}, name=f"desi-grpo-{a.run}"[:60])
    print(f"[F] job {job.id} on {a.flavor}, timeout {a.timeout}\n    {job.url}\n"
          f"    code {GITHUB}@{sha[:12]}  start {start}  -> {a.push_to}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="grpo-mt")
    p.add_argument("--start", default=START, help="adapter repo[@rev] to start from")
    p.add_argument("--push-to", default="ToolW/desi-agent-tinyaya-grpo-mt-lora")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234, help="prompt order + sampling seed (grpo-mt used 1234, sampling unseeded)")
    p.add_argument("--flavor", default="a100-large")
    p.add_argument("--timeout", default="3h")
    p.add_argument("--git-sha")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.inside:
        inside(a)
    elif a.smoke:
        smoke()
    else:
        launch(a)


if __name__ == "__main__":
    main()
