"""[R] Run the RL signal probe (scripts/eval_rl_signal.py) as ONE Hugging Face Job and upload its raw samples.

The job downloads the pushed GitHub commit, serves Tiny Aya + the LoRA adapter on vLLM (fp16, prefix caching, LoRA
served as `rft`, the same server flags as the Kaggle runs), samples K full conversations per prompt at T=1.0, and
uploads samples.jsonl + summary.json + manifest.json to the HF dataset ToolW/desi-agent-training-logs under
runs/<run>/ (never overwrites: the launcher refuses an existing folder, the job falls back to runs/<run>-<job id>/).

    python scripts/eval_rl_signal_hfjob.py --run round2-rft-probe --adapter ToolW/desi-agent-tinyaya-rft-lora \
        --prompts data/round2/langswitch_train.jsonl,data/round2/grounding_train.jsonl --flavor l40sx1 --timeout 3h
    python scripts/eval_rl_signal_hfjob.py --fetch --run round2-rft-probe --out results/rl_signal_round2

HF_TOKEN comes from the environment or the repo's .env and is passed as a job secret (never printed).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "CyberRik/desi-agent"
LOGS = "ToolW/desi-agent-training-logs"
BASE = "CohereLabs/tiny-aya-global"
IMAGE = "vllm/vllm-openai:v0.33.0"
PORT = 8001


def load_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for d in [ROOT, *ROOT.parents]:
        env = d / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("[R] no HF_TOKEN in the environment or in a .env file")


def launch(a) -> None:
    from huggingface_hub import HfApi
    token = load_token()
    api = HfApi(token=token)
    if any(f.startswith(f"runs/{a.run}/") for f in api.list_repo_files(LOGS, repo_type="dataset")):
        sys.exit(f"[R] runs/{a.run}/ already exists on {LOGS}. Pick a new --run.")
    adapter_sha = api.model_info(a.adapter).sha
    sha = subprocess.run(["git", "rev-parse", a.git_sha], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://codeload.github.com/{GITHUB}/tar.gz/{sha}",
                                                      method="HEAD"), timeout=60)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[R] commit {sha} not on github.com/{GITHUB} ({e}). Push it first.")
    inner = ["python3", "scripts/eval_rl_signal_hfjob.py", "--inside", "--run", a.run, "--adapter",
             f"{a.adapter}@{adapter_sha}", "--prompts", a.prompts, "--k", str(a.k), "--workers", str(a.workers)]
    script = " && ".join([
        "set -e", "mkdir -p /w", "cd /w",
        f"python3 -c \"import urllib.request,tarfile,io; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen("
        f"'https://codeload.github.com/{GITHUB}/tar.gz/{sha}').read())).extractall('/w')\"",
        "cd /w/desi-agent-*",
        "python3 -m pip install -q --root-user-action=ignore -e .",
        " ".join(inner)])
    job = api.run_job(image=IMAGE, command=["bash", "-c", script], flavor=a.flavor, timeout=a.timeout,
                      secrets={"HF_TOKEN": token},
                      env={"DESI_GIT_SHA": sha, "PYTHONUNBUFFERED": "1", "PIP_BREAK_SYSTEM_PACKAGES": "1"},
                      labels={"project": "desi-agent", "run": a.run[:60]}, name=f"desi-rl-{a.run}"[:60])
    print(f"[R] job {job.id} on {a.flavor}, timeout {a.timeout}\n    {job.url}\n"
          f"    code {GITHUB}@{sha[:12]}  adapter {a.adapter}@{adapter_sha[:12]}", flush=True)


def fetch(a) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=load_token())
    prefix = f"runs/{a.run}/"
    files = [f for f in api.list_repo_files(LOGS, repo_type="dataset") if f.startswith(prefix)]
    if not files:
        sys.exit(f"[R] nothing under {prefix} on {LOGS} yet")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = out / f[len(prefix):]
        dest.write_bytes(Path(hf_hub_download(LOGS, f, repo_type="dataset", token=api.token)).read_bytes())
        print(f"[R] {f} -> {dest}")


def serve(extra: str, log: Path) -> subprocess.Popen:
    p = subprocess.Popen(f"vllm serve {BASE} --port {PORT} --tensor-parallel-size 1 --dtype half "
                         f"--max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching {extra}",
                         shell=True, stdout=log.open("w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health")
            print("[R] vLLM up", flush=True)
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                break
            time.sleep(10)
    print(log.read_text(errors="replace")[-6000:], flush=True)
    raise RuntimeError("vLLM failed to start")


def inside(a) -> None:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    t0 = time.time()
    out = Path("/out")
    out.mkdir(parents=True, exist_ok=True)
    remote = f"runs/{a.run}"
    if any(f.startswith(remote + "/") for f in api.list_repo_files(LOGS, repo_type="dataset")):
        remote = f"runs/{a.run}-{os.environ.get('JOB_ID') or int(t0)}"
    repo, _, rev = a.adapter.partition("@")
    path = snapshot_download(repo, revision=rev or None)
    srv = serve(f"--enable-lora --max-lora-rank 64 --lora-modules rft={path}", out / "vllm.log")
    t1 = time.time()
    try:
        subprocess.run([sys.executable, "scripts/eval_rl_signal.py", "--openai-base-url", f"http://localhost:{PORT}/v1",
                        "--openai-model", "rft", "--k", str(a.k), "--workers", str(a.workers), "--temperature", "1.0",
                        "--prompts", a.prompts, "--out", str(out)], check=True)
    finally:
        srv.terminate()
        man = {"script": "scripts/eval_rl_signal.py", "launcher": "scripts/eval_rl_signal_hfjob.py",
               "git_sha": os.environ.get("DESI_GIT_SHA"), "image": IMAGE, "seed": 1234, "base_model": BASE,
               "base_revision": api.model_info(BASE).sha, "adapter": a.adapter, "served_as": "rft",
               "prompts": a.prompts, "k": a.k, "temperature": 1.0, "max_new_tokens": 256, "dtype": "float16",
               "workers": a.workers, "setup_s": round(t1 - t0), "sampling_s": round(time.time() - t1)}
        (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        api.upload_folder(folder_path=str(out), repo_id=LOGS, repo_type="dataset", path_in_repo=remote,
                          commit_message=f"[R] RL signal probe {a.run}: samples + reward stats")
        print(f"[R] uploaded to {LOGS}/{remote}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--adapter", default="ToolW/desi-agent-tinyaya-rft-lora")
    p.add_argument("--prompts", default="data/round2/langswitch_train.jsonl,data/round2/grounding_train.jsonl")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--flavor", default="l40sx1")
    p.add_argument("--timeout", default="3h")
    p.add_argument("--git-sha", default="HEAD")
    p.add_argument("--inside", action="store_true")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--out", default="results/rl_signal_round2")
    a = p.parse_args()
    inside(a) if a.inside else fetch(a) if a.fetch else launch(a)


if __name__ == "__main__":
    main()
