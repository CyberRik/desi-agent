"""[R] R6 on Kaggle (GPU T4 x2, Internet ON): baseline eval of Tiny Aya Global + teacher upper bound on the test split.

Runs as a Kaggle *script* kernel (see scripts/eval_kaggle_push.sh). Needs an HF token that can read the gated
CohereLabs/tiny-aya-global: Kaggle secret HF_TOKEN, or a file `hf_token` in an attached PRIVATE Kaggle dataset.

Phases (results land in /kaggle/working/out/results/<model_tag>/):
  1. parity: 20 test episodes through the HF transformers policy (train.policy.load_policy, GPU 0) and through
     vLLM (GPU 1); report exact-output agreement and score agreement. vLLM is only used if the scores agree.
  2. tinyaya-global-base: full test split through vLLM (prefix caching makes the 1.3k-token shared prompt cheap).
  3. teacher-qwen3-14b: Qwen/Qwen3-14B-AWQ (thinking off) on both GPUs, same agent loop = upper bound.
Env: PHASES=parity,base,teacher (default all), BRANCH (default main), ADAPTER_REPO/ADAPTER_TAG for an SFT adapter.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

BRANCH = os.environ.get("BRANCH", "main")
PHASES = os.environ.get("PHASES", "parity,base,teacher").split(",")
BASE = "CohereLabs/tiny-aya-global"
TEACHER = "Qwen/Qwen3-14B-AWQ"
WORK = "/kaggle/working/desi-agent"
OUT = "/kaggle/working/out"


def sh(c, env=None, check=True):
    print("$", c, flush=True)
    return subprocess.run(c, shell=True, check=check, env=env)


def token() -> str:
    t = None
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001
        pass
    if not t:
        files = glob.glob("/kaggle/input/**/hf_token", recursive=True)
        t = open(files[0]).read().strip() if files else None
    if not t:
        sys.exit("[R] no HF token (secret HF_TOKEN or private dataset with hf_token)")
    return t


def serve(model, port, gpus, tp, extra=""):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    log = open(f"{OUT}/vllm_{port}.log", "w")
    p = subprocess.Popen(f"vllm serve {model} --port {port} --tensor-parallel-size {tp} --dtype half "
                         f"--max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching {extra}",
                         shell=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(90):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health")
            print(f"[R] vLLM {model} up on :{port}", flush=True)
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                break
            time.sleep(10)
    sh(f"tail -60 {OUT}/vllm_{port}.log", check=False)
    raise RuntimeError(f"vLLM {model} failed to start")


def stop(p):
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(10)


def run_eval(tag, model, port, extra_args="", limit=None):
    lim = f"--limit {limit}" if limit else ""
    sh(f"python -m desi_agent.eval.runner --split test --model-tag {tag} --openai-base-url http://localhost:{port}/v1 "
       f"--openai-model {model} --workers 32 --out-dir {OUT}/results/{tag} {lim} {extra_args}")


def main():
    os.makedirs(f"{OUT}/results", exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh(f"git clone -q -b {BRANCH} https://github.com/CyberRik/desi-agent {WORK}")
    os.chdir(WORK)
    sh('pip install -q "vllm>=0.8" openai auditkit peft >/dev/null')
    sh("pip install -q --ignore-requires-python -e . >/dev/null")
    sh('python -c "import vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__)"')
    os.environ["HF_TOKEN"] = token()

    if "parity" in PHASES or "base" in PHASES:
        srv = serve(BASE, 8001, "1", 1)
        if "parity" in PHASES:
            sh(f"CUDA_VISIBLE_DEVICES=0 python scripts/eval_parity.py --port 8001 --n 20 --out {OUT}/results/parity.json")
        if "base" in PHASES:
            run_eval("tinyaya-global-base", BASE, 8001, f"--model-id {BASE}")
        stop(srv)

    if os.environ.get("ADAPTER_REPO"):
        tag = os.environ.get("ADAPTER_TAG", "tinyaya-global-sft")
        from huggingface_hub import snapshot_download
        repo, _, sub = os.environ["ADAPTER_REPO"].partition(":")  # "repo" or "repo:path/in/repo" (dataset repos)
        kind = "dataset" if sub else "model"
        local = snapshot_download(repo, repo_type=kind, allow_patterns=[f"{sub}/*"] if sub else None)
        path = os.path.join(local, sub) if sub else local
        srv = serve(BASE, 8001, "1", 1, f"--enable-lora --max-lora-rank 64 --lora-modules sft={path}")
        run_eval(tag, "sft", 8001, f"--model-id {BASE} --adapter {os.environ['ADAPTER_REPO']}")
        stop(srv)

    if "teacher" in PHASES:
        srv = serve(TEACHER, 8002, "0,1", 2)
        run_eval("teacher-qwen3-14b", TEACHER, 8002, "--no-thinking")
        stop(srv)

    for f in glob.glob(f"{OUT}/results/**/*.json", recursive=True):
        if f.endswith("test.json"):
            r = json.load(open(f))
            print(f"[R] {r['model_tag']}: " + json.dumps({k: r['overall'].get(k) for k in
                  ('task_success', 'tool_selection_correct', 'args_exact', 'schema_valid', 'false_action')}), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
