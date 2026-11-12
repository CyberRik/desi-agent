"""[R] Kaggle T4x2 driver for the robustness evals (OOD rewording + perturbed bank).

1. Serve the rewriter (default CohereLabs/aya-expanse-8b, a different family from the Qwen teacher), build
   data/test_ood.jsonl with scripts/data_ood_reword.py.
2. Serve tiny-aya-global with our LoRA as 'sft' on the same vLLM server, then evaluate base and fine-tuned on
   (a) the OOD set and (b) the frozen test with a perturbed bank (scripts/eval_variants.py).
Outputs go to /kaggle/working/out/{data,results}. HF token from a private Kaggle dataset file `hf_token` or the
HF_TOKEN secret (Tiny Aya and Aya Expanse are gated).
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
REWRITER = os.environ.get("REWRITER", "CohereLabs/aya-expanse-8b")
BASE = "CohereLabs/tiny-aya-global"
ADAPTER = os.environ.get("ADAPTER_REPO", "ToolW/desi-agent-tinyaya-sft-lora")
WORK, OUT = "/kaggle/working/desi-agent", "/kaggle/working/out"


def sh(c, check=True):
    print("$", c, flush=True)
    return subprocess.run(c, shell=True, check=check)


def token():
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t:
            return t
    except Exception:  # noqa: BLE001
        pass
    files = glob.glob("/kaggle/input/**/hf_token", recursive=True)
    if not files:
        sys.exit("[R] no HF token")
    return open(files[0]).read().strip()


def serve(model, port, extra=""):
    log = open(f"{OUT}/vllm_{port}.log", "w")
    p = subprocess.Popen(f"vllm serve {model} --port {port} --tensor-parallel-size 2 --dtype half --max-model-len 4096 "
                         f"--gpu-memory-utilization 0.90 --enable-prefix-caching {extra}",
                         shell=True, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(90):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health")
            print(f"[R] vLLM {model} up", flush=True)
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                break
            time.sleep(10)
    sh(f"tail -60 {OUT}/vllm_{port}.log", check=False)
    raise RuntimeError(f"vLLM {model} failed to start")


def stop(p):
    p.terminate()
    sh("pkill -f 'vllm serve' || true; pkill -f 'EngineCore' || true", check=False)
    for _ in range(30):
        used = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                              capture_output=True, text=True).stdout.split()
        if all(int(u) < 1000 for u in used):
            return
        time.sleep(5)
    sh("pkill -9 -f vllm || true", check=False)
    time.sleep(10)


def main():
    os.makedirs(f"{OUT}/data", exist_ok=True)
    sh(f"git clone -q -b {BRANCH} https://github.com/CyberRik/desi-agent {WORK}")
    os.chdir(WORK)
    sh('pip install -q "vllm>=0.8" openai auditkit >/dev/null')
    sh("pip install -q --ignore-requires-python -e . >/dev/null")
    os.environ["HF_TOKEN"] = token()

    srv = serve(REWRITER, 8000)
    sh(f"python scripts/data_ood_reword.py --base-url http://localhost:8000/v1 --model {REWRITER} "
       f"--inp data/test.jsonl --out {OUT}/data/test_ood.jsonl")
    stop(srv)

    from huggingface_hub import snapshot_download
    path = snapshot_download(ADAPTER)
    srv = serve(BASE, 8001, f"--enable-lora --max-lora-rank 64 --lora-modules sft={path}")
    common = f"--openai-base-url http://localhost:8001/v1 --model-id {BASE}"
    for tag, model, extra in [("tinyaya-global-base-ood", BASE, f"--episodes {OUT}/data/test_ood.jsonl"),
                              ("tinyaya-sft-a-final-ood", "sft", f"--episodes {OUT}/data/test_ood.jsonl --adapter {ADAPTER}"),
                              ("tinyaya-global-base-perturbed", BASE, "--perturb"),
                              ("tinyaya-sft-a-final-perturbed", "sft", f"--perturb --adapter {ADAPTER}")]:
        sh(f"python scripts/eval_variants.py --tag {tag} --openai-model {model} {common} {extra} --out-dir {OUT}/results/{tag}")
    stop(srv)
    for f in sorted(glob.glob(f"{OUT}/results/*/test.json")):
        r = json.load(open(f))
        print("[R]", r["model_tag"], json.dumps({k: r["overall"].get(k) for k in ("task_success", "false_action", "stale_value_rate")}))


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
