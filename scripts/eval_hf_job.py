"""[R] Score one LoRA adapter on the full eval suite in ONE Hugging Face Job, then pull the results back.

Same settings as the Kaggle runs behind results/ (so numbers are comparable): vLLM `--dtype half` (fp16), greedy
(temperature 0), max_new_tokens 256, stop at `</tool_call>`, max 6 steps, 32 client threads, the shared
`fmt.run_episode` loop, `--max-model-len 4096 --enable-prefix-caching`, LoRA served as `sft`. BANKING77 runs on HF
transformers in fp16, unbatched, as `scripts/eval_real_queries.py` did on the T4s (load_policy would pick bf16 on an
L4/A10G/L40S, so fp16 is forced here).

    # launch (prints the job URL; --wait follows it and fetches when done)
    python scripts/eval_hf_job.py --adapter ToolW/desi-agent-tinyaya-sft-lora@main --tag s2-sft-a-check \
        --suites test,ood,perturbed,multiturn,banking77 --flavor l4x1 --timeout 3h --wait
    # later / from another machine
    python scripts/eval_hf_job.py --fetch --tag s2-sft-a-check
    python scripts/eval_compare.py tinyaya-sft-a-final tinyaya-rft-a s2-sft-a-check

Suites -> local dirs (same layout as the existing ones):
    test       results/<tag>/test.json (+ test.traces.jsonl, test.auditkit.json)
    ood        results/<tag>-ood/test.json (+ validated.json: the 380 judge-validated rewrites, as in robustness.json)
    perturbed  results/<tag>-perturbed/test.json
    multiturn  results/<tag>-multiturn/multiturn.json
    banking77  results/<tag>-banking77/manifest.json (same shape as results/banking77.json, one tag)
plus results/<tag>/hfjob_manifest.json (git sha, image, package versions, base + adapter revisions, seed, timings).

Remote: HF dataset ToolW/desi-agent-eval-traces under stage2/<tag>/<local dir name>/... Existing files are never
overwritten: the launcher refuses a tag that already exists, and the job falls back to stage2/<tag>-<job id>/.
The code that runs is the GitHub commit given by --git-sha (default: local HEAD, which must be pushed).
HF_TOKEN comes from the environment or the repo's .env and is passed as a job secret (never printed).
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
GITHUB = "CyberRik/desi-agent"
DATASET = "ToolW/desi-agent-eval-traces"
BASE = "CohereLabs/tiny-aya-global"
IMAGE = "vllm/vllm-openai:v0.33.0"  # vLLM release current when the Kaggle runs did `pip install "vllm>=0.8"`
SEED = 1234
SUITES = ("test", "ood", "perturbed", "multiturn", "banking77")
SUFFIX = {"test": "", "ood": "-ood", "perturbed": "-perturbed", "multiturn": "-multiturn", "banking77": "-banking77"}
PRICE_PER_H = {"l4x1": 0.80, "a10g-large": 1.50, "a10g-small": 1.00, "l40sx1": 1.80, "t4-small": 0.40,
               "t4-medium": 0.60, "a100-large": 2.50}
PORT = 8001


# ----------------------------------------------------------------------------------------------- shared helpers
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


def parse_adapter(spec: str) -> tuple[str | None, str | None]:
    if spec.lower() in ("none", "base"):
        return None, None
    repo, _, rev = spec.partition("@")
    return repo, rev or None


def dirname(tag: str, suite: str) -> str:
    return tag + SUFFIX[suite]


# ----------------------------------------------------------------------------------------------- launcher (local)
def git_sha_on_github(sha: str | None) -> str:
    sha = sha or subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://api.github.com/repos/{GITHUB}/commits/{sha}", timeout=30) as r:
            full = json.load(r)["sha"]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[R] commit {sha} not found on github.com/{GITHUB} ({e}). Push it first.")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        print("[R] warning: local changes are NOT in the job (it runs the pushed commit)", flush=True)
    return full


def launch(a) -> None:
    from huggingface_hub import HfApi
    token = load_token()
    api = HfApi(token=token)
    repo, rev = parse_adapter(a.adapter)
    adapter_sha = api.model_info(repo, revision=rev).sha if repo else None  # pin the exact adapter commit
    existing = [f for f in api.list_repo_files(DATASET, repo_type="dataset") if f.startswith(f"stage2/{a.tag}/")]
    if existing:
        sys.exit(f"[R] stage2/{a.tag}/ already exists on {DATASET} ({len(existing)} files). Pick a new --tag.")
    sha = git_sha_on_github(a.git_sha)
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    bad = [s for s in suites if s not in SUITES]
    if bad:
        sys.exit(f"[R] unknown suites {bad}; choose from {SUITES}")
    suites = [s for s in SUITES if s in suites]  # test first: validated OOD pairs with the test traces
    inner = ["python3", "scripts/eval_hf_job.py", "--inside", "--tag", a.tag, "--suites", ",".join(suites),
             "--adapter", f"{repo}@{adapter_sha}" if repo else "none", "--flavor", a.flavor,
             "--multiturn-n", str(a.multiturn_n), "--per-intent", str(a.per_intent)]
    script = " && ".join([
        "set -e", "mkdir -p /w", "cd /w",
        f"python3 -c \"import urllib.request,tarfile,io; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen("
        f"'https://codeload.github.com/{GITHUB}/tar.gz/{sha}').read())).extractall('/w')\"",
        "cd /w/desi-agent-*",
        "python3 -m pip install -q --root-user-action=ignore auditkit peft accelerate",
        "python3 -m pip install -q --root-user-action=ignore -e .",
        " ".join(inner)])
    job = api.run_job(image=IMAGE, command=["bash", "-c", script], flavor=a.flavor, timeout=a.timeout,
                      secrets={"HF_TOKEN": token}, env={"DESI_GIT_SHA": sha, "PYTHONUNBUFFERED": "1", "PIP_BREAK_SYSTEM_PACKAGES": "1"},
                      labels={"project": "desi-agent", "tag": a.tag[:60]}, name=f"desi-eval-{a.tag}"[:60])
    est = PRICE_PER_H.get(a.flavor)
    print(f"[R] job {job.id} on {a.flavor} (${est}/h), timeout {a.timeout}\n    {job.url}\n"
          f"    code {GITHUB}@{sha[:12]}  adapter {repo}@{(adapter_sha or '-')[:12]}  suites {','.join(suites)}\n"
          f"    fetch: python scripts/eval_hf_job.py --fetch --tag {a.tag} --job {job.id}", flush=True)
    if a.wait:
        wait(api, job.id)
        fetch(a, job_id=job.id)


def wait(api, job_id: str) -> dict:
    last = None
    while True:
        j = api.inspect_job(job_id=job_id)
        if j.status.stage != last:
            print(f"[R] {time.strftime('%H:%M:%S')} job {job_id}: {j.status.stage} {j.status.message or ''}", flush=True)
            last = j.status.stage
        if j.status.stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            break
        time.sleep(30)
    return job_cost(j)


def job_cost(j) -> dict:
    flavor = str(getattr(j.flavor, "value", j.flavor))
    info = {"job_id": j.id, "stage": j.status.stage, "flavor": flavor}
    if j.started_at and j.finished_at:
        h = (j.finished_at - j.started_at).total_seconds() / 3600
        info |= {"runtime_min": round(h * 60, 1), "est_cost_usd": round(h * PRICE_PER_H.get(flavor, 0), 2)}
    print(f"[R] {info}", flush=True)
    return info


def fetch(a, job_id: str | None = None) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=load_token())
    prefix = f"stage2/{a.tag}/"
    files = [f for f in api.list_repo_files(DATASET, repo_type="dataset") if f.startswith(prefix)]
    if not files:
        sys.exit(f"[R] nothing under {prefix} on {DATASET} yet")
    out = Path(a.out) if a.out else ROOT / "results" / a.tag
    for f in files:
        rel = f[len(prefix):]
        if "/" in rel:  # "<tag>-ood/test.json" -> <out>-ood/test.json
            top, _, rest = rel.partition("/")
            dest = out.parent / (out.name + top[len(a.tag):]) / rest
        else:  # job-level files (manifest.json, vllm.log) -> <out>/hfjob_*
            dest = out / f"hfjob_{rel}"
        if dest.exists() and not a.force:
            print(f"[R] keep existing {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(hf_hub_download(DATASET, f, repo_type="dataset", token=api.token)).read_bytes())
        print(f"[R] {f} -> {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest}")
    if job_id:
        man = out / "hfjob_manifest.json"
        if man.exists():
            m = json.loads(man.read_text(encoding="utf-8"))
            m["job"] = job_cost(api.inspect_job(job_id=job_id))
            man.write_text(json.dumps(m, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- inside the job
def sh(cmd: str, log: Path | None = None, check: bool = True) -> int:
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fh = log.open("a", encoding="utf-8") if log else None
    for line in p.stdout:
        print(line, end="", flush=True)
        if fh:
            fh.write(line)
    rc = p.wait()
    if fh:
        fh.close()
    if check and rc:
        raise RuntimeError(f"exit {rc}: {cmd}")
    return rc


def serve(model: str, out: Path, extra: str = "") -> subprocess.Popen:
    import urllib.request
    log = (out / "vllm.log").open("w")
    p = subprocess.Popen(f"vllm serve {model} --port {PORT} --tensor-parallel-size 1 --dtype half "
                         f"--max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching "
                         f"{extra}", shell=True, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health")
            print("[R] vLLM up", flush=True)
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                break
            time.sleep(10)
    sh(f"tail -80 {out}/vllm.log", check=False)
    raise RuntimeError("vLLM failed to start")


def stop(p: subprocess.Popen) -> None:
    p.terminate()
    sh("pkill -f 'vllm serve' || true; pkill -f EngineCore || true", check=False)
    for _ in range(30):
        used = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                              capture_output=True, text=True).stdout.split()
        if all(int(u) < 1000 for u in used):
            return
        time.sleep(5)
    sh("pkill -9 -f vllm || true", check=False)
    time.sleep(10)


def ood_validated(ood_traces: Path, test_traces: Path | None = None) -> dict:
    """Same numbers as scripts/eval_ood_summary.py: only the judge-validated rewrites (data/test_ood.valid.json)."""
    import collections
    valid = set(json.loads((ROOT / "data/test_ood.valid.json").read_text(encoding="utf-8"))["valid_ids"])

    def traces(p):
        return {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

    def agg(rows):
        n = len(rows)
        return {"n": n, "task_success": round(sum(r["scores"]["task_success"] for r in rows) / max(1, n), 4),
                "false_action": round(sum(r["scores"]["false_action"] for r in rows) / max(1, n), 4)}
    ood = traces(ood_traces)
    ids = sorted(i for i in ood if i in valid)
    res = {"valid_ids": "data/test_ood.valid.json", "ood": agg([ood[i] for i in ids])}
    if test_traces and test_traces.exists():
        ind = traces(test_traces)
        res["in_distribution_same_episodes"] = agg([ind[i[:-4]] for i in ids if i[:-4] in ind])
    for key in ("lang", "category"):
        g = collections.defaultdict(list)
        for i in ids:
            g[ood[i][key]].append(ood[i])
        res[f"ood_by_{key}"] = {k: agg(v) for k, v in sorted(g.items())}
    return res


def banking77(tag: str, adapter_path: str | None, adapter_name: str | None, out: Path, per_intent: int) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_real_queries", ROOT / "scripts/eval_real_queries.py")
    rq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rq)
    from desi_agent import fmt
    from desi_agent.train.policy import load_policy
    eps = rq.sample(per_intent)
    pol = load_policy(adapter=adapter_path, dtype="float16")  # T4 reference ran fp16; auto would pick bf16 here
    rows, t0 = [], time.time()
    for i, e in enumerate(eps):
        run = fmt.run_episode(pol, [{"role": "user", "content": e["text"]}])
        rows.append({**e, "messages": run.messages, "raw_outputs": run.raw_outputs,
                     "scores": rq.score(e["text"], e["group"], run.messages, run.truncated)})
        if i % 25 == 0:
            print(f"[R] banking77 {i + 1}/{len(eps)} {time.time() - t0:.0f}s", flush=True)
    (out / f"{tag}.traces.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                             encoding="utf-8")
    summary = {"dataset": "mteb/banking77 (PolyAI BANKING77 test split, English, human-written)", "seed": rq.SEED,
               "per_intent": per_intent, "adapter": adapter_name, "dtype": "float16", "backend": "transformers",
               "intents": {"act_no_details": rq.ACT_NO_DETAILS, "out_of_scope": rq.OUT_OF_SCOPE},
               tag: rq.summarize(rows)}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[R] banking77:", json.dumps({k: v for k, v in summary[tag].items() if k != "by_intent"}), flush=True)


def inside(a) -> None:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    t_start = time.time()
    suites = a.suites.split(",")
    work = Path("/out")
    work.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("JOB_ID") or os.environ.get("HF_JOB_ID") or str(int(t_start))
    remote = f"stage2/{a.tag}"
    if any(f.startswith(remote + "/") for f in api.list_repo_files(DATASET, repo_type="dataset")):
        remote = f"stage2/{a.tag}-{job_id}"  # never overwrite
        print(f"[R] stage2/{a.tag}/ exists, uploading to {remote}/ instead", flush=True)
    sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv", check=False)
    versions = subprocess.run([sys.executable, "-c", "import vllm, torch, transformers, peft, auditkit, sys; "
                               "print(vllm.__version__, torch.__version__, transformers.__version__, peft.__version__, "
                               "getattr(auditkit, '__version__', '?'), sys.version.split()[0])"],
                              capture_output=True, text=True).stdout.split()
    print("[R] vllm torch transformers peft auditkit python:", versions, flush=True)
    repo, rev = parse_adapter(a.adapter)
    base_sha = api.model_info(BASE).sha
    snapshot_download(BASE, revision=base_sha)
    path = snapshot_download(repo, revision=rev) if repo else None
    adapter_name = repo  # recorded the same way as the Kaggle runs; the exact revision is in hfjob_manifest.json
    rank = json.loads(Path(path, "adapter_config.json").read_text())["r"] if path else 0
    manifest = {"tag": a.tag, "git_sha": os.environ.get("DESI_GIT_SHA"), "image": IMAGE, "flavor": a.flavor,
                "seed": SEED, "base_model": BASE, "base_revision": base_sha, "adapter": repo, "adapter_revision": rev,
                "lora_rank": rank, "versions": dict(zip(["vllm", "torch", "transformers", "peft", "auditkit", "python"],
                                                        versions)),
                "decoding": {"temperature": 0.0, "max_new_tokens": 256, "stop": ["</tool_call>"], "dtype": "float16",
                             "max_steps": 6, "workers": 32}, "remote": f"{DATASET}/{remote}", "suites": {}}

    def upload(local: Path, path_in_repo: str, msg: str) -> None:
        for attempt in range(3):
            try:
                if local.is_dir():
                    api.upload_folder(folder_path=str(local), repo_id=DATASET, repo_type="dataset",
                                      path_in_repo=path_in_repo, commit_message=msg, ignore_patterns=["vllm.log"])
                else:
                    api.upload_file(path_or_fileobj=str(local), repo_id=DATASET, repo_type="dataset",
                                    path_in_repo=path_in_repo, commit_message=msg)
                return
            except Exception as e:  # noqa: BLE001
                print(f"[R] upload failed ({e}), retry {attempt + 1}", flush=True)
                time.sleep(20)

    model = "sft" if repo else BASE
    common = f"--openai-base-url http://localhost:{PORT}/v1 --openai-model {model}"
    ident = f"--model-id {BASE}" + (f" --adapter {adapter_name}" if repo else "")
    cmds = {
        "test": f"python3 -m desi_agent.eval.runner --split test --model-tag {a.tag} {common} {ident} --workers 32",
        "ood": f"python3 scripts/eval_variants.py --tag {dirname(a.tag, 'ood')} {common} {ident} "
               f"--episodes data/test_ood.jsonl",
        "perturbed": f"python3 scripts/eval_variants.py --tag {dirname(a.tag, 'perturbed')} {common} {ident} --perturb",
        "multiturn": f"python3 scripts/eval_multiturn.py --tag {dirname(a.tag, 'multiturn')} {common} --n {a.multiturn_n}",
    }
    vsuites = [s for s in suites if s in cmds]
    failed = []
    if vsuites:
        lora_rank = max(64, 1 << (max(rank, 1) - 1).bit_length())
        extra = f"--enable-lora --max-lora-rank {lora_rank} --lora-modules sft={path}" if repo else ""
        srv = serve(BASE, work, extra)
        try:
            for s in vsuites:
                d = work / dirname(a.tag, s)
                d.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                try:
                    sh(f"{cmds[s]} --out-dir {d}", log=d / "run.log")
                    if s == "ood":
                        v = ood_validated(d / "test.traces.jsonl", work / a.tag / "test.traces.jsonl")
                        (d / "validated.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
                    manifest["suites"][s] = {"status": "ok", "minutes": round((time.time() - t0) / 60, 1)}
                except Exception as e:  # noqa: BLE001
                    failed.append(s)
                    manifest["suites"][s] = {"status": f"failed: {e}", "minutes": round((time.time() - t0) / 60, 1)}
                upload(d, f"{remote}/{d.name}", f"[R] stage2 eval {a.tag}: {s}")
        finally:
            sh(f"cp {work}/vllm.log {work}/{a.tag}-vllm.log || true", check=False)
            stop(srv)
    if "banking77" in suites:
        d = work / dirname(a.tag, "banking77")
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            banking77(a.tag, path, adapter_name, d, a.per_intent)
            manifest["suites"]["banking77"] = {"status": "ok", "minutes": round((time.time() - t0) / 60, 1)}
        except Exception as e:  # noqa: BLE001
            failed.append("banking77")
            manifest["suites"]["banking77"] = {"status": f"failed: {e}", "minutes": round((time.time() - t0) / 60, 1)}
        upload(d, f"{remote}/{d.name}", f"[R] stage2 eval {a.tag}: banking77")
    manifest["total_minutes"] = round((time.time() - t_start) / 60, 1)
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    upload(work / "manifest.json", f"{remote}/manifest.json", f"[R] stage2 eval {a.tag}: manifest")
    if (work / f"{a.tag}-vllm.log").exists():
        upload(work / f"{a.tag}-vllm.log", f"{remote}/vllm.log", f"[R] stage2 eval {a.tag}: vllm log")
    print("[R] done:", json.dumps(manifest["suites"]), flush=True)
    for s in suites:
        f = work / dirname(a.tag, s) / ("multiturn.json" if s == "multiturn" else
                                        "manifest.json" if s == "banking77" else "test.json")
        if f.exists() and s in ("test", "ood", "perturbed"):
            o = json.loads(f.read_text(encoding="utf-8"))["overall"]
            print(f"[R] {s}: task_success={o['task_success']} false_action={o['false_action']} n={o['n']}", flush=True)
    if failed:
        sys.exit(f"[R] failed suites: {failed}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="none", help="HF model repo[@revision] of a PEFT LoRA, or 'none' for the base")
    p.add_argument("--tag", required=True, help="results tag, e.g. tinyaya-s2-sft-b")
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--flavor", default="l4x1")
    p.add_argument("--timeout", default="3h")
    p.add_argument("--out", help="local dir for the test suite (default results/<tag>); siblings get -ood etc.")
    p.add_argument("--git-sha", help="GitHub commit to run (default: local HEAD, must be pushed)")
    p.add_argument("--multiturn-n", type=int, default=300)
    p.add_argument("--per-intent", type=int, default=10)
    p.add_argument("--wait", action="store_true", help="follow the job, then fetch")
    p.add_argument("--fetch", action="store_true", help="download stage2/<tag>/ into results/")
    p.add_argument("--job", help="job id (with --fetch: also record runtime + cost in hfjob_manifest.json)")
    p.add_argument("--force", action="store_true", help="--fetch: overwrite local files")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.inside:
        inside(a)
    elif a.fetch:
        if a.job:
            from huggingface_hub import HfApi
            wait(HfApi(token=load_token()), a.job)
        fetch(a, job_id=a.job)
    else:
        launch(a)


if __name__ == "__main__":
    main()
