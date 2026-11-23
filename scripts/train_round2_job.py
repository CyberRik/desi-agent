"""[F] Stage 2, round-2 RL: one more rejection-sampling fine-tuning (RFT) round, in ONE Hugging Face Job.

Same method as rft-a (docs/ARTICLE.md, "Signal first, then RL"), one round further and on R's round-2 sets
(data/round2/, built from train/val only, never test):

  1. sample   vLLM serves the current policy (rft-a by default). For every langswitch_train prompt, K full
              conversations at temperature 1.0 through the shared fmt.run_episode loop + mock bank, each scored with
              rewards.reward_lang (the same rollout as scripts/eval_rl_signal.py).
  2. build    keep each prompt's best sample if it is fully correct (reward_lang = 1.6) and fits max_len, as
              scripts/data_rft_from_probe.py does. Training set = round-1 RFT set (data/rl/train_rft.jsonl: train +
              117 rft-a samples) + these new samples + grounding_train gold rows (missing id -> ask, control -> act).
  3. train    unchanged train_sft pipeline from the base model, rft-a's config, pushed to a NEW adapter repo.

Everything the job makes goes to ToolW/desi-agent-training-logs under round2/<run>/ (samples, summary, the exact
training file + manifest) and checkpoints/<run>/. Existing adapter repos are never touched.

    # launch (R launches paid jobs, docs/COMPUTE.md); prints the job URL
    python scripts/train_round2_job.py --run rft2 --flavor l40sx1 --timeout 4h
    # CPU dry run of steps 1-2 with a gold-replay policy (no GPU, no upload), checks the data it would train on
    python scripts/train_round2_job.py --dry-run --limit 40
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GITHUB = "CyberRik/desi-agent"
BASE = "CohereLabs/tiny-aya-global"
POLICY = "ToolW/desi-agent-tinyaya-rft-lora"
LOGS = "ToolW/desi-agent-training-logs"
IMAGE = "vllm/vllm-openai:v0.33.0"  # same image as R's eval_hf_job.py
PORT = 8001
SEED = 1234
FULL = 1.0 + 0.3 + 0.3  # reward_lang maximum: success + schema_valid + language bonus
MAX_TOKENS = 1950       # as data_rft_from_probe.py: rows longer than this would be truncated at max_len 2048
PRICE_PER_H = {"l4x1": 0.80, "a10g-large": 1.50, "l40sx1": 1.80, "a100-large": 2.50}
PROTECTED = {"ToolW/desi-agent-tinyaya-sft-lora", "ToolW/desi-agent-tinyaya-rft-lora"}  # linked from the forms


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def wjl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sh(cmd, check: bool = True, **kw) -> int:
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    rc = subprocess.run(cmd, shell=isinstance(cmd, str), **kw).returncode
    if check and rc:
        raise RuntimeError(f"exit {rc}")
    return rc


# ------------------------------------------------------------------------------------------------ 1. sample
def rollout(policy, row: dict) -> dict:
    from desi_agent import fmt
    from desi_agent.eval.language import final_reply, lang_match
    from desi_agent.eval.rewards import reward_lang
    run = fmt.run_episode(policy, [dict(m) for m in row["prefix"]] + [dict(row["turn2"])])
    turn2 = run.messages[len(row["prefix"]):]
    return {"messages": turn2, "reward": round(reward_lang(turn2, row["gold"], row["lang"]), 4),
            "lang_match": float(lang_match(final_reply(turn2), row["lang"]))}


def gold_replay(row: dict):
    """Dry-run policy: replays the row's own gold trajectory (R guarantees it scores 1.6), one assistant turn per call."""
    from desi_agent import fmt
    turns = [m for m in row["messages"][len(row["prefix"]):] if m["role"] == "assistant"]
    it = iter(turns)

    def policy(chat):
        m = next(it)
        return fmt.to_chat([m], with_system=False)[0]["content"]
    return policy


def sample(rows: list[dict], k: int, policy_for, workers: int) -> tuple[list[dict], dict]:
    jobs = [(r, j) for r in rows for j in range(k)]
    with ThreadPoolExecutor(workers) as ex:
        outs = list(ex.map(lambda x: rollout(policy_for(x[0]), x[0]), jobs))
    groups = collections.defaultdict(list)
    for (r, _), o in zip(jobs, outs):
        groups[r["id"]].append(o)
    res = [{"id": r["id"], "lang": r["lang"], "lang1": r["lang1"], "condition": r.get("condition"),
            "category": r["category"], "target": r["target"], "target_set": r.get("target_set"),
            "prefix": r["prefix"], "gold": r["gold"], "samples": groups[r["id"]]} for r in rows]
    rs = [[s["reward"] for s in r["samples"]] for r in res]
    by = collections.defaultdict(list)
    for r in res:
        by[f"{r['condition']}->{r['lang']}"].append(max(s["reward"] for s in r["samples"]) >= FULL - 1e-6)
    summary = {"n_prompts": len(res), "k": k, "mean_reward": round(statistics.mean(x for g in rs for x in g), 4),
               "frac_prompts_with_variance": round(sum(statistics.pstdev(g) > 1e-9 for g in rs) / len(rs), 4),
               "frac_prompts_with_full_sample": round(sum(max(g) >= FULL - 1e-6 for g in rs) / len(rs), 4),
               "lang_match_rate": round(statistics.mean(s["lang_match"] for r in res for s in r["samples"]), 4),
               "full_sample_by_condition_lang": {k_: f"{sum(v)}/{len(v)}" for k_, v in sorted(by.items())}}
    return res, summary


# ------------------------------------------------------------------------------------------------ 2. build
def build(samples: list[dict], tok, policy_name: str) -> tuple[list[dict], dict]:
    from desi_agent import fmt
    from desi_agent.data.splits import load_split

    def n_tokens(msgs):
        text = tok.apply_chat_template(fmt.to_chat(msgs), tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    test_ids = {e["id"] for e in load_split("test")}
    stats = collections.Counter()
    new = []
    for r in samples:
        assert r["target"] not in test_ids, "round-2 prompts must come from train/val only"
        best = max(r["samples"], key=lambda s: s["reward"])
        if best["reward"] < FULL - 1e-6:
            stats["langswitch_no_full_sample"] += 1
            continue
        msgs = [dict(m) for m in r["prefix"]] + best["messages"]
        if tok is not None and n_tokens(msgs) > MAX_TOKENS:
            stats["langswitch_too_long"] += 1
            continue
        calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                 for m in msgs for c in (m.get("tool_calls") or [])]
        cat = r["category"] if not calls else ("single_tool" if len(calls) == 1 else "multi_tool")
        new.append({"id": "rft2-" + r["id"].replace(":", "-").replace(">", "-"), "lang": r["lang"], "category": cat,
                    "split": "train", "source": f"rft2:{policy_name}@reward_lang", "messages": msgs,
                    "gold": {"tool_sequence": calls, "must_ask_clarification": cat == "clarify"},
                    "meta": {"lang1": r["lang1"], "condition": r["condition"], "reward": best["reward"],
                             "turn2_category": r["category"], "turn2_gold": r["gold"], "target_set": r["target_set"]}})
    round1 = jl(ROOT / "data/rl/train_rft.jsonl")
    grounding = jl(ROOT / "data/round2/grounding_train.jsonl")
    ids = collections.Counter(e["id"] for e in round1 + grounding + new)
    dup = [i for i, n in ids.items() if n > 1]
    assert not dup, f"duplicate ids {dup[:5]}"
    rows = round1 + grounding + new
    assert all(e.get("split") == "train" for e in rows), "only train rows"
    assert not (test_ids & set(ids)), "test id in training data"
    random.Random(SEED).shuffle(rows)
    stats = dict(stats) | {"round1_rft_set": len(round1), "grounding_gold": len(grounding), "langswitch_rft2_added": len(new),
              "total": len(rows), "langswitch_added_by_lang": dict(collections.Counter(e["lang"] for e in new))}
    return rows, stats


# ------------------------------------------------------------------------------------------------ inside the job
def serve(adapter_path: str, log: Path) -> subprocess.Popen:
    import urllib.request
    rank = json.loads(Path(adapter_path, "adapter_config.json").read_text())["r"]
    cmd = (f"vllm serve {BASE} --port {PORT} --dtype half --max-model-len 4096 --gpu-memory-utilization 0.90 "
           f"--enable-prefix-caching --enable-lora --max-lora-rank {max(16, rank)} --lora-modules policy={adapter_path}")
    p = subprocess.Popen(cmd, shell=True, stdout=log.open("w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health")
            print("[F] vLLM up", flush=True)
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                break
            time.sleep(10)
    sh(f"tail -80 {log}", check=False)
    raise RuntimeError("vLLM failed to start")


def stop_vllm(p: subprocess.Popen) -> None:
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


def inside(a) -> None:
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    from desi_agent.eval.runner import openai_policy
    api = HfApi()
    t0 = time.time()
    out = Path("/out") / a.run
    out.mkdir(parents=True, exist_ok=True)
    remote = f"round2/{a.run}"

    def upload(path: Path, name: str) -> None:
        for attempt in range(3):
            try:
                api.upload_file(path_or_fileobj=str(path), repo_id=LOGS, repo_type="dataset",
                                path_in_repo=f"{remote}/{name}", commit_message=f"[F] {a.run}: {name}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[F] upload failed ({e}), retry {attempt + 1}", flush=True)
                time.sleep(20)
        raise RuntimeError(f"upload of {name} failed")

    sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    if a.train_from:  # the data was already built by an earlier job: train on exactly that file
        train_split = f"hf:round2/{a.train_from}/train.jsonl"
        print(f"[F] reusing {LOGS}/{train_split[3:]}, no sampling", flush=True)
        return train(a, train_split, upload, t0)
    repo, _, rev = a.policy.partition("@")
    path = snapshot_download(repo, revision=rev or None)
    rows = jl(ROOT / "data/round2/langswitch_train.jsonl")[: a.limit or None]
    srv = serve(path, out / "vllm.log")
    pol = openai_policy(f"http://localhost:{PORT}/v1", "policy", temperature=a.temperature)
    t1 = time.time()
    samples, summary = sample(rows, a.k, lambda r: pol, a.workers)
    summary |= {"policy": a.policy, "temperature": a.temperature, "sample_minutes": round((time.time() - t1) / 60, 1)}
    print("[F] sample:", json.dumps(summary), flush=True)
    stop_vllm(srv)
    wjl(out / "samples.jsonl", samples)
    (out / "sample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    upload(out / "samples.jsonl", "samples.jsonl")
    upload(out / "sample_summary.json", "sample_summary.json")

    tok = AutoTokenizer.from_pretrained(BASE)
    train, stats = build(samples, tok, a.policy)
    stats |= {"git_sha": os.environ.get("DESI_GIT_SHA"), "seed": SEED, "max_tokens": MAX_TOKENS,
              "sources": {"round1": "data/rl/train_rft.jsonl", "grounding": "data/round2/grounding_train.jsonl",
                          "langswitch_prompts": "data/round2/langswitch_train.jsonl"}, "sample": summary}
    print("[F] build:", json.dumps({k: v for k, v in stats.items() if k != "sample"}), flush=True)
    wjl(out / "train.jsonl", train)
    (out / "train.manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    upload(out / "train.jsonl", "train.jsonl")
    upload(out / "train.manifest.json", "train.manifest.json")
    if a.no_train:
        return
    train(a, f"hf:{remote}/train.jsonl", upload, t0)


def train(a, train_split: str, upload, t0: float) -> None:
    # Training in its own Python 3.12 venv, exactly as the Kaggle runs (train_kaggle.py), so vLLM's pins don't leak in.
    # AgentTune installs from git and the vLLM image has no git binary
    sh("command -v git || (apt-get update -qq && apt-get install -y -qq git >/dev/null)")
    sh("python3 -m pip install -q --root-user-action=ignore uv")
    sh("env -u UV_SYSTEM_PYTHON -u UV_PYTHON uv venv -q -p 3.12 --python-preference managed /tmp/venv")
    sh("uv pip install --python /tmp/venv/bin/python -q --index-strategy unsafe-best-match "
       "--extra-index-url https://download.pytorch.org/whl/cu126 torch==2.11.0 -e '.[train]'", cwd=ROOT)
    env = dict(os.environ, DESI_HF_DATASET=LOGS, TOKENIZERS_PARALLELISM="false")
    cmd = ["/tmp/venv/bin/python", "scripts/train_sft.py", "--split", train_split,
           "--out", f"outputs/{a.run}", "--push-to", a.push_to, "--set", f"run_name={a.run}",
           "--set", f"ckpt_repo={LOGS}", "--set", f"max_hours={a.max_hours}"]
    t2 = time.time()
    sh(cmd, cwd=ROOT, env=env)
    for f in ("manifest.json", "log_history.json"):
        p = ROOT / "outputs" / a.run / f
        if p.exists():
            upload(p, f"train_{f}")
    print(f"[F] done: train {(time.time() - t2) / 60:.0f} min, "
          f"total {(time.time() - t0) / 60:.0f} min -> {a.push_to}", flush=True)


# ------------------------------------------------------------------------------------------------ launcher / dry run
def load_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for d in [ROOT, *ROOT.parents]:
        env = d / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    from huggingface_hub import get_token
    tok = get_token()
    if not tok:
        sys.exit("[F] no HF token (HF_TOKEN, .env or `hf auth login`)")
    return tok


def launch(a) -> None:
    from huggingface_hub import HfApi
    if a.push_to in PROTECTED:
        sys.exit(f"[F] {a.push_to} is linked from the submission forms; push to a new repo")
    token = load_token()
    api = HfApi(token=token)
    if api.repo_exists(a.push_to):
        sys.exit(f"[F] {a.push_to} already exists; pick a new --push-to (never overwrite an adapter)")
    files = api.list_repo_files(LOGS, repo_type="dataset")
    if a.train_from and f"round2/{a.train_from}/train.jsonl" not in files:
        sys.exit(f"[F] round2/{a.train_from}/train.jsonl not on {LOGS}")
    if not a.train_from and any(f.startswith(f"round2/{a.run}/") for f in files):
        sys.exit(f"[F] round2/{a.run}/ already exists on {LOGS}; pick a new --run")
    repo, _, rev = a.policy.partition("@")
    policy = f"{repo}@{api.model_info(repo, revision=rev or None).sha}"  # pin the exact sampling policy
    sha = subprocess.run(["git", "rev-parse", a.git_sha or "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://codeload.github.com/{GITHUB}/tar.gz/{sha}",
                                                      method="HEAD"), timeout=60)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[F] commit {sha[:12]} is not on github.com/{GITHUB} ({e}); push it first")
    inner = ["python3", "scripts/train_round2_job.py", "--inside", "--run", a.run, "--policy", policy,
             "--push-to", a.push_to, "--k", str(a.k), "--temperature", str(a.temperature), "--workers",
             str(a.workers), "--max-hours", str(a.max_hours)] + (["--limit", str(a.limit)] if a.limit else []) \
        + (["--no-train"] if a.no_train else []) + (["--train-from", a.train_from] if a.train_from else [])
    script = " && ".join([
        "set -e", "mkdir -p /w", "cd /w",
        f"python3 -c \"import urllib.request,tarfile,io; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen("
        f"'https://codeload.github.com/{GITHUB}/tar.gz/{sha}').read())).extractall('/w')\"",
        "cd /w/desi-agent-*",
        "python3 -m pip install -q --root-user-action=ignore -e '.[eval]'",
        " ".join(inner)])
    job = api.run_job(image=IMAGE, command=["bash", "-c", script], flavor=a.flavor, timeout=a.timeout,
                      secrets={"HF_TOKEN": token},
                      env={"DESI_GIT_SHA": sha, "PYTHONUNBUFFERED": "1", "PIP_BREAK_SYSTEM_PACKAGES": "1"},
                      labels={"project": "desi-agent", "run": a.run[:60]}, name=f"desi-train-{a.run}"[:60])
    print(f"[F] job {job.id} on {a.flavor} (${PRICE_PER_H.get(a.flavor)}/h), timeout {a.timeout}\n    {job.url}\n"
          f"    code {GITHUB}@{sha[:12]}  policy {policy}  -> {a.push_to}\n"
          f"    logs: hf jobs logs {job.id}", flush=True)


def dry_run(a) -> None:
    rows = jl(ROOT / "data/round2/langswitch_train.jsonl")
    rows = random.Random(SEED).sample(rows, a.limit) if a.limit else rows
    samples, summary = sample(rows, 2, gold_replay, 8)
    print("[F] dry-run sample:", json.dumps(summary))
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE)
    except Exception as e:  # noqa: BLE001
        print(f"[F] no tokenizer ({type(e).__name__}); skipping the length filter")
    train, stats = build(samples, tok, "gold-replay")
    print("[F] dry-run build:", json.dumps(stats))
    out = ROOT / "outputs/round2_dry/train.jsonl"
    wjl(out, train)
    sh([sys.executable, str(ROOT / "scripts/validate_data.py"), str(out)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="rft2", help="run name: round2/<run>/ and checkpoints/<run>/ on " + LOGS)
    p.add_argument("--policy", default=POLICY, help="adapter repo[@rev] to sample from (default: rft-a)")
    p.add_argument("--push-to", default="ToolW/desi-agent-tinyaya-rft2-lora", help="NEW adapter repo")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke runs)")
    p.add_argument("--max-hours", type=float, default=3.0, help="train_sft stops + checkpoints after this")
    p.add_argument("--no-train", action="store_true", help="stop after sampling + building the data")
    p.add_argument("--train-from", help="skip sampling: train on round2/<run>/train.jsonl from an earlier job")
    p.add_argument("--flavor", default="l40sx1")
    p.add_argument("--timeout", default="4h")
    p.add_argument("--git-sha")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.inside:
        inside(a)
    elif a.dry_run:
        dry_run(a)
    else:
        launch(a)


if __name__ == "__main__":
    main()
