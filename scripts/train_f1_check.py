"""[F] F1 env check: inspect the chat template and run seed episodes through fmt.run_episode.

    python scripts/train_f1_check.py --out outputs/f1_check                       # Tiny Aya (gated, needs HF_TOKEN)
    python scripts/train_f1_check.py --model HuggingFaceTB/SmolLM2-135M-Instruct --limit 2 --out outputs/f1_smoke

Writes <out>/f1_check.json (template findings, per-episode raw outputs + scores) and <out>/manifest.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desi_agent import fmt  # noqa: E402
from desi_agent.api import load_policy, load_split, score_episode  # noqa: E402
from desi_agent.train.chat import apply_template, supports_system  # noqa: E402

SEED = 1234


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="seed")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    torch.manual_seed(SEED)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pol = load_policy(a.model, a.adapter, load_in_4bit=a.load_in_4bit, max_new_tokens=a.max_new_tokens)
    load_s = time.time() - t0
    eps = load_split(a.split)[: a.limit] if a.limit else load_split(a.split)

    sample_chat = fmt.to_chat(eps[0]["messages"])
    template = {
        "supports_system_role": supports_system(pol.tok),
        "has_chat_template": bool(getattr(pol.tok, "chat_template", None)),
        "bos_token": pol.tok.bos_token, "eos_token": pol.tok.eos_token, "pad_token": pol.tok.pad_token,
        "rendered_example": apply_template(pol.tok, sample_chat, add_generation_prompt=False),
        "system_prompt_tokens": len(pol.tok(fmt.system_prompt(), add_special_tokens=False)["input_ids"]),
    }
    print(json.dumps({k: v for k, v in template.items() if k != "rendered_example"}, indent=2))
    print("---- rendered example ----\n" + template["rendered_example"][-1500:])

    rows = []
    for ep in eps:
        t = time.time()
        run = fmt.run_episode(pol, [ep["messages"][0]])
        s = score_episode(run.messages, ep["gold"])
        rows.append({"id": ep["id"], "lang": ep["lang"], "category": ep["category"],
                     "raw_outputs": run.raw_outputs, "truncated": run.truncated, "scores": s,
                     "latency_s": round(time.time() - t, 2)})
        print(f"[{ep['id']}] success={s.get('task_success')} schema={s.get('schema_valid')} "
              f"first_out={run.raw_outputs[0][:120]!r}")

    n = len(rows)
    summary = {k: round(sum(r["scores"].get(k, 0) for r in rows) / n, 3) for k in ("task_success", "schema_valid")}
    print("summary:", summary)
    (out / "f1_check.json").write_text(json.dumps(
        {"template": template, "summary": summary, "load_s": round(load_s, 1), "episodes": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "manifest.json").write_text(json.dumps(
        {"script": "scripts/train_f1_check.py", "git_sha": sha, "model_id": a.model, "adapter": a.adapter,
         "split": a.split, "limit": a.limit, "seed": SEED, "device": str(pol.model.device)}, indent=2))


if __name__ == "__main__":
    main()
