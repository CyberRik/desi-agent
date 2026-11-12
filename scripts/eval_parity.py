"""[R] R6 parity check: HF transformers policy (train.policy.load_policy) vs a vLLM server, same episodes.

Both go through fmt.run_episode + score_episode, greedy, stop at </tool_call>. fp16 kernels differ, so raw text may
differ in places; what matters is that the scores agree. Writes a JSON report.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_parity.py --port 8001 --n 20 --out parity.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desi_agent import fmt  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402
from desi_agent.eval.metrics import score_episode  # noqa: E402
from desi_agent.eval.runner import openai_policy  # noqa: E402

KEYS = ("task_success", "tool_selection_correct", "args_exact", "schema_valid", "false_action")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    test = load_split("test")
    step = max(1, len(test) // a.n)
    eps = test[::step][: a.n]  # spread over languages/categories (test is sorted by id)

    from desi_agent.train.policy import load_policy
    hf = load_policy(a.model, device_map={"": 0})
    vl = openai_policy(f"http://localhost:{a.port}/v1", a.model)

    rows, same_text, same_scores = [], 0, 0
    for ep in eps:
        r_hf = fmt.run_episode(hf, [ep["messages"][0]])
        r_vl = fmt.run_episode(vl, [ep["messages"][0]])
        s_hf, s_vl = score_episode(r_hf.messages, ep["gold"]), score_episode(r_vl.messages, ep["gold"])
        t_eq = r_hf.raw_outputs == r_vl.raw_outputs
        s_eq = all(s_hf[k] == s_vl[k] for k in KEYS)
        same_text += t_eq
        same_scores += s_eq
        rows.append({"id": ep["id"], "same_text": t_eq, "same_scores": s_eq,
                     "hf": {k: s_hf[k] for k in KEYS}, "vllm": {k: s_vl[k] for k in KEYS},
                     "hf_raw": r_hf.raw_outputs, "vllm_raw": r_vl.raw_outputs})
        print(f"[R] parity {ep['id']}: text={'=' if t_eq else '!='} scores={'=' if s_eq else '!='}", flush=True)
    rep = {"n": len(eps), "same_text": same_text, "same_scores": same_scores,
           "hf_task_success": sum(r["hf"]["task_success"] for r in rows) / len(rows),
           "vllm_task_success": sum(r["vllm"]["task_success"] for r in rows) / len(rows), "episodes": rows}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[R] PARITY", json.dumps({k: rep[k] for k in ("n", "same_text", "same_scores", "hf_task_success",
                                                         "vllm_task_success")}), flush=True)


if __name__ == "__main__":
    main()
