# [R] GRPO audit: paired McNemar rft2 vs grpo-mt on multi-turn, train/eval overlap, rollout hacking checks.
# Usage: python scripts/eval_grpo_audit.py <dir with stage2/... eval traces and round2/grpo-mt/ logs from HF>
import json, sys, math, collections as C
D = sys.argv[1]
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
A = {t["id"]: t for t in jl(f"{D}/stage2/s2-rft2/s2-rft2-multiturn/multiturn.traces.jsonl")}
B = {t["id"]: t for t in jl(f"{D}/stage2/s2-grpo-mt/s2-grpo-mt-multiturn/multiturn.traces.jsonl")}
both = lambda t: int(t["scores"]["task_success"] == 1 and t["lang_match"] == 1)
def mcnemar(ids):
    b = sum(both(A[i]) and not both(B[i]) for i in ids); c = sum(both(B[i]) and not both(A[i]) for i in ids)
    n = b + c; p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * 2 / 2**n if n else 1
    return b, c, min(p, 1)
ids = sorted(set(A) & set(B)); print("paired", len(ids))
for name, sel in [("all", ids), ("switch", [i for i in ids if A[i]["condition"] == "switch"]),
                  ("switch->en", [i for i in ids if A[i]["condition"] == "switch" and A[i]["lang"] == "en"]),
                  ("same", [i for i in ids if A[i]["condition"] != "switch"])]:
    b, c, p = mcnemar(sel)
    print(f"{name:11s} n={len(sel):3d} rft2={sum(both(A[i]) for i in sel)/len(sel)*100:.1f} grpo={sum(both(B[i]) for i in sel)/len(sel)*100:.1f} lost={b} gained={c} p={p:.4f}")
for name, M in [("rft2", A), ("grpo", B)]:
    s = [M[i] for i in ids]
    print(name, "false_action", round(sum(t["scores"]["false_action"] for t in s) / len(s) * 100, 1),
          "task", round(sum(t["scores"]["task_success"] for t in s) / len(s) * 100, 1),
          "lang", round(sum(t["lang_match"] for t in s) / len(s) * 100, 1),
          "last_reply_chars", round(sum(len((t["raw_outputs"][-1] if t.get("raw_outputs") else "") or "") for t in s) / len(s)))
# leakage: eval vs GRPO training prompts
R = jl(f"{D}/round2/grpo-mt/rollouts.jsonl")
tr = {r["id"] for r in R}; parts = lambda i: set(i.replace("switch:", "").split(">"))
trp = set().union(*(parts(i) for i in tr))
print("train prompts", len(tr), "exact overlap", len(tr & set(ids)), "component overlap", sum(bool(parts(i) & trp) for i in ids))
# reward hacking audit on rollouts
print("rollouts", len(R))
hi = [r for r in R if r["reward"] >= max(x["reward"] for x in R) - 1e-9]
print("max-reward rollouts", len(hi), "of which task fail", sum(r["task_success"] < 1 for r in hi), "lang miss", sum(r["lang_match"] < 1 for r in hi), "false_action", sum(r["false_action"] > 0 for r in hi))
pos = [r for r in R if (r["advantage"] or 0) > 0]
print("positive-adv rollouts", len(pos), "task fail", sum(r["task_success"] < 1 for r in pos), "false_action", sum(r["false_action"] > 0 for r in pos), "echo", sum(r["echo"] > 0 for r in pos), "lang miss", sum(r["lang_match"] < 1 for r in pos))
fa = C.Counter(r["step"] for r in R if r["false_action"] > 0); n = C.Counter(r["step"] for r in R)
print("false_action by step", {s: f"{fa[s]}/{n[s]}" for s in sorted(n) if fa[s] / n[s] > 0.08})
L = json.load(open(f"{D}/round2/grpo-mt/log.json", encoding="utf-8"))
print("corr(reward, both) over steps:", round(__import__("statistics").correlation([x["reward"] for x in L], [x["both"] for x in L]), 2))
