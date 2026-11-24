"""[F] Online multi-turn GRPO for the language-switch case (Stage 2), on top of a LoRA policy.

Each step:
  1. rollouts  B prompts x G samples. Every sample is a full turn-2 conversation run through the SHARED agent loop
               (`fmt.run_episode`, mock bank, max 6 steps), so tool execution and parsing are exactly as in eval.
               Episodes run in threads; their model calls are batched into one `generate` by `Batcher`.
  2. reward    `rewards.reward_lang` (R's audited reward: contract reward + language term, echo and 'ask first
               must ask' checks) on the turn-2 messages, as scripts/eval_rl_signal.py scores them.
  3. update    group-normalised advantages A = (r - mean_g) / (std_g + eps); groups with no variance are skipped.
               Loss = -A * mean log p(model tokens of turn 2) per sequence, averaged over the batch, one optimizer
               step per batch. Fully on-policy (one update per rollout batch), so the PPO ratio is 1 and clipping
               is a no-op. No KL term (the reference would be the base model, not the starting adapter); drift is
               bounded by few steps and a small learning rate, and every rollout is saved for the hacking audit.

Only the model's own tokens after the prefix are trained: the prefix conversation and tool results are masked.
"""
from __future__ import annotations

import json
import queue
import random
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .. import fmt
from .chat import tokenize_sft

DEFAULTS = {"group": 8, "prompts_per_step": 16, "steps": 25, "lr": 1e-5, "temperature": 1.0, "max_new_tokens": 256,
            "max_len": 2560, "gen_batch": 32, "seed": 1234, "save_every": 10}


class Batcher:
    """Thread-safe policy(chat) -> text whose calls are gathered into batched `HFPolicy.batch` calls."""

    def __init__(self, policy, max_batch: int, wait_s: float = 0.05):
        self.policy, self.max_batch, self.wait_s = policy, max_batch, wait_s
        self.q: queue.Queue = queue.Queue()
        self.calls = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def __call__(self, chat: list[dict]) -> str:
        f: Future = Future()
        self.q.put((chat, f))
        return f.result()

    def _loop(self) -> None:
        while True:
            items = [self.q.get()]
            t = time.time()
            while len(items) < self.max_batch and time.time() - t < self.wait_s:
                try:
                    items.append(self.q.get(timeout=self.wait_s))
                except queue.Empty:
                    break
            try:
                outs = self.policy.batch([c for c, _ in items])
                self.calls += 1
                for (_, f), o in zip(items, outs):
                    f.set_result(o)
            except Exception as e:  # noqa: BLE001 - surface the error in every waiting episode
                for _, f in items:
                    f.set_exception(e)


def score(turn2: list[dict], row: dict) -> dict:
    """reward_lang plus its parts, so the audit can see what moved."""
    from ..eval.language import final_reply, lang_match
    from ..eval.metrics import score_episode
    from ..eval.rewards import _jaccard4, asks, reward_lang
    s = score_episode(turn2, row["gold"])
    reply = final_reply(turn2)
    user = next((m.get("content") or "" for m in turn2 if m["role"] == "user"), "")
    return {"reward": round(reward_lang(turn2, row["gold"], row["lang"]), 4), "task_success": s["task_success"],
            "lang_match": float(lang_match(reply, row["lang"])), "false_action": s["false_action"],
            "echo": float(_jaccard4(reply, user) >= 0.6),
            "ask_missing": float(bool(row["gold"].get("must_ask_clarification")) and not asks(reply, row["lang"])),
            "n_calls": sum(len(m.get("tool_calls") or []) for m in turn2 if m["role"] == "assistant"),
            "reply_chars": len(reply or "")}


def rollouts(policy, rows: list[dict], group: int) -> list[dict]:
    def one(row):
        run = fmt.run_episode(policy, [dict(m) for m in row["prefix"]] + [dict(row["turn2"])])
        turn2 = run.messages[len(row["prefix"]):]
        return {"id": row["id"], "lang": row["lang"], "condition": row.get("condition"), "messages": turn2,
                "truncated": run.truncated, **score(turn2, row)}
    jobs = [r for r in rows for _ in range(group)]
    with ThreadPoolExecutor(len(jobs)) as ex:
        return list(ex.map(one, jobs))


def sequence(tok, row: dict, turn2: list[dict], max_len: int) -> dict | None:
    """Tokens of prefix + turn 2 with labels only on the model's own turn-2 tokens."""
    prefix_ids = tokenize_sft(tok, fmt.to_chat(row["prefix"]), max_len=10**9)["input_ids"]
    enc = tokenize_sft(tok, fmt.to_chat(row["prefix"] + turn2), max_len=10**9)
    if enc["input_ids"][: len(prefix_ids)] != prefix_ids:
        raise ValueError(f"{row['id']}: prefix tokens differ inside the full conversation")
    labels = [-100] * len(prefix_ids) + enc["labels"][len(prefix_ids):]
    if len(enc["input_ids"]) > max_len or all(l == -100 for l in labels):
        return None
    return {"input_ids": enc["input_ids"], "labels": labels}


def advantages(rewards: list[float], group: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(0, len(rewards), group):
        g = rewards[i:i + group]
        sd = statistics.pstdev(g)
        m = statistics.mean(g)
        out += [None] * len(g) if sd < 1e-6 else [(r - m) / (sd + 1e-4) for r in g]
    return out


def pg_step(model, tok, opt, seqs: list[dict], advs: list[float]) -> dict:
    """One optimizer step: mean over sequences of -A * mean token log-prob (on-policy)."""
    import torch
    model.train()
    opt.zero_grad(set_to_none=True)
    n, tot = len(seqs), 0.0
    for s, a in zip(seqs, advs):
        ids = torch.tensor([s["input_ids"]], device=model.device)
        lab = torch.tensor([s["labels"]], device=model.device)
        tgt = lab[0, 1:]
        mask = tgt != -100
        logits = model(input_ids=ids).logits[0, :-1][mask].float()  # only the trained positions (vocab is 262k)
        lp = torch.log_softmax(logits, -1).gather(-1, tgt[mask].unsqueeze(-1)).squeeze(-1)
        loss = -(a * lp.mean()) / n
        loss.backward()
        tot += loss.item()
        del logits, lp
    gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0).item()
    opt.step()
    model.eval()
    return {"loss": tot, "grad_norm": gn}


def summarize(step: int, rolls: list[dict], advs: list, group: int, t: dict) -> dict:
    m = lambda k: round(statistics.mean(r[k] for r in rolls), 4)  # noqa: E731
    sw = [r for r in rolls if r["condition"] == "switch"]
    en = [r for r in sw if r["lang"] == "en"]
    return {"step": step, "reward": m("reward"), "task_success": m("task_success"), "lang_match": m("lang_match"),
            "both": round(statistics.mean(r["task_success"] * r["lang_match"] for r in rolls), 4),
            "switch_both": round(statistics.mean(r["task_success"] * r["lang_match"] for r in sw), 4) if sw else None,
            "switch_en_lang": round(statistics.mean(r["lang_match"] for r in en), 4) if en else None,
            "false_action": m("false_action"), "echo": m("echo"), "ask_missing": m("ask_missing"),
            "reply_chars": m("reply_chars"), "n_calls": m("n_calls"),
            "groups_with_variance": round(sum(a is not None for a in advs[::group]) / max(1, len(advs) // group), 4),
            **t}


def train_grpo_mt(policy, rows: list[dict], cfg: dict, out: Path, on_save=None) -> list[dict]:
    """policy: an HFPolicy whose model is a trainable PeftModel. Returns the per-step log."""
    import torch
    c = {**DEFAULTS, **cfg}
    rng = random.Random(c["seed"])
    torch.manual_seed(c["seed"])
    order = rows[:]
    rng.shuffle(order)
    model, tok = policy.model, policy.tok
    policy.gen_kw.update(do_sample=True, temperature=c["temperature"], top_p=1.0, max_new_tokens=c["max_new_tokens"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=0.0)
    batcher = Batcher(policy, c["gen_batch"])
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    for step in range(c["steps"]):
        lo = (step * c["prompts_per_step"]) % len(order)
        batch = (order + order)[lo:lo + c["prompts_per_step"]]
        t0 = time.time()
        model.eval()
        rolls = rollouts(batcher, batch, c["group"])
        t1 = time.time()
        advs = advantages([r["reward"] for r in rolls], c["group"])
        by_id = {r["id"]: r for r in batch}
        seqs, used = [], []
        for r, a in zip(rolls, advs):
            if a is None or abs(a) < 1e-8:
                continue
            s = sequence(tok, by_id[r["id"]], r["messages"], c["max_len"])
            if s is not None:
                seqs.append(s)
                used.append(a)
        upd = pg_step(model, tok, opt, seqs, used) if seqs else {"loss": 0.0, "grad_norm": 0.0}
        rec = summarize(step, rolls, advs, c["group"],
                        {"n_trained_seqs": len(seqs), "rollout_s": round(t1 - t0), "update_s": round(time.time() - t1),
                         **upd})
        log.append(rec)
        print("[F] grpo", json.dumps(rec), flush=True)
        with (out / "rollouts.jsonl").open("a", encoding="utf-8") as f:
            for r, a in zip(rolls, advs):
                f.write(json.dumps({"step": step, "advantage": a, **r}, ensure_ascii=False) + "\n")
        (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if on_save and ((step + 1) % c["save_every"] == 0 or step + 1 == c["steps"]):
            on_save(step + 1)
    return log
