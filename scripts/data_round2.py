"""[R] Round-2 training prompt/episode sets for online RL / more RFT rounds (K's lane trains on them).

Two known weaknesses, two sets, all `split: train` (plus a small `val`), never built from any test split:

1. grounding  (data/round2/grounding_{train,val}.jsonl, DATA_FORMAT rows, single turn)
   A state-changing request (block_card / raise_dispute / transfer_money) or a UPI status check WITHOUT the identifier
   it needs (card last-4, transaction id, payee UPI id, amount, UPI reference). Gold = ask (category `clarify`,
   empty tool_sequence, must_ask_clarification=true), never invent an identifier. Each one has a positive control
   built from the same message with the identifier added (category `single_tool`, gold = the right call), so the
   policy cannot win by always asking. Sources:
     - en: real customer messages from BANKING77 **train** (PolyAI, via mteb/banking77), intents that map to our tools
       (the `act_no_details` list of scripts/eval_real_queries.py). Any text that also appears in BANKING77 test is
       dropped. Controls append a detail sentence ("The card ends in 4821."). cancel_transfer -> `no_tool`.
     - hi / bn / hi-en / bn-en (+ en transfers): code templates from data/round2/phrases.json
       (scripts/data_round2_phrases.py), values sampled like src/desi_agent/data/scenarios.py.
   Final replies are templates too (not scored by `reward`; they pass the data gates and reward_lang's checks).

2. langswitch (data/round2/langswitch_{train,val}.jsonl, multi-turn prompts, same shape as scripts/eval_rl_signal.py)
   prefix = a full train conversation in L1, turn2 = another train episode's customer message in L2, gold = its gold.
   80% switch (L1 != L2, English targets up-weighted: switching into English was 15% after round 1), 20% same-language
   controls. Round-1 probe targets are excluded so the prompts are new. ~20% of targets are grounding rows (switch +
   missing identifier). `messages` = prefix + the target's full gold trajectory, for SFT if wanted.
   Score a rollout on the turn-2 part only: `reward_lang(msgs[len(prefix):], gold, lang)` (or `api.reward`).

Every row passes: validate_data (grounding), the user/final data gates of data/checks.py, lang_match on the reply,
`asks()` for clarify replies, and reward_lang(gold trajectory) == 1.6 (the max). Leakage vs data/test.jsonl,
data/test_ood.jsonl and BANKING77 test is checked (exact, digit-masked normalised, and char-4gram Jaccard >= 0.8)
and reported in the manifest; offending rows are dropped.

    python scripts/data_round2.py --out data/round2
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desi_agent import tools  # noqa: E402
from desi_agent.data.checks import final_problems, required_refs, user_msg_problems  # noqa: E402
from desi_agent.data.scenarios import LANGS, UPI_IDS, _amount  # noqa: E402
from desi_agent.data.splits import load_split  # noqa: E402
from desi_agent.eval.language import lang_match  # noqa: E402
from desi_agent.eval.metrics import reward  # noqa: E402
from desi_agent.eval.rewards import asks, reward_lang  # noqa: E402

SEED = 1234
FULL_LANG_REWARD = 1.6
VAL_FRAC = 0.08

# BANKING77 intent -> (group, dispute reason). Group decides the tool and the missing identifier.
B77 = {
    "lost_or_stolen_card": ("block", None), "compromised_card": ("block", "fraud"), "card_swallowed": ("block", None),
    "card_payment_not_recognised": ("dispute", "unauthorized"), "transaction_charged_twice": ("dispute", "duplicate"),
    "cash_withdrawal_not_recognised": ("dispute", "unauthorized"),
    "direct_debit_payment_not_recognised": ("dispute", "unauthorized"),
    "extra_charge_on_statement": ("dispute", "wrong_amount"), "wrong_amount_of_cash_received": ("dispute", "wrong_amount"),
    "pending_transfer": ("upi", None), "transfer_not_received_by_recipient": ("upi", None),
    "cancel_transfer": ("no_tool", None),
}
# rows per intent: (negatives, of which also get a positive control)
B77_N = {"lost_or_stolen_card": 40, "compromised_card": 30, "card_swallowed": 15, "card_payment_not_recognised": 14,
         "transaction_charged_twice": 14, "cash_withdrawal_not_recognised": 12, "direct_debit_payment_not_recognised": 12,
         "extra_charge_on_statement": 12, "wrong_amount_of_cash_received": 12, "pending_transfer": 18,
         "transfer_not_received_by_recipient": 18, "cancel_transfer": 15}
NO_CONTROL = {"card_swallowed", "cancel_transfer"}
# template pairs per language: group -> number of (negative, control) pairs
TPL_PAIRS = {"block": 30, "dispute": 28, "upi": 22, "transfer_no_upi": 12, "transfer_no_amount": 12}
TPL_GENERIC_BLOCK = 8   # extra negatives with neither card number nor reason
EN_TPL_PAIRS = {"transfer_no_upi": 14, "transfer_no_amount": 14}

FORBID = {"block": [r"\d{4}"], "dispute": [r"TXN\d", r"txn\s*id\s*\S*\d", r"\d{6,}"], "upi": [r"\d{6,}"],
          "transfer_no_upi": [r"@"], "transfer_no_amount": [r"(?<![\w.])\d+(?![\w@])"]}


def stolen_or_lost(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\bstol|theft|thief|robbed|pickpocket|snatch", t):
        return "stolen"
    if re.search(r"\blost\b|\blose\b|losing|misplac|can'?t find|cannot find|missing|left my card|dropped", t):
        return "lost"
    return None


def norm(text: str) -> str:
    t = re.sub(r"[0-9\u0966-\u096f\u09e6-\u09ef]", "0", (text or "").lower())
    t = re.sub(r"[^\w\s@]", " ", t)
    return " ".join(t.split())


def grams(text: str) -> set[str]:
    t = norm(text)
    return {t[i:i + 4] for i in range(len(t) - 3)}


def jacc(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def fill(tpl: str, **kv) -> str:
    for k, v in kv.items():
        tpl = tpl.replace(f"<<{k}>>", str(v))
    assert "<<" not in tpl, tpl.encode("ascii", "backslashreplace")
    return tpl


def sentence(parts: list[str], lang: str, rng: random.Random) -> str:
    parts = [p.strip() for p in parts if p is not None]
    if parts and parts[0] and len(parts) > 1:  # opener joins the first sentence: "Bhai, mera card ..."
        keep = lang in ("hi", "bn") or re.match(r"I\b", parts[1]) or "@" in parts[1].split()[0]
        first = parts[1] if keep else parts[1][0].lower() + parts[1][1:]
        parts = [f"{parts[0]}, {first}"] + parts[2:]
    parts = [p for p in parts if p]
    if lang in ("hi", "bn"):
        return "\u0964 ".join(parts) + "\u0964"
    out = []
    for p in parts:
        out.append(p if "@" in p.split()[0] else p[0].upper() + p[1:])
    ends = [".", ".", "!"] + ([] if "please" in out[-1].lower() else [" please."])
    return ". ".join(out) + rng.choice(ends)


def episode(lang: str, user: str, calls: list[tuple[str, dict]], reply_tpls: list[str], rng: random.Random, **kv) -> dict:
    msgs = [{"role": "user", "content": user}]
    results = []
    for i, (name, args) in enumerate(calls):
        out = tools.execute(name, args)
        results.append(out)
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "name": name, "content": out})
    vals = dict(kv)
    if calls:
        r = json.loads(results[-1])
        name = calls[-1][0]
        if name == "block_card":
            vals.update(ref=r["reference"], eta=r["replacement_eta_days"])
        elif name == "raise_dispute":
            vals.update(dsp=r["dispute_id"], days=r["resolution_days"])
        elif name == "check_upi_status":
            vals.update(ref=r["upi_ref"])
        elif name == "transfer_money":
            vals.update(ref=r["upi_ref"])
    msgs.append({"role": "assistant", "content": fill(rng.choice(reply_tpls), **vals)})
    return {"messages": msgs, "results": results}


def upi_reply_key(ref: str) -> str:
    return "upi_" + json.loads(tools.execute("check_upi_status", {"upi_ref": ref}))["status"]


def rand_vals(rng: random.Random) -> dict:
    return {"last4": f"{rng.randint(0, 9999):04d}", "txn": f"TXN{rng.randint(100000, 999999)}",
            "ref": f"{rng.randint(10**11, 10**12 - 1)}", "upi": rng.choice(UPI_IDS), "amt": _amount(rng)}


def row(rid, lang, category, source, ep, calls, meta) -> dict:
    return {"id": rid, "lang": lang, "category": category, "split": "train", "source": source,
            "messages": ep["messages"],
            "gold": {"tool_sequence": [{"name": n, "arguments": a} for n, a in calls],
                     "must_ask_clarification": category == "clarify"},
            "meta": {**meta, "round": 2}}


# ------------------------------------------------------------------------------------------------- grounding
def template_pairs(lang: str, P: dict, rng: random.Random, counts: dict, generic: int) -> list[tuple[dict, dict | None]]:
    """(negative, control) pairs from the phrase bank. Control is None for neither-detail negatives."""
    R = P["reply"]
    out = []
    for group, n in counts.items():
        for _ in range(n):
            v = rand_vals(rng)
            op = rng.choice(P["opener"])
            if group in ("block", "dispute", "upi"):
                G = P[group]
                reason = rng.choice(sorted(G["problem"]))
                prob, req, det = rng.choice(G["problem"][reason]), rng.choice(G["request"]), fill(rng.choice(G["detail"]), **v)
                neg_txt = sentence([op, prob, req], lang, rng)
                ctl_parts = [op, prob, det, req] if rng.random() < 0.5 else [op, prob, req, det]
                ctl_txt = sentence(ctl_parts, lang, rng)
                if group == "block":
                    calls = [("block_card", {"card_last4": v["last4"], "reason": reason})]
                    neg_reply, ctl_reply, inc = R["clarify_block"], R["block"], [v["last4"]]
                elif group == "dispute":
                    calls = [("raise_dispute", {"transaction_id": v["txn"], "reason": reason})]
                    neg_reply, ctl_reply, inc = R["clarify_dispute"], R["dispute"], [v["txn"]]
                else:
                    calls = [("check_upi_status", {"upi_ref": v["ref"]})]
                    neg_reply, ctl_reply, inc = R["clarify_upi"], R[upi_reply_key(v["ref"])], [v["ref"]]
                pattern = f"{group}:{reason}"
            else:
                T = P["transfer"]
                rel = rng.choice(T["rel"])
                calls = [("transfer_money", {"to_upi_id": v["upi"], "amount": v["amt"]})]
                ctl_txt = sentence([op, fill(rng.choice(T["control"]), upi=v["upi"], amt=v["amt"])], lang, rng)
                if group == "transfer_no_upi":
                    neg_txt = sentence([op, fill(rng.choice(T["no_upi"]), rel=rel, amt=v["amt"])], lang, rng)
                    neg_reply = R["clarify_transfer_upi"]
                else:
                    neg_txt = sentence([op, fill(rng.choice(T["no_amount"]), upi=v["upi"])], lang, rng)
                    neg_reply = R["clarify_transfer_amount"]
                ctl_reply, inc, pattern = R["transfer"], [v["upi"]], group
            neg = {"lang": lang, "group": group, "pattern": pattern, "text": neg_txt, "calls": [], "reply": neg_reply,
                   "vals": v, "forbid": FORBID[group], "inc": [], "origin": "template"}
            ctl = {"lang": lang, "group": group, "pattern": pattern, "text": ctl_txt, "calls": calls, "reply": ctl_reply,
                   "vals": v, "forbid": [], "inc": inc, "origin": "template"}
            out.append((neg, ctl))
    for _ in range(generic):
        v = rand_vals(rng)
        txt = sentence([rng.choice(P["opener"]), rng.choice(P["block"]["generic"])], lang, rng)
        out.append(({"lang": lang, "group": "block", "pattern": "block:generic", "text": txt, "calls": [],
                     "reply": R["clarify_block_reason"], "vals": v, "forbid": FORBID["block"], "inc": [],
                     "origin": "template"}, None))
    return out


def banking77_pairs(P: dict, rng: random.Random) -> tuple[list, dict]:
    from huggingface_hub import hf_hub_download
    rd = lambda s: [json.loads(l) for l in open(hf_hub_download("mteb/banking77", f"{s}.jsonl", repo_type="dataset"),  # noqa: E731
                                                encoding="utf-8")]
    train, test = rd("train"), rd("test")
    test_norm = {norm(r["text"]) for r in test}
    E = P["en"]
    R = E["reply"]
    out, stats = [], {"b77_train_rows": len(train), "b77_dropped_in_test": 0}
    seen = set()
    for intent, n in B77_N.items():
        pool = []
        for r in train:
            t = r["text"].strip()
            if r["label_text"] != intent or re.search(r"[\d@]", t) or norm(t) in seen:
                continue
            if norm(t) in test_norm:
                stats["b77_dropped_in_test"] += 1
                continue
            seen.add(norm(t))
            pool.append(t)
        group, dreason = B77[intent]
        for t in rng.sample(pool, min(n, len(pool))):
            v = rand_vals(rng)
            base = {"lang": "en", "group": group, "pattern": f"b77:{intent}", "vals": v, "origin": "banking77-train",
                    "b77_intent": intent}
            if group == "no_tool":
                out.append(({**base, "text": t, "calls": [], "reply": R["no_tool_cancel"], "forbid": [], "inc": [],
                             "category": "no_tool"}, None))
                continue
            ctl = None
            if group == "block":
                reason = dreason or stolen_or_lost(t)
                neg = {**base, "text": t, "calls": [], "forbid": FORBID["block"], "inc": [],
                       "reply": R["clarify_block"] if reason else R["clarify_block_reason"]}
                if reason and intent not in NO_CONTROL:
                    det = fill(rng.choice(E["block"]["detail"]), **v)
                    tail = "" if re.search(r"\bblock|freeze|cancel", t, re.I) else " Please block it."
                    ctl = {**base, "text": f"{_end(t)} {det}{tail}", "inc": [v["last4"]], "forbid": [],
                           "calls": [("block_card", {"card_last4": v["last4"], "reason": reason})], "reply": R["block"]}
            elif group == "dispute":
                neg = {**base, "text": t, "calls": [], "forbid": FORBID["dispute"], "inc": [], "reply": R["clarify_dispute"]}
                det = fill(rng.choice(E["dispute"]["detail"]), **v)
                ctl = {**base, "text": f"{_end(t)} {det}", "inc": [v["txn"]], "forbid": [],
                       "calls": [("raise_dispute", {"transaction_id": v["txn"], "reason": dreason})], "reply": R["dispute"]}
            else:
                neg = {**base, "text": t, "calls": [], "forbid": FORBID["upi"], "inc": [], "reply": R["clarify_upi"]}
                det = fill(rng.choice(E["upi"]["detail"]), **v)
                ctl = {**base, "text": f"{_end(t)} {det}", "inc": [v["ref"]], "forbid": [],
                       "calls": [("check_upi_status", {"upi_ref": v["ref"]})], "reply": R[upi_reply_key(v["ref"])]}
            out.append((neg, ctl))
    return out, stats


def _end(t: str) -> str:
    return t if t[-1:] in ".!?" else t + "."


def materialize(spec: dict, rid: str, rng: random.Random) -> tuple[dict | None, list[str]]:
    lang, v = spec["lang"], spec["vals"]
    category = spec.get("category") or ("single_tool" if spec["calls"] else "clarify")
    ep = episode(lang, spec["text"], spec["calls"], spec["reply"], rng, **v)
    reply = ep["messages"][-1]["content"]
    probs = []
    if spec["origin"] != "banking77-train" or spec["calls"]:
        probs += ["user:" + p for p in user_msg_problems(spec["text"], lang, spec["inc"], spec["forbid"])]
    else:
        probs += ["user:" + p for p in user_msg_problems(spec["text"], lang, [], spec["forbid"])
                  if p not in ("not_romanized_en",)]
    ids = [str(x) for x in v.values()]
    probs += ["reply:" + p for p in final_problems(reply, lang, ids, spec["text"], required_refs(ep["results"]))]
    if not lang_match(reply, lang):
        probs.append("reply:lang_match")
    if category == "clarify" and not asks(reply, lang):
        probs.append("reply:no_ask")
    meta = {"set": "grounding", "variant": "control" if spec["calls"] else ("no_tool" if category == "no_tool" else "missing"),
            "group": spec["group"], "pattern": spec["pattern"], "origin": spec["origin"]}
    if spec.get("b77_intent"):
        meta["b77_intent"] = spec["b77_intent"]
    src = "human" if spec["origin"] == "banking77-train" and not spec["calls"] else "template"
    r = row(rid, lang, category, src, ep, spec["calls"], meta)
    rl = reward_lang(r["messages"], r["gold"], lang)
    if abs(rl - FULL_LANG_REWARD) > 1e-6:
        probs.append(f"reward_lang={rl:.2f}")
    return (None if probs else r), probs


def build_grounding(P: dict, rng: random.Random) -> tuple[list, dict]:
    pairs, stats = banking77_pairs(P, rng)
    for lang in LANGS:
        if lang == "en":
            pairs += template_pairs(lang, P[lang], rng, EN_TPL_PAIRS, 0)
        else:
            pairs += template_pairs(lang, P[lang], rng, TPL_PAIRS, TPL_GENERIC_BLOCK)
    rows, drops, seen = [], collections.Counter(), set()
    counter = collections.Counter()
    for neg, ctl in pairs:
        split = "val" if rng.random() < VAL_FRAC else "train"
        if norm(neg["text"]) in seen:
            drops["duplicate_message"] += 1
            continue
        got = []
        for spec in (neg, ctl):
            if spec is None:
                continue
            seen.add(norm(spec["text"]))
            kind = "ctl" if spec["calls"] else "neg"
            counter[(spec["lang"], kind)] += 1
            rid = f"r2g-{spec['lang']}-{spec['group']}-{kind}-{counter[(spec['lang'], kind)]:04d}"
            r, probs = materialize(spec, rid, rng)
            for p in probs:
                drops[p] += 1
            got.append(r)
        if any(r is None for r in got):  # keep pairs intact: drop both halves
            drops["pair_dropped"] += 1
            continue
        for r in got:
            r["split"] = split
            r["meta"]["pair"] = got[0]["id"]
        rows += got
    stats["gate_drops"] = dict(drops)
    return rows, stats


# ------------------------------------------------------------------------------------------------- language switch
def round1_targets() -> set[str]:
    spec = importlib.util.spec_from_file_location("eval_rl_signal", ROOT / "scripts" / "eval_rl_signal.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {e["target"] for e in m.build_train_switch(300)}


def build_switch(base: list[dict], targets: list[dict], n: int, rng: random.Random, exclude: set[str], split: str) -> list:
    by_lang = collections.defaultdict(list)
    for e in base:
        by_lang[e["lang"]].append(e)
    tb = collections.defaultdict(list)
    for e in targets:
        if e["id"] not in exclude:
            tb[e["lang"]].append(e)
    weights = {"en": 0.30, "hi": 0.175, "bn": 0.175, "hi-en": 0.175, "bn-en": 0.175}
    out, used = [], set()
    while len(out) < n:
        cond = "switch" if rng.random() < 0.8 else "same"
        lang = rng.choices(list(weights), weights=list(weights.values()))[0]
        pool = [e for e in tb[lang] if e["id"] not in used]
        if not pool:
            continue
        t = rng.choice(pool)
        used.add(t["id"])
        l1 = rng.choice([l for l in LANGS if l != lang]) if cond == "switch" else lang
        pre = rng.choice([e for e in by_lang[l1] if e["id"] != t["id"]])
        out.append({"id": f"{cond}:{pre['id']}>{t['id']}", "split": split, "condition": cond, "lang1": l1, "lang": lang,
                    "category": t["category"], "target": t["id"],
                    "target_set": "grounding" if t["id"].startswith("r2g-") else "base",
                    "prefix": pre["messages"], "turn2": t["messages"][0], "gold": t["gold"],
                    "messages": pre["messages"] + t["messages"]})
    return out


def check_switch(rows: list[dict]) -> tuple[collections.Counter, set[str]]:
    """The target's own gold trajectory must get full reward_lang (reply in the target language, asks if clarify)."""
    bad, ids = collections.Counter(), set()
    for r in rows:
        tail = r["messages"][len(r["prefix"]):]
        rl = reward_lang(tail, r["gold"], r["lang"])
        if abs(rl - FULL_LANG_REWARD) > 1e-6:
            bad["gold_tail_not_full_reward"] += 1
            ids.add(r["id"])
        if reward(tail, r["gold"]) < 1.3 - 1e-6:
            bad["gold_tail_not_full_contract_reward"] += 1
            ids.add(r["id"])
    return bad, ids


# ------------------------------------------------------------------------------------------------- leakage
def leakage(rows_g: list[dict], rows_s: list[dict], b77_test: list[str]) -> dict:
    test_msgs = []
    for f in ("data/test.jsonl", "data/test_ood.jsonl"):
        for l in (ROOT / f).read_text(encoding="utf-8").splitlines():
            if l.strip():
                e = json.loads(l)
                test_msgs += [(f, m["content"]) for m in e["messages"] if m["role"] == "user" and m.get("content")]
    test_msgs += [("banking77/test", t) for t in b77_test]
    exact = {t.strip() for _, t in test_msgs}
    normed = {norm(t) for _, t in test_msgs}
    tg = [grams(t) for _, t in test_msgs]

    def msgs_of(r):
        return [m["content"] for m in r["messages"] if m["role"] == "user" and m.get("content")]

    def hit(text):
        if text.strip() in exact:
            return "exact"
        if norm(text) in normed:
            return "normalized"
        g = grams(text)
        if any(jacc(g, x) >= 0.8 for x in tg):
            return "jaccard>=0.8"
        return None

    rep = {"n_test_messages": len(test_msgs), "grounding": collections.Counter(), "langswitch": collections.Counter()}
    bad_ids = set()
    for name, rows in (("grounding", rows_g), ("langswitch", rows_s)):
        for r in rows:
            for t in msgs_of(r):
                h = hit(t)
                rep[name]["checked"] += 1
                if h:
                    rep[name][h] += 1
                    bad_ids.add(r["id"])
    rep["grounding"], rep["langswitch"] = dict(rep["grounding"]), dict(rep["langswitch"])
    rep["rows_with_overlap"] = sorted(bad_ids)
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/round2")
    p.add_argument("--phrases", default="data/round2/phrases.json")
    p.add_argument("--n-switch-train", type=int, default=520)
    p.add_argument("--n-switch-val", type=int, default=64)
    a = p.parse_args()
    rng = random.Random(SEED)
    P = json.loads((ROOT / a.phrases).read_text(encoding="utf-8"))
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g_rows, g_stats = build_grounding(P, rng)

    train = [e for e in load_split("train") if e["split"] == "train"]
    val = [e for e in load_split("val") if e["split"] == "val"]
    test_ids = {e["id"] for e in load_split("test")}
    assert not ({e["id"] for e in train + val} & test_ids)
    g_train = [r for r in g_rows if r["split"] == "train"]
    g_val = [r for r in g_rows if r["split"] == "val"]
    excl = round1_targets()
    # ~20% grounding targets: sample them into the target pool at the right ratio
    gt = rng.sample(g_train, min(len(g_train), len(train) // 4))
    s_train = build_switch(train, train + gt, a.n_switch_train, rng, excl, "train")
    s_val = build_switch(val, val + g_val, a.n_switch_val, rng, set(), "val")
    s_bad, s_bad_ids = check_switch(s_train + s_val)

    from huggingface_hub import hf_hub_download
    b77_test = [json.loads(l)["text"] for l in open(hf_hub_download("mteb/banking77", "test.jsonl", repo_type="dataset"),
                                                     encoding="utf-8")]
    leak = leakage(g_rows, s_train + s_val, b77_test)
    drop = set(leak["rows_with_overlap"]) | s_bad_ids
    g_rows = [r for r in g_rows if r["id"] not in drop]
    s_train = [r for r in s_train if r["id"] not in drop and r["target"] not in drop]
    s_val = [r for r in s_val if r["id"] not in drop and r["target"] not in drop]

    files = {"grounding_train.jsonl": [r for r in g_rows if r["split"] == "train"],
             "grounding_val.jsonl": [r for r in g_rows if r["split"] == "val"],
             "langswitch_train.jsonl": s_train, "langswitch_val.jsonl": s_val}
    for name, rows in files.items():
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def cnt(rows, *keys):
        c = collections.Counter("|".join(str(k(r)) for k in keys) for r in rows)
        return dict(sorted(c.items()))

    sha = lambda f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16]  # noqa: E731
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git = None
    manifest = {
        "script": "scripts/data_round2.py", "git_sha": git, "seed": SEED, "model_id": None,
        "inputs": {"data/train.jsonl": sha("data/train.jsonl"), "data/val.jsonl": sha("data/val.jsonl"),
                   a.phrases: sha(a.phrases), "banking77": "mteb/banking77 train.jsonl (test.jsonl only for dedup)"},
        "files": {k: len(v) for k, v in files.items()},
        "grounding": {
            "by_split_lang_variant": cnt(g_rows, lambda r: r["split"], lambda r: r["lang"], lambda r: r["meta"]["variant"]),
            "by_group_variant": cnt(g_rows, lambda r: r["meta"]["group"], lambda r: r["meta"]["variant"]),
            "by_origin": cnt(g_rows, lambda r: r["meta"]["origin"]),
            **g_stats},
        "langswitch": {
            "by_split_condition_lang": cnt(s_train + s_val, lambda r: r["split"], lambda r: r["condition"], lambda r: r["lang"]),
            "by_target_set_category": cnt(s_train + s_val, lambda r: r["target_set"], lambda r: r["category"]),
            "round1_targets_excluded": len(excl), "gold_check_failures": dict(s_bad)},
        "leakage": leak,
        "scoring": "grounding: api.reward(pred_messages, row['gold']) or rewards.reward_lang(pred, gold, row['lang']); "
                   "langswitch: reward_lang(pred[len(row['prefix']):], row['gold'], row['lang'])",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"files": manifest["files"], "gate_drops": g_stats["gate_drops"],
                      "switch_gold_failures": dict(s_bad),
                      "leakage": {k: v for k, v in leak.items() if k != "rows_with_overlap"},
                      "leak_rows": len(leak["rows_with_overlap"])}, indent=1))


if __name__ == "__main__":
    main()
