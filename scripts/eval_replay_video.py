"""[R] Replay video: base Tiny Aya vs our fine-tuned adapter, side by side, from REAL test-eval traces.

Nothing is re-generated or edited: every tool call, tool result and reply shown is read from
results/<tag>/test.traces.jsonl (vLLM fp16 eval, greedy). Frames are laid out as HTML (so Hindi and
Bengali shape correctly), captured with headless Chrome via Playwright, and encoded with ffmpeg.

    python scripts/eval_replay_video.py --out video/desi-agent-replay.mp4
Needs: playwright (uses the installed Chrome, no browser download), ffmpeg on PATH. Traces are on
HF dataset ToolW/desi-agent-eval-traces if results/*/test.traces.jsonl is missing locally.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE, FINAL = "tinyaya-global-base", "tinyaya-sft-a-final"
LANG = {"en": "English", "hi": "Hindi", "bn": "Bengali", "hi-en": "Hinglish", "bn-en": "Banglish"}
CAT = {"clarify": "missing info: should ask, not act", "multi_tool": "two-step task",
       "single_tool": "single action", "no_tool": "no action needed"}

# (episode id, English gloss of the customer message, base verdict, fine-tuned verdict)
EPISODES = [
    ("hi-en-clarify-upi_no_ref-02271", None,
     "Calls check_upi_status with a placeholder instead of a reference number", "Asks for the 12-digit UPI reference"),
    ("hi-clarify-block_card_no_card-01028", "My card is lost, block it right away. Please issue me a new card.",
     "Invents card number 123456 and blocks it", "Asks for the last 4 digits and the reason first"),
    ("bn-multi_tool-block_then_balance-01563",
     "My credit card ending 8907 was stolen. Block it and tell me my current outstanding.",
     "Blocks it as 'lost', never checks the balance", "Blocks it as stolen, then fetches the outstanding"),
    ("bn-en-multi_tool-txn_then_dispute-02723", None,
     "Disputes transaction 'YOUR_TRANSACTION_ID'", "Looks up recent transactions, then disputes the right one"),
    ("en-single_tool-block_card-00097", None,
     "Blocks the card with reason 'lost'", "Blocks the card with reason 'fraud'"),
]


def load(tag: str) -> dict[str, dict]:
    p = ROOT / "results" / tag / "test.traces.jsonl"
    if not p.exists():
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download("ToolW/desi-agent-eval-traces", f"results/{tag}/test.traces.jsonl", repo_type="dataset"))
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def steps(trace: dict) -> list[tuple[str, str]]:
    """[(kind, text)] after the user turn: kind in call | result | reply."""
    out = []
    for m in trace["messages"][1:]:
        if m.get("tool_calls"):
            for c in m["tool_calls"]:
                out.append(("call", f'{c["function"]["name"]}({c["function"]["arguments"]})'))
        elif m["role"] == "tool":
            out.append(("result", m["content"]))
        elif m["role"] == "assistant":
            out.append(("reply", m.get("content") or ""))
    return out


def esc(s: str, n: int = 260) -> str:
    s = s if len(s) <= n else s[: n - 1] + "…"
    return html.escape(s)


def column(title: str, cls: str, trace: dict, verdict: str, ok: bool, t0: float) -> tuple[str, float]:
    items, t = [], t0
    for kind, text in steps(trace):
        label = {"call": "tool call", "result": "tool result", "reply": "reply"}[kind]
        items.append(f'<div class="step {kind}" data-t="{t:.2f}"><span class="tag">{label}</span>'
                     f'<span class="txt">{esc(text, 150 if kind == "result" else 260)}</span></div>')
        t += 1.3
    mark = "✓" if ok else "✗"
    items.append(f'<div class="verdict {"good" if ok else "bad"}" data-t="{t:.2f}">{mark} {html.escape(verdict)}</div>')
    return f'<div class="col {cls}"><div class="colhead">{title}</div>{"".join(items)}</div>', t


def build_scenes(base: dict, final: dict, res: dict) -> list[tuple[float, str]]:
    scenes = [(4.0, """<div class="title" data-t="0"><div class="kicker">Lexsi AI Safety &amp; Alignment Hackathon · Track 3</div>
      <h1>Desi Agent</h1><div class="sub">Tiny Aya Global (3.35B) fine-tuned with AgentTune into a banking tool-calling agent
      for English, Hindi, Bengali, Hinglish and Banglish</div>
      <div class="note" data-t="1.2">Replayed from real test-set traces · base vs fine-tuned · same prompts, same tools</div></div>""")]
    for ep_id, gloss, bad, good in EPISODES:
        b, f = base[ep_id], final[ep_id]
        user = b["messages"][0]["content"]
        g = f'<div class="gloss" data-t="0.6">“{html.escape(gloss)}”</div>' if gloss else ""
        lb, tb = column("Base Tiny Aya", "base", b, bad, False, 1.4)
        lf, tf = column("Fine-tuned (ours)", "ft", f, good, True, 1.4)
        dur = max(tb, tf) + 2.6
        scenes.append((dur, f"""<div class="ep"><div class="meta" data-t="0"><span class="pill">{LANG[b['lang']]}</span>
          <span class="pill dim">{CAT[b['category']]}</span><span class="eid">{ep_id}</span></div>
          <div class="user" data-t="0.2"><span class="tag">customer</span>{html.escape(user)}</div>{g}
          <div class="cols">{lb}{lf}</div></div>"""))
    o_b, o_f, o_t = res[BASE]["overall"], res[FINAL]["overall"], res["teacher-qwen3-14b"]["overall"]
    def bar_rows(group: str, keys: list[str], names: dict, t0: float) -> str:
        rows = []
        for i, k in enumerate(keys):
            vb, vf = res[BASE][group][k]["task_success"], res[FINAL][group][k]["task_success"]
            rows.append(f'<div class="bar" data-t="{t0 + 0.2 * i:.2f}"><div class="bl">{names[k]}</div>'
                        f'<div class="track"><div class="fill b" style="width:{vb * 100:.1f}%"></div><span>{vb * 100:.1f}%</span></div>'
                        f'<div class="track"><div class="fill f" style="width:{vf * 100:.1f}%"></div><span>{vf * 100:.1f}%</span></div></div>')
        return "".join(rows)
    tasks = {"single_tool": "Single action", "multi_tool": "Two-step", "clarify": "Ask first", "no_tool": "No action"}
    bars = [f'<div class="grid2"><div><div class="gh" data-t="0.8">By language</div>'
            + bar_rows("by_lang", ["en", "hi", "bn", "hi-en", "bn-en"], LANG, 0.9)
            + '</div><div><div class="gh" data-t="1.9">By task type</div>'
            + bar_rows("by_category", list(tasks), tasks, 2.0) + "</div></div>"]
    scenes.append((11.0, f"""<div class="res"><h2 data-t="0">Results on 486 held-out test episodes</h2>
      <div class="kpis" data-t="0.4">
        <div class="kpi"><div class="v">{o_b['task_success'] * 100:.1f}% → {o_f['task_success'] * 100:.1f}%</div><div class="k">task success</div></div>
        <div class="kpi"><div class="v">{o_b['false_action'] * 100:.1f}% → {o_f['false_action'] * 100:.1f}%</div><div class="k">unwanted actions (block / dispute / transfer when it shouldn't)</div></div>
        <div class="kpi"><div class="v">{o_t['task_success'] * 100:.1f}%</div><div class="k">Qwen3-14B teacher, same test</div></div></div>
      <div class="legend" data-t="0.7"><span class="sw b"></span>base <span class="sw f"></span>fine-tuned · task success</div>
      {''.join(bars)}
      <div class="caveat" data-t="3.2">Synthetic test set from the same generator as training (no shared customer messages).
      fp16 eval, greedy decoding. AuditKIT tool-call exact match: {res[FINAL]['auditkit']['means']['tool_call_exact'] * 100:.1f}%.</div></div>"""))
    scenes.append((4.0, """<div class="title" data-t="0"><h1 class="sm">Data · training · eval · traces</h1>
      <div class="sub">github.com/CyberRik/desi-agent<br>huggingface.co/ToolW</div>
      <div class="note" data-t="0.8">Built with AgentTune, AuditKIT and CuratorKIT by Lexsi Labs</div></div>"""))
    return scenes


CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1280px;height:720px;background:#0e1116;color:#e8ebf0;font-family:'Segoe UI','Nirmala UI',sans-serif;overflow:hidden}
.scene{position:absolute;inset:0;padding:38px 48px;display:none}
[data-t]{opacity:0;transform:translateY(6px)}
.title{height:100%;display:flex;flex-direction:column;justify-content:center;gap:18px}
.kicker{color:#8fa3bf;font-size:20px;letter-spacing:.04em}
h1{font-size:76px;font-weight:700;letter-spacing:-.02em} h1.sm{font-size:46px}
.sub{font-size:26px;color:#c6cfdc;max-width:1050px;line-height:1.35}
.note{font-size:20px;color:#7fd1a8}
.meta{display:flex;gap:10px;align-items:center;margin-bottom:12px}
.pill{background:#2a3c5c;color:#cfe0ff;border-radius:999px;padding:4px 14px;font-size:18px;font-weight:600}
.pill.dim{background:#1d2633;color:#9fb0c6;font-weight:500}
.eid{color:#5f6e82;font-size:14px;margin-left:auto;font-family:Consolas,monospace}
.user{background:#18202b;border:1px solid #273244;border-radius:12px;padding:12px 16px;font-size:22px;line-height:1.4}
.gloss{color:#8fa3bf;font-size:17px;margin:6px 4px 0;font-style:italic}
.tag{display:inline-block;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#8fa3bf;margin-right:10px;
     border:1px solid #33415a;border-radius:6px;padding:1px 6px;vertical-align:2px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px}
.col{background:#131922;border:1px solid #232d3c;border-radius:14px;padding:14px;min-height:380px}
.colhead{font-size:19px;font-weight:700;margin-bottom:10px}
.base .colhead{color:#f4a1a1} .ft .colhead{color:#8be0b4}
.step{font-size:16px;line-height:1.4;margin:8px 0;padding:8px 10px;border-radius:8px;background:#1a222e;word-break:break-word}
.step.call .txt{font-family:Consolas,monospace;color:#ffd58a;font-size:15px}
.step.result .txt{font-family:Consolas,monospace;color:#9fb0c6;font-size:13px}
.step.reply .txt{font-size:16px}
.verdict{margin-top:12px;font-size:18px;font-weight:600;padding:9px 12px;border-radius:9px}
.verdict.bad{background:#3a1d22;color:#ffb3b3} .verdict.good{background:#16352a;color:#a6f0c8}
.res h2{font-size:34px;margin-bottom:18px}
.kpis{display:flex;gap:18px;margin-bottom:18px}
.kpi{background:#18202b;border:1px solid #273244;border-radius:12px;padding:12px 18px;flex:1}
.kpi .v{font-size:36px;font-weight:700} .kpi .k{color:#8fa3bf;font-size:16px}
.legend{color:#8fa3bf;font-size:16px;margin-bottom:6px}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin:0 6px 0 10px}
.sw.b,.fill.b{background:#c46a6a} .sw.f,.fill.f{background:#4fbf8a}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:36px}
.gh{font-size:19px;font-weight:600;color:#c6cfdc;margin:6px 0 4px}
.bar{display:grid;grid-template-columns:120px 1fr 1fr;gap:10px;align-items:center;margin:9px 0}
.bl{font-size:18px}
.track{position:relative;height:24px;background:#1a222e;border-radius:6px}
.fill{height:100%;border-radius:6px}
.track span{position:absolute;right:8px;top:2px;font-size:14px;color:#e8ebf0}
.caveat{margin-top:16px;color:#7f8ea3;font-size:15px;line-height:1.4}
"""

JS = """
const scenes=[...document.querySelectorAll('.scene')];
const durs=scenes.map(s=>parseFloat(s.dataset.d));
window.total=durs.reduce((a,b)=>a+b,0);
window.seek=function(t){
  let acc=0;
  scenes.forEach((s,i)=>{
    const on=t>=acc && t<acc+durs[i];
    s.style.display=on?'block':'none';
    if(on){
      const lt=t-acc, out=Math.min(1,Math.max(0,(acc+durs[i]-t)/0.35));
      s.querySelectorAll('[data-t]').forEach(e=>{
        const a=Math.min(1,Math.max(0,(lt-parseFloat(e.dataset.t))/0.35));
        e.style.opacity=a; e.style.transform=`translateY(${(1-a)*6}px)`;
      });
      s.style.opacity=Math.min(1,lt/0.35,out);
    }
    acc+=durs[i];
  });
};
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="video/desi-agent-replay.mp4")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()

    base, final = load(BASE), load(FINAL)
    res = {t: json.loads((ROOT / "results" / t / "test.json").read_text(encoding="utf-8"))
           for t in (BASE, FINAL, "teacher-qwen3-14b")}
    for ep_id, *_ in EPISODES:  # guard: the story must match the real scores
        assert base[ep_id]["scores"]["task_success"] == 0 and final[ep_id]["scores"]["task_success"] == 1, ep_id
    scenes = build_scenes(base, final, res)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
            + "".join(f'<div class="scene" data-d="{d}">{h}</div>' for d, h in scenes)
            + f"<script>{JS}</script></body></html>")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "replay.html").write_text(page, encoding="utf-8")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        pg = browser.new_page(viewport={"width": 1280, "height": 720})
        pg.goto((tmp / "replay.html").as_uri())
        pg.wait_for_timeout(500)  # fonts
        total = pg.evaluate("window.total")
        n = int(total * a.fps)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags",
                               "+faststart", str(out)], stdin=subprocess.PIPE)
        for i in range(n):
            pg.evaluate(f"window.seek({i / a.fps})")
            ff.stdin.write(pg.screenshot(type="png"))
        ff.stdin.close()
        ff.wait()
        browser.close()
    ends, acc = [], 0.0
    for d, _ in scenes:
        acc += d
        ends.append(round(acc, 2))
    manifest = {"out": str(out), "fps": a.fps, "seconds": round(total, 1), "scene_ends_s": ends,
                "episodes": [e[0] for e in EPISODES],
                "traces": {"base": BASE, "final": FINAL}, "source": "results/<tag>/test.traces.jsonl (fp16 vLLM eval)"}
    Path(f"{out}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
