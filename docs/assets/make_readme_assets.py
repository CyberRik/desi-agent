"""[F] Build the README visuals from the real result files, so the numbers can't drift.

    pip install playwright && playwright install chromium     # once
    python docs/assets/make_readme_assets.py                  # -> docs/assets/*.png (+ tour.gif with --video)
    python docs/assets/make_readme_assets.py --video path/to/desi-agent-lab-tour.mp4 --clip 26:38

Banner and charts are HTML rendered by headless Chromium at 2x (real Devanagari / Bengali shaping).
Charts read results/<tag>/test.json for base, final SFT and the teacher. Set CHROME=/path/to/chrome to reuse
an existing browser binary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE, FINAL, TEACHER = "tinyaya-global-base", "tinyaya-sft-a-final", "teacher-qwen3-14b"
LANG = [("en", "English"), ("hi", "Hindi"), ("bn", "Bengali"), ("hi-en", "Hinglish"), ("bn-en", "Banglish")]
CAT = [("single_tool", "Single action"), ("multi_tool", "Two-step task"), ("clarify", "Ask for missing info"),
       ("no_tool", "No action needed")]
FONTS = ("https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800"
         "&family=Noto+Sans+Devanagari:wght@500;600&family=Noto+Sans+Bengali:wght@500;600"
         "&family=JetBrains+Mono:wght@400;500&display=swap")


def res(tag: str) -> dict:
    return json.loads((ROOT / "results" / tag / "test.json").read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------- banner (original design)
def banner_html(b: dict, f: dict, t: dict) -> str:
    ob, of_, ot = b["overall"], f["overall"], t["overall"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><link href="{FONTS}" rel="stylesheet"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{width:1280px;height:400px;font-family:Manrope,'Noto Sans Devanagari','Noto Sans Bengali',sans-serif;overflow:hidden;
  background:#0f1730;color:#eef1f8;position:relative}}
.grain{{position:absolute;inset:0;background:
  radial-gradient(900px 420px at 92% 10%,rgba(232,179,73,.16),transparent 60%),
  radial-gradient(700px 380px at 0% 100%,rgba(70,110,220,.22),transparent 60%)}}
.rule{{position:absolute;left:64px;right:64px;top:0;height:4px;background:linear-gradient(90deg,#e8b349,#e8b34900)}}
.left{{position:absolute;left:64px;top:58px;width:640px}}
.kicker{{font-size:15px;letter-spacing:.06em;color:#9aa6c4;font-weight:600}}
h1{{font-size:84px;line-height:.95;font-weight:800;letter-spacing:-.035em;margin-top:16px}}
h1 span{{color:#e8b349}}
.tag{{font-size:21px;line-height:1.45;color:#c9d0e2;margin-top:18px;max-width:600px}}
.langs{{display:flex;gap:8px;margin-top:22px;flex-wrap:wrap}}
.langs i{{font-style:normal;font-size:15px;font-weight:600;padding:5px 12px;border-radius:999px;border:1px solid #2b3a66;color:#dfe5f3;background:#16214a}}
.slip{{position:absolute;right:64px;top:46px;width:430px;background:#fbf7ee;color:#1b2033;border-radius:6px;padding:22px 26px 54px;
  box-shadow:0 18px 50px rgba(0,0,0,.45);transform:rotate(1.6deg)}}
.slip:before,.slip:after{{content:"";position:absolute;left:0;right:0;height:10px;
  background:radial-gradient(circle at 6px 0,transparent 5px,#fbf7ee 5.5px) 0 0/12px 10px repeat-x}}
.slip:before{{top:-9px;transform:scaleY(-1)}} .slip:after{{bottom:-9px}}
.slip .h{{display:flex;justify-content:space-between;font-size:12.5px;letter-spacing:.08em;color:#7b7f8f;font-weight:700}}
.row{{display:grid;grid-template-columns:1fr auto;align-items:baseline;padding:11px 0;border-bottom:1px dashed #d9d2c2}}
.row:last-of-type{{border-bottom:0}}
.k{{font-size:15px;color:#4a4f63;font-weight:600}}
.v{{font-size:27px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.v s{{color:#9a9eac;text-decoration:none;font-weight:600;font-size:20px}}
.v b{{color:#1f6f4a}}
.foot{{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#7b7f8f;margin-top:8px}}
.stamp{{position:absolute;right:24px;bottom:12px;transform:rotate(-6deg);border:2.5px solid #1f6f4a;color:#1f6f4a;
  font-weight:800;font-size:14px;letter-spacing:.1em;padding:3px 10px;border-radius:5px;opacity:.85}}
</style></head><body><div class="grain"></div><div class="rule"></div>
<div class="left"><div class="kicker">LEXSI AI SAFETY &amp; ALIGNMENT HACKATHON · TRACK 3</div>
<h1>Desi <span>Agent</span></h1>
<p class="tag">A 3.35B Tiny Aya model, distilled with AgentTune into a banking agent that calls the right tool, with the right
arguments, and asks when something is missing.</p>
<div class="langs"><i>English</i><i>हिन्दी</i><i>বাংলা</i><i>Hinglish</i><i>Banglish</i></div></div>
<div class="slip"><div class="h"><span>EVALUATION SLIP</span><span>{f['n']} TEST CONVERSATIONS</span></div>
<div class="row"><span class="k">Task success</span><span class="v"><s>{pct(ob['task_success'])} →</s> <b>{pct(of_['task_success'])}</b></span></div>
<div class="row"><span class="k">Wrong bank actions</span><span class="v"><s>{pct(ob['false_action'])} →</s> <b>{pct(of_['false_action'])}</b></span></div>
<div class="row"><span class="k">14B teacher, same test</span><span class="v">{pct(ot['task_success'])}</span></div>
<div class="foot">fp16 · greedy · mock bank · base → fine-tuned</div>
<div class="stamp">EVALUATED</div></div>
</body></html>"""


# ---------------------------------------------------------------- dumbbell charts (dataviz skill: emphasis form)
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  base="#898781", final="#2a78d6", teacher="#52514e"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 base="#898781", final="#3987e5", teacher="#c3c2b7"),
}


def dumbbell_svg(rows: list[tuple[str, float, float, float, int]], th: dict, title: str, w: int = 560) -> str:
    """rows: (label, base, final, teacher, n). x = 0..100%. Base + final are labelled (relief rule)."""
    left, right, top, rh = 150, 28, 44, 44
    pw = w - left - right
    h = top + rh * len(rows) + 30
    x = lambda v: left + v * pw  # noqa: E731
    g = [f'<text x="0" y="18" fill="{th["ink"]}" font-size="15" font-weight="700">{title}</text>']
    for v in (0, .25, .5, .75, 1):
        g.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 10}" y2="{h - 26}" stroke="{th["grid"]}" stroke-width="1"/>')
        g.append(f'<text x="{x(v):.1f}" y="{h - 8}" fill="{th["muted"]}" font-size="11.5" text-anchor="middle">{int(v * 100)}%</text>')
    for i, (lab, b, f, t, n) in enumerate(rows):
        y = top + rh * i + rh / 2 - 6
        g.append(f'<text x="0" y="{y + 4:.1f}" fill="{th["ink"]}" font-size="13.5" font-weight="600">{lab}</text>')
        g.append(f'<text x="0" y="{y + 19:.1f}" fill="{th["muted"]}" font-size="11">n={n}</text>')
        g.append(f'<line x1="{x(b):.1f}" x2="{x(f):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["axis"]}" stroke-width="2"/>')
        if t is not None:
            g.append(f'<circle cx="{x(t):.1f}" cy="{y:.1f}" r="5.5" fill="{th["surface"]}" stroke="{th["teacher"]}" stroke-width="2"/>')
        g.append(f'<circle cx="{x(b):.1f}" cy="{y:.1f}" r="6" fill="{th["base"]}" stroke="{th["surface"]}" stroke-width="2"/>')
        g.append(f'<circle cx="{x(f):.1f}" cy="{y:.1f}" r="7" fill="{th["final"]}" stroke="{th["surface"]}" stroke-width="2"/>')
        if b > 0.08:  # base value left of its dot; near 0% there is no room, so put it under the dot
            g.append(f'<text x="{x(b) - 10:.1f}" y="{y + 4:.1f}" fill="{th["ink2"]}" font-size="12" text-anchor="end">{pct(b)}</text>')
        else:
            g.append(f'<text x="{x(b):.1f}" y="{y + 21:.1f}" fill="{th["ink2"]}" font-size="12" text-anchor="middle">{pct(b)}</text>')
        g.append(f'<text x="{x(f) - 10:.1f}" y="{y - 11:.1f}" fill="{th["ink"]}" font-size="12.5" font-weight="700" text-anchor="end">{pct(f)}</text>')
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Manrope,sans-serif">{"".join(g)}</svg>'


def chart_html(b: dict, f: dict, t: dict, mode: str) -> str:
    th = THEMES[mode]
    row = lambda grp, key, lab: (lab, b[grp][key]["task_success"], f[grp][key]["task_success"],  # noqa: E731
                                 t[grp].get(key, {}).get("task_success"), f[grp][key]["n"])
    langs = dumbbell_svg([row("by_lang", k, lab) for k, lab in LANG], th, "By language")
    cats = dumbbell_svg([row("by_category", k, lab) for k, lab in CAT], th, "By task type")
    leg = (f'<span><i style="background:{th["base"]}"></i>Base Tiny Aya</span>'
           f'<span><i style="background:{th["final"]}"></i>Fine-tuned (ours)</span>'
           f'<span><i class="ring" style="border-color:{th["teacher"]}"></i>Qwen3-14B teacher</span>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><link href="{FONTS}" rel="stylesheet"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{width:1280px;background:{th['surface']};color:{th['ink']};font-family:Manrope,sans-serif;padding:34px 44px 26px}}
h2{{font-size:22px;font-weight:800;letter-spacing:-.01em}}
.sub{{color:{th['ink2']};font-size:14px;margin-top:4px}}
.leg{{display:flex;gap:22px;margin:18px 0 10px;font-size:13px;color:{th['ink2']}}}
.leg i{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px;vertical-align:-1px}}
.leg i.ring{{background:transparent;border:2px solid}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:56px;margin-top:8px}}
.note{{color:{th['muted']};font-size:12px;margin-top:14px}}
</style></head><body>
<h2>Task success on {f['n']} held-out test conversations</h2>
<div class="sub">Each row: base model → fine-tuned. Hollow ring: the 14B teacher whose trajectories we distilled.</div>
<div class="leg">{leg}</div><div class="grid"><div>{langs}</div><div>{cats}</div></div>
<div class="note">fp16 evaluation, greedy decoding, mock bank. Generated from results/*/test.json by docs/assets/make_readme_assets.py.</div>
</body></html>"""


def render(pages: dict[str, tuple[str, int, int | None]]) -> None:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        kw = {"executable_path": os.environ["CHROME"]} if os.environ.get("CHROME") else {}
        br = pw.chromium.launch(**kw)
        for name, (html, w, h) in pages.items():
            pg = br.new_page(viewport={"width": w, "height": h or 200}, device_scale_factor=2)
            pg.set_content(html, wait_until="networkidle")
            pg.wait_for_timeout(600)
            pg.screenshot(path=str(OUT / name), full_page=h is None)
            print("[F] wrote", (OUT / name).relative_to(ROOT))
        br.close()


def gif(video: str, clip: str) -> None:
    a, b = clip.split(":")
    out = OUT / "tour.gif"
    pal = OUT / "_palette.png"
    vf = "fps=12,scale=960:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", a, "-to", b, "-i", video, "-vf", f"{vf},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", a, "-to", b, "-i", video, "-i", str(pal), "-lavfi",
                    f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(out)], check=True)
    pal.unlink()
    print("[F] wrote", out.relative_to(ROOT), f"{out.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="tour mp4 to cut the README GIF from")
    ap.add_argument("--clip", default="26:38", help="start:end seconds of the GIF")
    a = ap.parse_args()
    b, f, t = res(BASE), res(FINAL), res(TEACHER)
    render({"banner.png": (banner_html(b, f, t), 1280, 400),
            "results-light.png": (chart_html(b, f, t, "light"), 1280, None),
            "results-dark.png": (chart_html(b, f, t, "dark"), 1280, None)})
    if a.video:
        gif(a.video, a.clip)


if __name__ == "__main__":
    main()
