"""[R] Record the Desi Agent Lab tour (record mode, autoplay) as a crisp 1080p video.

Uses Chrome's DevTools screencast (lossless PNG frames with real timestamps) instead of a low-bitrate
browser recording, renders the 1280x720 layout at 1.5x pixel density (1920x1080), and encodes with ffmpeg.
Stops when the tour reaches its last step and has held it.

    python scripts/eval_record_lab.py --out video/desi-agent-lab-tour.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

URL = "https://toolw-desi-agent-lab.static.hf.space/?record&speed=1"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=URL)
    p.add_argument("--out", default="video/desi-agent-lab-tour.mp4")
    p.add_argument("--max-seconds", type=float, default=240)
    p.add_argument("--tail", type=float, default=4.0, help="seconds to keep after the last step settles")
    a = p.parse_args()

    tmp = Path(tempfile.mkdtemp())
    frames: list[tuple[float, Path]] = []
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        page = ctx.new_page()
        cdp = ctx.new_cdp_session(page)

        def on_frame(ev):
            path = tmp / f"f{len(frames):05d}.png"
            path.write_bytes(base64.b64decode(ev["data"]))
            frames.append((ev["metadata"]["timestamp"], path))
            cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

        cdp.on("Page.screencastFrame", on_frame)
        page.goto(a.url, wait_until="networkidle")
        page.wait_for_timeout(800)  # fonts + data
        cdp.send("Page.startScreencast", {"format": "png", "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 1})
        t0, last_idx, settled_at = time.time(), -1, None
        n_steps = page.evaluate("document.querySelectorAll('.dots b').length") or 0
        while time.time() - t0 < a.max_seconds:
            page.wait_for_timeout(250)
            idx = page.evaluate("typeof play !== 'undefined' ? play.idx : -1")
            if idx != last_idx:
                last_idx, settled_at = idx, None
            busy = page.evaluate("document.querySelectorAll('.typing').length")
            is_last = n_steps and idx >= n_steps - 1
            if is_last and not busy:
                settled_at = settled_at or time.time()
                if time.time() - settled_at > a.tail:
                    break
        cdp.send("Page.stopScreencast")
        page.wait_for_timeout(300)
        browser.close()

    # concat demuxer with real per-frame durations -> constant 30 fps output
    lst = tmp / "frames.txt"
    with lst.open("w") as f:
        for i, (ts, path) in enumerate(frames):
            dur = (frames[i + 1][0] - ts) if i + 1 < len(frames) else 0.5
            f.write(f"file '{path.as_posix()}'\nduration {max(dur, 0.001):.4f}\n")
        f.write(f"file '{frames[-1][1].as_posix()}'\n")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "16", "-movflags", "+faststart", str(out)], check=True)
    secs = frames[-1][0] - frames[0][0]
    man = {"out": str(out), "url": a.url, "frames": len(frames), "seconds": round(secs, 1), "steps": n_steps,
           "last_step_index": last_idx}
    Path(f"{out}.manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man))


if __name__ == "__main__":
    main()
