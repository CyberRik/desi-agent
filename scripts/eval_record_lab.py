"""[R] Record the Desi Agent Lab tour (record mode, autoplay) as a crisp 1080p video.

Uses Chrome's DevTools screencast (lossless PNG frames with real timestamps) instead of a low-bitrate
browser recording, renders the 1280x720 layout at 1.5x pixel density (1920x1080), and encodes with ffmpeg.
Stops when the tour reaches its last step and has held it.

Encoding builds the constant-30 fps timeline itself (each output frame = the latest screencast frame captured at
or before that instant) and pipes PNGs to ffmpeg, so the video length is fixed by the frame count. The earlier
concat-demuxer version relied on per-image `duration` lines, which some ffmpeg builds (seen on Windows) stretched
from 85 s to 7 min of frozen frames. After encoding, the output duration is checked against the captured span.

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
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--chrome", help="path to a Chrome/Chromium binary (default: installed Chrome via channel='chrome')")
    a = p.parse_args()

    tmp = Path(tempfile.mkdtemp())
    frames: list[tuple[float, Path]] = []
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = (pw.chromium.launch(executable_path=a.chrome, headless=True) if a.chrome
                   else pw.chromium.launch(channel="chrome", headless=True))
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

    # constant-fps timeline built here: output frame k shows the latest screencast frame captured at or before
    # t0 + k/fps. Independent of how a given ffmpeg build treats concat `duration` lines.
    frames.sort(key=lambda f: f[0])
    t_first = frames[0][0]
    span = frames[-1][0] - t_first + 0.5
    n_out = max(1, round(span * a.fps))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    codec = (["-c:v", "libx264", "-preset", "slow", "-crf", "16"] if "libx264" in encs
             else ["-c:v", "libopenh264", "-b:v", "10M"])
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps), "-i", "-",
                           "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p", *codec, "-r", str(a.fps),
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    j, cache = 0, (None, b"")
    for k in range(n_out):
        t = t_first + k / a.fps
        while j + 1 < len(frames) and frames[j + 1][0] <= t:
            j += 1
        if cache[0] != j:
            cache = (j, frames[j][1].read_bytes())
        ff.stdin.write(cache[1])
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    out_s = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                           "-of", "csv=p=0", str(out)], text=True).strip())
    if abs(out_s - span) > 1.5:
        raise SystemExit(f"encoded video is {out_s:.1f} s but the capture spans {span:.1f} s; not publishing")
    secs = frames[-1][0] - frames[0][0]
    man = {"out": str(out), "url": a.url, "frames": len(frames), "seconds": round(secs, 1),
           "video_seconds": round(out_s, 1), "fps": a.fps, "encoder": codec[1], "steps": n_steps,
           "last_step_index": last_idx}
    Path(f"{out}.manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man))


if __name__ == "__main__":
    main()
