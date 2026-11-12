"""Agent-to-agent collaboration API on top of GitHub Issues (needs `gh` logged in).

Identity: env DESI_AGENT=R|F, or a one-line file `.agent` in the repo root (gitignored).

    python scripts/collab.py whoami
    python scripts/collab.py bootstrap                 # once: labels + one issue per task in docs/PLAN.md
    python scripts/collab.py status                    # board: every task, owner lane, state
    python scripts/collab.py inbox                     # open messages to me + PRs waiting on my review
    python scripts/collab.py send F "need X" -b "..."  # message the other agent (kind: request|question|fyi)
    python scripts/collab.py reply 42 "text"           # comment on a message/task issue
    python scripts/collab.py done 42 ["final note"]    # close a message
    python scripts/collab.py claim R3                  # assign + status:doing
    python scripts/collab.py block R3 F2               # mark R3 blocked on F2
    python scripts/collab.py review R3                 # status:review (PR open)
    python scripts/collab.py finish R3                 # close task issue
    python scripts/collab.py handoff R3 --done "..." --next "..." [--blocked "..."] [--artifacts "..."]

Everything an agent writes is prefixed with its id, e.g. "[R]".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "lane:R": "1f6feb", "lane:F": "a371f7", "shared": "8b949e",
    "to:R": "1f6feb", "to:F": "a371f7", "msg": "fbca04",
    "status:doing": "0e8a16", "status:blocked": "d73a4a", "status:review": "e99695",
    "kind:request": "c5def5", "kind:question": "c5def5", "kind:fyi": "c5def5",
}


def me() -> str:
    a = os.environ.get("DESI_AGENT") or ((ROOT / ".agent").read_text().strip() if (ROOT / ".agent").exists() else "")
    if a not in ("R", "F"):
        sys.exit("Set your identity: `echo R > .agent` (or F), or env DESI_AGENT.")
    return a


def other(a: str) -> str:
    return "F" if a == "R" else "R"


def gh(*args: str, capture: bool = True) -> str:
    r = subprocess.run(["gh", *args], cwd=ROOT, capture_output=capture, text=True, encoding="utf-8")
    if r.returncode != 0:
        sys.exit(f"gh {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def issues(*extra: str) -> list[dict]:
    return json.loads(gh("issue", "list", "--state", "all", "--limit", "300",
                         "--json", "number,title,state,labels,assignees,updatedAt", *extra))


def task_issue(task_id: str) -> dict:
    for i in issues():
        if i["title"].startswith(f"{task_id}:"):
            return i
    sys.exit(f"no issue titled '{task_id}: ...'. Run `collab.py bootstrap` first?")


def set_status(num: int, status: str | None) -> None:
    for s in ("status:doing", "status:blocked", "status:review"):
        subprocess.run(["gh", "issue", "edit", str(num), "--remove-label", s], cwd=ROOT, capture_output=True)
    if status:
        gh("issue", "edit", str(num), "--add-label", status)


def parse_plan() -> list[dict]:
    rows = []
    for line in (ROOT / "docs/PLAN.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*([RFS]\d+)\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|", line)
        if m:
            rows.append(dict(zip(("id", "title", "owner", "deps", "due"), m.groups())))
    return rows


def cmd_bootstrap(a) -> None:
    for name, color in LABELS.items():
        subprocess.run(["gh", "label", "create", name, "--color", color, "--force"], cwd=ROOT, capture_output=True)
    existing = {i["title"].split(":")[0] for i in issues()}
    for t in parse_plan():
        if t["id"] in existing:
            continue
        lane = {"R": "lane:R", "F": "lane:F"}.get(t["owner"], "shared")
        body = f"Task from docs/PLAN.md\n\n- owner: {t['owner']}\n- depends on: {t['deps']}\n- due: {t['due']}\n"
        gh("issue", "create", "--title", f"{t['id']}: {t['title']}", "--body", body, "--label", lane)
        print("created", t["id"])


def cmd_status(a) -> None:
    for i in sorted((i for i in issues() if re.match(r"^[RFS]\d+:", i["title"])),
                    key=lambda i: (i["title"][0], int(re.match(r"^.(\d+)", i["title"]).group(1)))):
        labels = [l["name"] for l in i["labels"]]
        st = "done" if i["state"] == "CLOSED" else next((l[7:] for l in labels if l.startswith("status:")), "todo")
        who = ",".join(x["login"] for x in i["assignees"]) or "-"
        print(f"#{i['number']:<4} {st:<8} {who:<14} {i['title']}")


def cmd_inbox(a) -> None:
    m = me()
    msgs = json.loads(gh("issue", "list", "--label", f"to:{m}", "--state", "open",
                         "--json", "number,title,updatedAt,labels"))
    print(f"== messages to {m} ({len(msgs)})")
    for i in msgs:
        print(f"#{i['number']:<4} {i['updatedAt'][:16]}  {i['title']}")
    prs = json.loads(gh("pr", "list", "--state", "open", "--json", "number,title,headRefName,reviewDecision"))
    mine = f"{m.lower()}/"
    print("== open PRs from the other lane (need my review if CONTRACT:)")
    for p in prs:
        if not p["headRefName"].startswith(mine):
            print(f"PR #{p['number']:<4} {p['headRefName']:<28} {p['title']}")
    blocked = [i for i in issues("--state", "open") if any(l["name"] == "status:blocked" for l in i["labels"])]
    if blocked:
        print("== blocked tasks")
        for i in blocked:
            print(f"#{i['number']:<4} {i['title']}")


def cmd_send(a) -> None:
    m = me()
    to = a.to.upper()
    gh("issue", "create", "--title", f"[{m}->{to}] {a.title}", "--body", f"[{m}] {a.body or a.title}",
       "--label", f"to:{to}", "--label", "msg", "--label", f"kind:{a.kind}")
    print(f"sent to {to}")


def cmd_reply(a) -> None:
    gh("issue", "comment", str(a.num), "--body", f"[{me()}] {a.text}")


def cmd_done(a) -> None:
    if a.text:
        gh("issue", "comment", str(a.num), "--body", f"[{me()}] {a.text}")
    gh("issue", "close", str(a.num))


def cmd_claim(a) -> None:
    i = task_issue(a.task)
    gh("issue", "edit", str(i["number"]), "--add-assignee", "@me")
    set_status(i["number"], "status:doing")
    gh("issue", "comment", str(i["number"]), "--body", f"[{me()}] claimed, working on it.")
    print(f"claimed #{i['number']} {i['title']}")


def cmd_block(a) -> None:
    i, b = task_issue(a.task), task_issue(a.on)
    set_status(i["number"], "status:blocked")
    gh("issue", "comment", str(i["number"]), "--body", f"[{me()}] blocked on #{b['number']} ({b['title']}).")
    gh("issue", "comment", str(b["number"]), "--body", f"[{me()}] heads-up: #{i['number']} is waiting on this.")


def cmd_review(a) -> None:
    set_status(task_issue(a.task)["number"], "status:review")


def cmd_finish(a) -> None:
    i = task_issue(a.task)
    set_status(i["number"], None)
    gh("issue", "close", str(i["number"]), "--comment", f"[{me()}] done.")


def cmd_handoff(a) -> None:
    i = task_issue(a.task)
    body = (f"[{me()}] HANDOFF\n```\ndone: {a.done}\nnext: {a.next}\nblocked on: {a.blocked or '-'}\n"
            f"artifacts: {a.artifacts or '-'}\n```")
    gh("issue", "comment", str(i["number"]), "--body", body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("whoami").set_defaults(f=lambda a: print(me()))
    s.add_parser("bootstrap").set_defaults(f=cmd_bootstrap)
    s.add_parser("status").set_defaults(f=cmd_status)
    s.add_parser("inbox").set_defaults(f=cmd_inbox)
    x = s.add_parser("send"); x.add_argument("to"); x.add_argument("title"); x.add_argument("-b", "--body")
    x.add_argument("-k", "--kind", default="request", choices=["request", "question", "fyi"]); x.set_defaults(f=cmd_send)
    x = s.add_parser("reply"); x.add_argument("num", type=int); x.add_argument("text"); x.set_defaults(f=cmd_reply)
    x = s.add_parser("done"); x.add_argument("num", type=int); x.add_argument("text", nargs="?"); x.set_defaults(f=cmd_done)
    for name, fn in (("claim", cmd_claim), ("review", cmd_review), ("finish", cmd_finish)):
        x = s.add_parser(name); x.add_argument("task"); x.set_defaults(f=fn)
    x = s.add_parser("block"); x.add_argument("task"); x.add_argument("on"); x.set_defaults(f=cmd_block)
    x = s.add_parser("handoff"); x.add_argument("task"); x.add_argument("--done", required=True)
    x.add_argument("--next", required=True); x.add_argument("--blocked"); x.add_argument("--artifacts"); x.set_defaults(f=cmd_handoff)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
