#!/usr/bin/env python3
"""The shared board: who owns what, and what each team just found. Plain files in git, one file per note, so two
people pushing at once never conflict. Protocol: harness/README.md.

    python teacher/harness/wf.py status [-n 15]
    python teacher/harness/wf.py team NAME --owner WHO --approach "what makes this team different" [--not-doing "..."]
    python teacher/harness/wf.py post TEAM KIND "one-line title" [--evidence PATH_OR_NUMBERS] [--body "..."] [--to TEAM]
    python teacher/harness/wf.py sync
KIND: finding | question | answer | blocked | steer
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys

ROOT = os.environ.get("WF_HARNESS_ROOT") or os.path.dirname(os.path.abspath(__file__))
KINDS = ("finding", "question", "answer", "blocked", "steer")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "note"


def read_fields(path):
    """The `key: value` header of a note, up to the first blank line."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                break
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def listing(sub):
    d = os.path.join(ROOT, sub)
    return sorted(n for n in os.listdir(d) if n.endswith(".md")) if os.path.isdir(d) else []


def status(args):
    print("TEAMS")
    for n in listing("teams"):
        t = read_fields(os.path.join(ROOT, "teams", n))
        print(f"  {n[:-3]:<14} {t.get('state', '?'):<9} {t.get('owner', '?'):<12} {t.get('approach', '')}")
    print("\nLATEST POSTS")
    for n in listing("posts")[-args.n:]:
        p = read_fields(os.path.join(ROOT, "posts", n))
        to = f" -> {p['to']}" if p.get("to") else ""
        print(f"  {p.get('when', '?')} {p.get('team', '?'):<12} {p.get('kind', '?'):<8}{to} {p.get('title', '')}")
        print(f"      {n}")


def team(args):
    path = os.path.join(ROOT, "teams", slug(args.name) + ".md")
    if os.path.exists(path):
        sys.exit(f"team exists, edit it instead: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"owner: {args.owner}\nstate: active\napproach: {args.approach}\n\n")
        f.write(f"## Not doing\n{args.not_doing or '(say what you are leaving to other teams)'}\n\n## Log\n")
    print(path)


def post(args):
    if args.kind not in KINDS:
        sys.exit(f"kind must be one of {KINDS}")
    if not os.path.exists(os.path.join(ROOT, "teams", slug(args.team) + ".md")):
        sys.exit(f"no such team: {args.team} (run `wf.py status`)")
    if args.kind == "finding" and not args.evidence:
        sys.exit("a finding needs --evidence: a results path, a video, or the numbers. No evidence = post a question.")
    now = dt.datetime.now(dt.timezone.utc)
    path = os.path.join(ROOT, "posts", f"{now:%Y%m%d-%H%M%S}-{slug(args.team)}-{slug(args.title)}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"when: {now:%Y-%m-%d %H:%M}Z\nteam: {slug(args.team)}\nkind: {args.kind}\ntitle: {args.title}\n")
        if args.to:
            f.write(f"to: {slug(args.to)}\n")
        if args.evidence:
            f.write(f"evidence: {args.evidence}\n")
        f.write(f"verified: no\n\n{args.body or ''}\n")
    print(path)


def sync(_):
    """Commit only the board, take everyone else's, publish."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    rel = os.path.relpath(ROOT, top)
    subprocess.run(["git", "add", "--", rel], cwd=top, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", rel], cwd=top).returncode:
        subprocess.run(["git", "commit", "-m", "harness: board update", "--", rel], cwd=top, check=True)
    subprocess.run(["git", "pull", "--rebase", "--autostash"], cwd=top, check=True)
    subprocess.run(["git", "push"], cwd=top, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("-n", type=int, default=15)
    s.set_defaults(fn=status)
    t = sub.add_parser("team")
    t.add_argument("name")
    t.add_argument("--owner", required=True)
    t.add_argument("--approach", required=True)
    t.add_argument("--not-doing")
    t.set_defaults(fn=team)
    p = sub.add_parser("post")
    p.add_argument("team")
    p.add_argument("kind")
    p.add_argument("title")
    p.add_argument("--evidence")
    p.add_argument("--body")
    p.add_argument("--to")
    p.set_defaults(fn=post)
    sub.add_parser("sync").set_defaults(fn=sync)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
