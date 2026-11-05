# How we work together on neat folding

Anyone on the team can join, along with their own Claude. This is how we share what we know without stepping on each
other. The idea is copied from OpenAI's "agent teams" talk on their Navier-Stokes run, scaled down from 10,000 agents.

**Every Claude session: read this file, then run `python teacher/harness/wf.py status`, before doing anything else.**

## The goal (does not change)
A policy that folds T-shirts and trousers **neatly** in sim, scored on garments it never trained on, by early December.
Then real SO-101 arms. How we split the work can change. The goal does not.

Start with `knowledge/00_start_here.md`.

## The rules, from the talk
1. **Each team owns one unique approach.** Before starting, read `teams/`. If someone already has your idea, pick a
   different one or join them. The talk's main warning: if everyone works on the same thing, it's expensive and wasted.
2. **Teams go deep, but the set of teams stays wide.** A team can test a hypothesis, challenge an assumption or run
   experiments, as long as it keeps its own approach.
3. **Tell the other teams.** Found something? Post it. Stuck? Post that. Want to know something another team knows? Ask.
4. **The steer can change.** New findings can start new teams, or end old ones. If one approach clearly wins, the
   others move onto it (a `steer` post).

## Added for us: nothing counts until it can be checked
That run could scale because Lean could check the proof. Our check is:
- The neatness score, on garments held out of training.
- A video a human has watched.

The neatness pass line isn't calibrated yet, so team `verifier` goes first. Claude writing "it works" isn't evidence.

## The board
Everything is plain files in this folder, one file per note, so people pushing at once don't conflict.

| Where | What | Who writes |
|---|---|---|
| `teams/<name>.md` | Owner, approach, what it's not doing, a log | That team |
| `posts/<time>-<team>-<title>.md` | Findings, questions, answers, blockers, steers. Never edited afterwards, post a new one instead | Anyone |
| `knowledge/` | What we treat as true | Promoted by Roy after checking |

```bash
python teacher/harness/wf.py status
python teacher/harness/wf.py team trousers --owner yourname --approach "why the teacher's trousers fail" --not-doing "T-shirts"
python teacher/harness/wf.py post trousers finding "waist grasp misses in 12/20" --evidence "results/trousers/run3.json + run3.mp4"
python teacher/harness/wf.py post trousers question "which pass line are we using?" --to verifier
python teacher/harness/wf.py sync     # commits only this folder, pulls everyone's notes, pushes
```

A `finding` must have evidence: a results path, a video or the numbers. Mark each claim as **measured** (we ran it),
**published** (from a paper) or **interpretation** (our reading). New posts start as `verified: no`. A different person
or team reruns the result or watches the video before it gets promoted.

## When a Claude session should sync
- At the start.
- After any result.
- Whenever it's blocked.
- Before stopping.

## Never commit (this repo is public)
- API keys, tokens, `.env` files, or account and billing details.
- The Coltraro cloth dataset (CC BY-NC-SA, kept locally for research only). Fitted parameters are fine.
- Big run outputs. Put the numbers and one short video in a post, and keep the rest on your own disk or bucket.

## Everyone pays for their own compute
Before any paid run, state the GPU, expected minutes and cost based on measured speeds. Stop your VM when you finish,
and post what it actually cost.
