# How we work together on neat folding

Anyone on the team can join, along with their own Claude. Nothing here is assigned. Read where things stand, pick what
you think moves it, and tell the others.

**Every Claude session: read this file, then `knowledge/00_start_here.md`, then run
`python teacher/harness/wf.py status`, before doing anything else.**

## The goal (does not change)
A policy that folds T-shirts and trousers **neatly** in sim, scored on garments it never trained on, by early December.
Then real SO-101 arms.

## What this harness is
It's a shared folder in this repo that each person's Claude reads and writes. It's how separate Claude sessions, on
separate laptops, know what the others found.

**It is not a live multi-agent system.** There's no manager agent and no chat between Claudes. Each person starts their
own Claude, and notes only move when someone runs `sync`.

**Where the idea came from:** OpenAI's "agent teams" talk on their Navier-Stokes run (about 10,000 agents, their own
claim). We took four rules from it:
1. **Different people take different approaches.** If everyone works on the same thing, it's expensive and wasted. Check
   `teams/` before starting.
2. **Go deep, but keep the set of approaches wide.**
3. **Tell the others.** Found something, stuck, or need something another person knows: post it.
4. **Steer on findings.** New results can start new lines or end old ones. If one clearly wins, everyone moves onto it.
   The goal stays fixed.

**What we added:** that run could scale because Lean could check the output. Our checks are the neatness score on held-out
garments, and a video a human has watched. Claude writing "it works" isn't evidence.

## The board
Everything is plain files, one file per note, so people pushing at once don't conflict.

| Where | What | Who writes |
|---|---|---|
| `teams/<name>.md` | Who owns an approach, what it isn't doing, a log | That person |
| `posts/<time>-<team>-<title>.md` | Findings, questions, answers, blockers, steers. Post a new one instead of editing | Anyone |
| `knowledge/` | What we treat as true | Promoted by Roy after checking |

```bash
python teacher/harness/wf.py status
python teacher/harness/wf.py team trousers --owner yourname --approach "why the teacher's trousers fail" --not-doing "T-shirts"
python teacher/harness/wf.py post trousers finding "waist grasp misses in 12/20" --evidence "results/trousers/run3.json + run3.mp4"
python teacher/harness/wf.py post neat-rl question "which pass line are we using?"
python teacher/harness/wf.py sync     # commits only this folder, pulls everyone's notes, pushes
```

`neat-rl` is the shared line for stage 1. Start your own team only when your approach is really different.

A `finding` must have evidence: a results path, a video or the numbers. Mark each claim as **measured** (we ran it),
**published** (from a paper) or **interpretation** (our reading). New posts start as `verified: no`. Someone else reruns
the result or watches the video before it gets promoted.

## When a Claude session should sync
At the start, after any result, whenever it's blocked, and before stopping. You need push access to this repo: ask Roy.

## Never commit (this repo is public)
- API keys, tokens, `.env` files, or account and billing details.
- The Coltraro cloth dataset (CC BY-NC-SA, kept locally for research only). Fitted parameters are fine.
- Big run outputs. Put the numbers and one short video in a post, and keep the rest on your own disk or bucket.

## Everyone pays for their own compute
Before any paid run, state the GPU, expected minutes and cost based on measured speeds. Stop your VM when you finish, and
post what it actually cost.
