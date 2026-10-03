# Subagent log

Every delegation to a subagent, newest first. The policy (which model for which task shape)
is in [CLAUDE.md](../CLAUDE.md#subagents--use-them-sized-to-the-task). Use subagents for
tedious, well-scoped work; the lead agent re-verifies results before a milestone closes.

| date | milestone | model | task | outcome | lead re-verified |
|---|---|---|---|---|---|
| 2026-10-03 | I0.4 | sonnet | build `imitation/verify.py` + tests (registry, Wilson, eval/bench readers, I0.1/I0.2/I0.4 checks, SKIP stubs) | done; 12 tests, 3 PASS | yes: re-ran tests (12 passed) and verify (PASS ×3) |
| 2026-10-02 | I0.2 | sonnet | security review of lehome-challenge @a805ad2 (install code, import path, risky-pattern grep, ASSETS_ROOT, HF assets, uv.lock) | SAFE with caveats (CPU torch on Windows, cwd-based ASSETS_ROOT, `.cn` mirror URLs) | yes: read constant.py; checked 2 isaacsim wheel hashes against pypi.nvidia.com (identical) |
| 2026-10-02 | I0.2 | sonnet | security review of the LeHome IsaacLab fork @69f6fa5 vs upstream | SAFE: 2 commits, 3 benign changes; telemetry and registry on by default in the kit file | yes: grepped the fork lines and kit settings |
| 2026-10-02 | I0.1 | sonnet | read-only review of the merge resolution against both parents | no bugs; 4 nits (one fixed: comment on `max_episode_steps` with `base_env`) | yes: findings checked against the file |
| 2026-10-02 | I0.1 | sonnet | merge fix-ups: `isaac/tests` stub to the 139-D wrapper contract, `pytest.ini` testpaths + `isaac` marker, `HALF_FOLD_MAX_STEPS` single source, `tests/imitation/test_merge_invariants.py` | done; 116 fast pass, 7/7 invariants | yes: fast suite re-run 116 passed |
| 2026-10-02 | planning | sonnet (Explore) | map imitation ↔ MuJoCo coupling (file:line) | done | spot-checked key files |
| 2026-10-02 | planning | sonnet (Explore) | inspect unfetched remote branches via GitHub API | done; nothing newer than main for Isaac | confirmed with `git ls-remote` |
| 2026-10-02 | planning | opus (Plan) | research LeHome Windows support, cloth state restore, determinism; critique plan | done; findings folded into plan | sources cited; checked at install time |
