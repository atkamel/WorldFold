"""Parallel rollouts: one subprocess per env (each with its own scripted teacher),
stepped in lockstep from the main process so a student policy can be evaluated
for all envs in one batched (GPU) forward pass.

Every rollout -- expert demos, student evaluation, DAgger -- goes through
`rollout()`, driven by a Controller that decides, at each replan point, which
chunk to execute and whether to ask the teacher for a label.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection as mp_connection
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from imitation.data.schema import ACTOR_PERTURB, ACTOR_STUDENT, ACTOR_TEACHER, Episode

ACTION_DIM = 12


# ---------------------------------------------------------------------------
# worker side

def _info_small(info, rig=None):
    extra = {"images": rig.render()} if rig else {}
    return extra | {"stage": int(info["stage"]), "fold_score": float(info["fold_score"]),
            "grasped": (bool(info["grasped"]["left_"]), bool(info["grasped"]["right_"])),
            "success": bool(info["success"]), "termination_reason": info["termination_reason"],
            "anchor_drift": float(info["anchor_drift"]), "move_distance": list(info["move_distance"])}


def _worker(pipe, env_kwargs):
    from imitation.tasks import HalfFoldEnv
    from imitation.teachers import ScriptedTeacher

    env_kwargs = dict(env_kwargs)
    render = env_kwargs.pop("render", False)
    env = HalfFoldEnv(**env_kwargs)
    teacher = ScriptedTeacher(env)
    rig, shadow = None, False
    if render:                     # Phase 4: camera images ride along in info["images"]
        from imitation.vision.render import CameraRig
        rig = CameraRig(env)
    while True:
        cmd, arg = pipe.recv()
        try:
            if cmd == "reset":
                seed, options, shadow = arg
                obs, info = env.reset(seed=seed, options=options)
                teacher.reset()
                if rig:
                    rig.reset(seed)
                meta = {"domain_params": dict(env.unwrapped._domain_params),
                        "start_corners": env._start[[0, 10, 110, 120]].round(5).tolist(),
                        "max_steps": env.unwrapped.max_episode_steps}
                pipe.send(("ok", (obs, _info_small(info, rig), meta)))
            elif cmd == "step":        # arg None -> the teacher acts
                if arg is not None and shadow:     # someone else acts: the labelling teacher shadows
                    teacher.observe()
                action = teacher.act() if arg is None else np.asarray(arg, dtype=np.float32)
                obs, r, term, trunc, info = env.step(action)
                pipe.send(("ok", (obs, float(r), bool(term), bool(trunc), _info_small(info, rig),
                                  np.asarray(action, dtype=np.float32))))
            elif cmd == "label":
                pipe.send(("ok", teacher.label_chunk(env, horizon=int(arg))))
            elif cmd == "resync":
                pipe.send(("ok", teacher.expert.resync()))
            elif cmd == "close":
                pipe.send(("ok", None))
                break
            else:
                raise ValueError(cmd)
        except Exception:
            pipe.send(("error", traceback.format_exc()))


# Workers are single-threaded physics; letting each one's BLAS grab every core
# oversubscribes the CPU several times over. Children inherit this at spawn.
_WORKER_THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


class EnvPool:
    def __init__(self, n, env_kwargs=None):
        self.render = bool((env_kwargs or {}).get("render"))
        ctx = mp.get_context("spawn")
        saved = {k: os.environ.get(k) for k in _WORKER_THREAD_ENV}
        os.environ.update(_WORKER_THREAD_ENV)
        self.pipes, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, env_kwargs or {}), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __len__(self):
        return len(self.pipes)

    def call(self, idx, cmd, args):
        """Send cmd to every env in idx (args aligned), then gather -- the envs run in parallel."""
        for i, a in zip(idx, args):
            self.pipes[i].send((cmd, a))
        out = []
        for i in idx:
            status, payload = self.pipes[i].recv()
            if status == "error":
                raise RuntimeError(f"env worker {i} failed:\n{payload}")
            out.append(payload)
        return out

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
                pipe.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# controllers

@dataclass
class Plan:
    """What an env does until its next replan. actions=None -> the teacher acts live."""
    actions: np.ndarray | None
    label: np.ndarray | None = None      # teacher chunk label to store at this step
    actor: int = ACTOR_STUDENT


class Controller:
    horizon = 1            # obs history length the controller needs
    needs_labels = False
    needs_images = False   # True -> plan() gets each env's current camera images
    label_horizon = 0
    source = "expert"

    def plan(self, slots, obs_hist, labels, rngs, images=None) -> list[Plan]:
        raise NotImplementedError


class ExpertController(Controller):
    """The scripted teacher drives the whole episode."""

    def plan(self, slots, obs_hist, labels, rngs, images=None):
        return [Plan(actions=None, actor=ACTOR_TEACHER) for _ in slots]


PREDICT_BUCKET = 16


def padded_predict(policy, obs_hist, images=None):
    """policy.predict on a batch padded to a multiple of PREDICT_BUCKET. GPU kernels
    change a row's floats (~1e-6) with the batch size, and the cloth sim amplifies that
    into different outcomes; which envs share a batch depends on worker timing, so
    without a fixed shape the same checkpoint scored 145 then 134/200 on id_hard."""
    n = len(obs_hist)
    size = -(-n // PREDICT_BUCKET) * PREDICT_BUCKET
    obs = np.concatenate([obs_hist, np.zeros((size - n,) + obs_hist.shape[1:], obs_hist.dtype)])
    if images is None:
        return policy.predict(obs)[:n]
    return policy.predict(obs, list(images) + [images[0]] * (size - n))[:n]


class PolicyController(Controller):
    """A student policy. With label=True the teacher labels every replan state;
    with beta > 0 (DAgger's mixture) the teacher's labelled chunk is executed
    instead of the student's with probability beta."""

    def __init__(self, policy, replan_every=8, beta=0.0, label=False, source="student"):
        self.policy = policy
        self.replan_every = replan_every
        self.horizon = policy.obs_horizon
        self.needs_labels = label or beta > 0
        self.label_horizon = policy.chunk
        self.beta = beta
        self.source = source
        self.needs_images = policy.needs_images

    def plan(self, slots, obs_hist, labels, rngs, images=None):
        # [B, K, A]: one batched call for all envs
        chunks = padded_predict(self.policy, obs_hist, images if self.needs_images else None)
        plans = []
        for j, slot in enumerate(slots):
            label = labels[j] if labels is not None else None
            use_teacher = label is not None and self.beta > 0 and rngs[slot].random() < self.beta
            chunk = label if use_teacher else chunks[j]
            plans.append(Plan(actions=chunk[:self.replan_every], label=label,
                              actor=ACTOR_TEACHER if use_teacher else ACTOR_STUDENT))
        return plans


@dataclass
class Perturbation:
    """k uniform-random actions starting at step t (recovery data / recovery eval)."""
    t: int
    k: int

    def active(self, t):
        return self.t <= t < self.t + self.k


# ---------------------------------------------------------------------------
# driver

@dataclass
class _Slot:
    seed: int
    meta: dict
    hist: deque
    perturb: Perturbation | None
    info: dict
    rows: dict = field(default_factory=lambda: {k: [] for k in
                                                ("obs", "actions", "rewards", "stage", "fold_score", "grasped", "actor",
                                                 "terminated", "truncated")})
    img: dict | None = None          # current camera images (render=True pools)
    images: list = field(default_factory=list)
    label_steps: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    queue: deque = field(default_factory=deque)
    queue_actor: int = ACTOR_STUDENT
    teacher_live: bool = False       # the teacher is acting step by step
    teacher_stale: bool = False      # ...but something else acted since, so resync first

    @property
    def t(self):
        return len(self.rows["actions"])


def rollout(pool: EnvPool, seeds, controller: Controller, reset_options=None, perturb_fn=None,
            meta_extra=None, progress=None, on_done=None, max_wait=0.01) -> list[Episode]:
    """Run one episode per seed across the pool; returns Episodes in seed order.

    Event-driven: each env gets its next command the moment its last one returns,
    so one env stalled in a slow expert IK solve never holds up the others. Envs
    that need a new chunk wait (at most `max_wait` s, or until nothing else is in
    flight) so their policy calls go out as one batch.

    reset_options(seed) -> dict of env reset options (e.g. a harder cloth pose).
    perturb_fn(seed, rng) -> Perturbation or None.
    on_done(ep) is called as each episode finishes, e.g. to persist it at once.
    """
    seeds = list(seeds)
    todo = deque(range(len(seeds)))
    active: dict[int, _Slot] = {}
    order, done, rngs = {}, {}, {}
    inflight: dict[int, str] = {}          # env -> command awaiting its reply
    step_actor: dict[int, int] = {}
    awaiting: dict[int, np.ndarray | None] = {}   # env -> teacher label (or None), waiting for a plan
    waiting_since = None

    def send(i, cmd, arg=None):
        pool.pipes[i].send((cmd, arg))
        inflight[i] = cmd

    def start(i):
        k = todo.popleft()
        seed = int(seeds[k])
        order[i] = k
        rngs[i] = np.random.default_rng([seed, 7919])
        send(i, "reset", (seed, reset_options(seed) if reset_options else None, controller.needs_labels))

    def advance(i):
        """Send env i its next command (or park it until the next batched plan)."""
        s = active[i]
        if s.perturb is not None and s.perturb.active(s.t):
            s.queue.clear()
            s.teacher_stale = True
            step_actor[i] = ACTOR_PERTURB
            send(i, "step", rngs[i].uniform(-1, 1, ACTION_DIM).astype(np.float32))
        elif s.teacher_live:
            if s.teacher_stale:
                s.teacher_stale = False
                send(i, "resync")
            else:
                step_actor[i] = ACTOR_TEACHER
                send(i, "step", None)
        elif s.queue:
            step_actor[i] = s.queue_actor
            send(i, "step", s.queue.popleft())
        elif controller.needs_labels:
            send(i, "label", controller.label_horizon)
        else:
            awaiting[i] = None

    def plan_batch():
        need = list(awaiting)
        labels = [awaiting[i] for i in need] if controller.needs_labels else None
        obs_hist = np.stack([np.stack(active[i].hist) for i in need]).astype(np.float32)
        images = [active[i].img for i in need] if controller.needs_images else None
        plans = controller.plan(need, obs_hist, labels, rngs, images=images)
        awaiting.clear()
        for i, p in zip(need, plans):
            s = active[i]
            if p.label is not None:
                s.label_steps.append(s.t)
                s.labels.append(np.asarray(p.label, dtype=np.float32))
            if p.actions is None:
                s.teacher_live = True
            else:
                s.queue.extend(np.asarray(p.actions, dtype=np.float32))
                s.queue_actor = p.actor
            advance(i)

    for i in range(min(len(pool), len(seeds))):
        start(i)

    while inflight or awaiting:
        if awaiting and (not inflight or (waiting_since is not None
                                          and time.perf_counter() - waiting_since > max_wait)):
            plan_batch()
            waiting_since = None
            continue
        timeout = None if not awaiting else max(0.0, max_wait - (time.perf_counter() - waiting_since))
        ready = mp_connection.wait([pool.pipes[i] for i in inflight], timeout=timeout)
        for conn in ready:
            i = pool.pipes.index(conn)
            cmd = inflight.pop(i)
            status, payload = conn.recv()
            if status == "error":
                raise RuntimeError(f"env worker {i} failed during {cmd}:\n{payload}")
            if cmd == "reset":
                obs, info, meta = payload
                img = info.pop("images", None)
                seed = int(seeds[order[i]])
                active[i] = _Slot(seed=seed, meta=meta, info=info,
                                  hist=deque([obs] * controller.horizon, maxlen=controller.horizon),
                                  perturb=perturb_fn(seed, rngs[i]) if perturb_fn else None, img=img)
                advance(i)
            elif cmd == "resync":
                advance(i)
            elif cmd == "label":
                awaiting[i] = payload
            elif cmd == "step":
                obs, r, term, trunc, info, executed = payload
                s = active[i]
                img = info.pop("images", None)
                if s.img is not None:                 # images of the state the action was chosen from
                    s.images.append(s.img)
                s.img = img
                # A solver blow-up ends the episode but is not an MDP terminal (imitation.md 4).
                unstable = term and info["termination_reason"] == "unstable"
                for key, val in (("obs", s.hist[-1]), ("actions", executed), ("rewards", r),
                                 ("stage", info["stage"]), ("fold_score", info["fold_score"]),
                                 ("grasped", info["grasped"]), ("actor", step_actor[i]),
                                 ("terminated", term and not unstable), ("truncated", trunc or unstable)):
                    s.rows[key].append(val)
                s.hist.append(obs)
                s.info = info
                if term or trunc:
                    done[order[i]] = _finish(s, obs, controller, meta_extra)
                    if on_done:
                        on_done(done[order[i]])
                    if progress:
                        progress(len(done), len(seeds), done[order[i]])
                    del active[i]
                    if todo:
                        start(i)
                else:
                    advance(i)
            if i in awaiting and waiting_since is None:
                waiting_since = time.perf_counter()
    return [done[k] for k in range(len(seeds))]


def _finish(s: _Slot, final_obs, controller, meta_extra) -> Episode:
    info = s.info
    meta = {"seed": s.seed, "source": controller.source, "success": bool(info["success"]),
            "termination_reason": info["termination_reason"] or "truncated",
            "final_fold_score": float(info["fold_score"]), "final_stage": int(info["stage"]),
            "final_move_distance": [float(d) for d in info["move_distance"]],
            "final_anchor_drift": float(info["anchor_drift"]),
            "perturb": None if s.perturb is None else [s.perturb.t, s.perturb.k],
            **s.meta, **(meta_extra or {})}
    rows = s.rows
    labels = np.stack(s.labels).astype(np.float32) if s.labels else np.zeros((0, 0, 0), dtype=np.float32)
    images = {cam: np.stack([im[cam] for im in s.images]) for cam in s.images[0]} if s.images else None
    return Episode(obs=np.stack(rows["obs"]).astype(np.float32),
                   actions=np.stack(rows["actions"]).astype(np.float32),
                   rewards=np.asarray(rows["rewards"], dtype=np.float32),
                   stage=np.asarray(rows["stage"], dtype=np.int8),
                   fold_score=np.asarray(rows["fold_score"], dtype=np.float32),
                   grasped=np.asarray(rows["grasped"], dtype=bool),
                   actor=np.asarray(rows["actor"], dtype=np.int8),
                   final_obs=np.asarray(final_obs, dtype=np.float32),
                   terminated=np.asarray(rows["terminated"], dtype=bool),
                   truncated=np.asarray(rows["truncated"], dtype=bool),
                   discount=np.where(rows["terminated"], 0.0, 1.0).astype(np.float32), meta=meta,
                   label_steps=np.asarray(s.label_steps, dtype=np.int32), labels=labels, images=images)
