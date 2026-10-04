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

def _info_small(info, images=None):
    extra = {"images": images} if images is not None else {}
    if "_timing" in info:          # (teacher s, physics s, render s) of this step, for profiling (M5c.1)
        extra["timing"] = info["_timing"]
    return extra | {"stage": int(info["stage"]), "fold_score": float(info["fold_score"]),
            "grasped": (bool(info["grasped"]["left_"]), bool(info["grasped"]["right_"])),
            "success": bool(info["success"]), "termination_reason": info["termination_reason"],
            "anchor_drift": float(info["anchor_drift"]), "move_distance": list(info["move_distance"])}


def _split(obs):
    """Dict observation (cameras) -> (139-D state, images or None)."""
    if isinstance(obs, dict):
        return obs["state"], {k: v for k, v in obs.items() if k != "state"}
    return obs, None


def _finite_or(obs, last):
    """An Isaac `unstable` ending (non-finite particles) can return a NaN observation, which the dataset
    validator rejects; keep the last finite one instead (the episode is still stored as unstable)."""
    state = obs["state"] if isinstance(obs, dict) else obs
    if last is None or np.all(np.isfinite(state)):
        return obs
    return last


class _LazyTeacher:
    """Builds the labelling teacher on first use: a policy rollout never needs the scripted expert,
    and on Isaac the expert is a separate milestone (I2.1). `factory(env)` replaces the default teachers
    (tests)."""

    def __init__(self, env, teacher_ckpt, factory=None):
        self._env, self._ckpt, self._t, self._factory = env, teacher_ckpt, None, factory

    @property
    def built(self):
        return self._t is not None

    def get(self):
        # built while the env sits at an episode start (reset, or the step that starts acting) and reset
        # at once, so a lazily built teacher starts exactly like the eagerly built one used to
        if self._t is None:
            if self._factory is not None:
                self._t = self._factory(self._env)
            elif self._ckpt:
                from imitation.teachers.policy import PolicyTeacher
                self._t = PolicyTeacher(self._env, self._ckpt)
            else:
                from imitation.teachers.scripted import ScriptedTeacher
                self._t = ScriptedTeacher(self._env)
            self._t.reset()
        return self._t


class _SlotServer:
    """One env's side of the driver protocol (reset / step / label / resync); `close` is the worker's."""

    def __init__(self, env, teacher_ckpt=None, isaac=False, teacher_factory=None):
        self.env, self.teacher_ckpt, self.isaac = env, teacher_ckpt, isaac
        self.lazy = _LazyTeacher(env, teacher_ckpt, teacher_factory)
        self.shadow, self.last_obs = False, None

    def handle(self, cmd, arg):
        """-> (status, payload, episode_over): episode_over is True after a terminal or truncated step."""
        env, lazy = self.env, self.lazy
        try:
            if cmd == "reset":
                seed, options, self.shadow = arg
                obs, info = env.reset(seed=seed, options=options)
                obs, images = _split(obs)
                self.last_obs = obs
                # MuJoCo builds the teacher at the first episode start, exactly as before Phase I; Isaac defers
                # it until it is needed (policy rollouts never need the scripted expert)
                if self.shadow or self.teacher_ckpt or not self.isaac:
                    lazy.get()
                if lazy.built:
                    lazy.get().reset()
                    lazy.get().see(obs)
                meta = {"domain_params": dict(env.unwrapped._domain_params),
                        "start_corners": env._start[[0, 10, 110, 120]].round(5).tolist(),
                        "max_steps": env.unwrapped.max_episode_steps}
                return "ok", (obs, _info_small(info, images), meta), False
            if cmd == "step":          # arg None -> the teacher acts
                t0 = time.perf_counter()
                teacher = lazy.get() if (arg is None or self.shadow or lazy.built) else None
                if arg is not None and self.shadow:    # someone else acts: the labelling teacher shadows
                    teacher.observe()
                action = teacher.act() if arg is None else np.asarray(arg, dtype=np.float32)
                t1 = time.perf_counter()
                obs, r, term, trunc, info = env.step(action)
                t2 = time.perf_counter()
                obs, images = _split(obs)
                obs = self.last_obs = _finite_or(obs, self.last_obs)
                if teacher is not None:
                    teacher.see(obs)
                info = dict(info, _timing=(t1 - t0, t2 - t1 - env.last_render_s, env.last_render_s))
                return "ok", (obs, float(r), bool(term), bool(trunc), _info_small(info, images),
                              np.asarray(action, dtype=np.float32)), bool(term or trunc)
            if cmd == "label":         # (labels, seconds): the expert's look-ahead is real simulation (M5c.1)
                t0 = time.perf_counter()
                labels = lazy.get().label_chunk(env, horizon=int(arg))
                return "ok", (labels, time.perf_counter() - t0), False
            if cmd == "resync":
                teacher = lazy.get()
                return "ok", (teacher.expert.resync() if hasattr(teacher, "expert") else {}), False
            raise ValueError(cmd)
        except Exception:
            return "error", traceback.format_exc(), False


def _env_setup(env_kwargs):
    env_kwargs = dict(env_kwargs)
    backend = env_kwargs.pop("backend", "mujoco")
    render = env_kwargs.pop("render", False)
    cameras = env_kwargs.pop("cameras", None)
    teacher_ckpt = env_kwargs.pop("teacher", None)
    if render:                     # Phase 4: the env returns {"state", cameras...} (M4.1)
        env_kwargs.update(obs_mode="dict", cameras=dict(cameras) if cameras else None)
    return backend, teacher_ckpt, env_kwargs


def _worker(pipe, env_kwargs):
    from imitation.tasks import make_env

    backend, teacher_ckpt, env_kwargs = _env_setup(env_kwargs)
    from imitation.tasks.half_fold import is_isaac
    isaac = is_isaac(backend)
    try:
        env = make_env(backend=backend, **env_kwargs)
    except Exception:
        pipe.send(("error", traceback.format_exc()))
        os._exit(1) if isaac else None
        return
    pipe.send(("ready", None))
    server = _SlotServer(env, teacher_ckpt, isaac)
    while True:
        try:
            if isaac:              # tick Kit while idle (e.g. the parent is training) or its hang detector aborts
                from imitation.isaac_runtime import KIT_TICK_S
                while not pipe.poll(KIT_TICK_S):
                    env.unwrapped.keep_alive()
            cmd, arg = pipe.recv()
        except (EOFError, OSError):  # the parent is gone: don't leave a Kit app holding GPU memory
            if isaac:
                os._exit(0)
            return
        if cmd == "close":
            pipe.send(("ok", None))
            if isaac:              # leaving the interpreter with Kit up hangs at shutdown (isaac/README.md)
                pipe.close()
                os._exit(0)
            break
        status, payload, _ = server.handle(cmd, arg)
        pipe.send((status, payload))


def _slot_loop(lockstep, i, pipe, server, closed):
    """Slot thread of a multi-env worker: today's command loop, with the scene touched only under the baton
    and the slot marked active while its episode is in flight (imitation/lockstep.py)."""
    while True:
        try:
            cmd, arg = pipe.recv()
        except (EOFError, OSError):
            cmd, arg = "close", None
        with lockstep.baton:
            if cmd == "close":
                lockstep.set_active(i, False)
                status, payload, over = "ok", None, True
            else:
                if cmd == "reset":
                    lockstep.set_active(i, True)
                status, payload, over = server.handle(cmd, arg)
                base = server.env.unwrapped
                if over and hasattr(base, "park") and base._failed():
                    base.park()        # an exploded cloth must not drift while the other copies step
                if over or status == "error":
                    lockstep.set_active(i, False)
        try:                       # outside the baton: a large reply may block until the driver reads it
            pipe.send((status, payload))
        except (EOFError, OSError):
            cmd = "close"
        if cmd == "close":
            with lockstep.baton:
                closed.add(i)
                if len(closed) == closed.total:
                    lockstep.close()
            return


class _Closed(set):
    def __init__(self, total):
        super().__init__()
        self.total = total


def _serve_slots(pipes, batch, envs, teacher_ckpt=None, isaac=False, teacher_factory=None, tick_s=None):
    """Serve len(envs) slots of one batch, one pipe each, on slot threads; runs the lockstep scheduler on the
    calling thread until every slot is closed. Sends `ready` on every pipe first."""
    import threading

    from imitation.lockstep import Lockstep
    if tick_s is None:
        from imitation.isaac_runtime import KIT_TICK_S
        tick_s = KIT_TICK_S
    lockstep = Lockstep(batch.advance, keep_alive=batch.keep_alive, tick_s=tick_s)
    batch.scheduler = lockstep
    closed = _Closed(len(envs))
    threads = [threading.Thread(target=_slot_loop, daemon=True,
                                args=(lockstep, i, pipe, _SlotServer(env, teacher_ckpt, isaac, teacher_factory),
                                      closed))
               for i, (pipe, env) in enumerate(zip(pipes, envs))]
    for pipe in pipes:
        pipe.send(("ready", None))
    for t in threads:
        t.start()
    lockstep.run()
    for t in threads:
        t.join(timeout=5)
    return lockstep


def _worker_multi(pipes, env_kwargs, n_envs):
    """A worker hosting n_envs sub-envs of one Isaac batch (milestone V), one pipe per sub-env."""
    from imitation.tasks import make_env_batch

    backend, teacher_ckpt, env_kwargs = _env_setup(env_kwargs)
    try:
        batch, envs = make_env_batch(backend=backend, n=n_envs, **env_kwargs)
    except Exception:
        for pipe in pipes:
            pipe.send(("error", traceback.format_exc()))
        os._exit(1)
    try:
        _serve_slots(pipes, batch, envs, teacher_ckpt, isaac=True)
    finally:
        os._exit(0)        # leaving the interpreter with Kit up hangs at shutdown (isaac/README.md)


def envs_per_proc(backend) -> int:
    """Sub-envs per worker process: WORLDFOLD_ISAAC_ENVS_PER_PROC for the Isaac weld profile (milestone V),
    1 otherwise."""
    from imitation.tasks.half_fold import is_isaac
    if not backend or not is_isaac(backend):
        return 1
    b = int(os.environ.get("WORLDFOLD_ISAAC_ENVS_PER_PROC", "1"))
    if b < 1:
        raise ValueError(f"WORLDFOLD_ISAAC_ENVS_PER_PROC must be >= 1, got {b}")
    if b > 1 and backend != "isaac_weld":
        raise ValueError(f"WORLDFOLD_ISAAC_ENVS_PER_PROC > 1 needs the GPU weld profile (isaac_weld), not {backend}")
    return b


# Workers are single-threaded physics; letting each one's BLAS grab every core
# oversubscribes the CPU several times over. Children inherit this at spawn.
_WORKER_THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


class EnvPool:
    def __init__(self, n, env_kwargs=None):
        """n worker processes; len(pool) = n * envs_per_proc(backend) logical slots, each with its own pipe.

        env_kwargs: HalfFoldEnv kwargs plus `render` (cameras on; images reach image
        controllers), `cameras` ({name: size}) and `teacher` (a state-policy checkpoint
        that replaces the scripted expert as the labelling teacher)."""
        self.render = bool((env_kwargs or {}).get("render"))
        self.cameras = (env_kwargs or {}).get("cameras")
        ctx = mp.get_context("spawn")
        saved = {k: os.environ.get(k) for k in _WORKER_THREAD_ENV}
        os.environ.update(_WORKER_THREAD_ENV)
        self.pipes, self.procs = [], []
        # Isaac workers start one at a time: several Kit apps compiling shaders at once can hang
        from imitation.tasks.half_fold import is_isaac
        backend = (env_kwargs or {}).get("backend")
        serial = backend and is_isaac(backend)
        self.envs_per_proc = B = envs_per_proc(backend)
        try:
            for _ in range(n):
                ends = [ctx.Pipe() for _ in range(B)]
                if B == 1:
                    p = ctx.Process(target=_worker, args=(ends[0][1], env_kwargs or {}), daemon=True)
                else:
                    p = ctx.Process(target=_worker_multi, args=([b for _, b in ends], env_kwargs or {}, B),
                                    daemon=True)
                p.start()
                for a, b in ends:
                    b.close()   # the child owns its end; a worker that dies now gives EOF instead of a hang
                    self.pipes.append(a)
                self.procs.append(p)
                if serial:
                    for k in range(len(self.pipes) - B, len(self.pipes)):
                        self._ready(k)
            if not serial:
                for i in range(len(self.pipes)):
                    self._ready(i)
        except Exception:
            self.close()
            raise
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _ready(self, i):
        status, payload = self.pipes[i].recv()
        if status != "ready":
            raise RuntimeError(f"env worker {i} failed to start:\n{payload}")

    def __len__(self):
        return len(self.pipes)

    def call(self, idx, cmd, args):
        """Send cmd to every env in idx (args aligned), then gather -- the envs run in parallel.
        Not for physics commands with several envs per process (milestone V): a lockstep slot that finishes first
        sits idle mid-episode and holds its neighbours' remaining steps, so waiting in slot order can block forever.
        Drive those through rollout(), which always sends an idle slot its next command."""
        if getattr(self, "envs_per_proc", 1) > 1 and cmd in ("reset", "step", "label"):
            raise NotImplementedError("EnvPool.call can't drive physics with several envs per process; use rollout()")
        for i, a in zip(idx, args):
            self.pipes[i].send((cmd, a))
        out = []
        for i in idx:
            status, payload = self.pipes[i].recv()
            if status == "error":
                raise RuntimeError(f"env worker {i} failed:\n{payload}")
            out.append(payload)
        return out

    def close(self, timeout=60.0):
        # every slot is told first: in a multi-env worker a slot still in a physics step only finishes once its
        # neighbours have stopped (closed slots never block), so a send/recv per pipe in turn could wait forever
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        deadline = time.monotonic() + timeout
        for pipe in self.pipes:
            try:
                while pipe.poll(max(0.0, deadline - time.monotonic())):
                    if pipe.recv() == ("ok", None):     # the close ack (an in-flight reply may come first)
                        break
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():      # a hung worker (e.g. Kit at shutdown) must not outlive the pool
                p.terminate()
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
    takeover: int = 0                    # >0: the teacher takes over for this many real steps, and what it does
                                         # becomes the label of this state (Isaac DAgger labels, Phase I I2.2)


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
    instead of the student's with probability beta.

    `teacher_policy`: a *policy* teacher (distillation) labels here, in the main process,
    in one batched GPU call from the observation histories already held for the student.
    The workers then never load or run it (it cost each CPU core ~77 ms per label for
    the diffusion teacher). The scripted expert still labels in the workers, since it
    needs the live sim."""

    def __init__(self, policy, replan_every=8, beta=0.0, label=False, source="student", teacher_policy=None,
                 takeover=0.0):
        """takeover: probability that the scripted teacher takes over at a replan point; its executed chunk is
        the label (no look-ahead, so no sim snapshot -- Isaac has none). Used instead of label/beta."""
        self.policy = policy
        self.takeover = takeover
        self.teacher_policy = teacher_policy
        self.replan_every = replan_every
        self.horizon = max(policy.obs_horizon, teacher_policy.obs_horizon if teacher_policy else 0)
        self.needs_labels = (label or beta > 0) and teacher_policy is None      # worker-side labels
        self.gpu_labels = (label or beta > 0) and teacher_policy is not None
        self.label_horizon = policy.chunk
        self.beta = beta
        self.source = source
        self.needs_images = policy.needs_images

    def plan(self, slots, obs_hist, labels, rngs, images=None):
        # [B, K, A]: one batched call for all envs
        chunks = padded_predict(self.policy, obs_hist[:, -self.policy.obs_horizon:],
                                images if self.needs_images else None)
        if self.gpu_labels:
            t = self.teacher_policy
            labels = list(padded_predict(t, obs_hist[:, -t.obs_horizon:])[:, :self.label_horizon])
        plans = []
        for j, slot in enumerate(slots):
            if self.takeover > 0 and rngs[slot].random() < self.takeover:
                plans.append(Plan(actions=None, actor=ACTOR_TEACHER, takeover=self.label_horizon))
                continue
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
    takeover_left: int = 0           # takeover label in progress: teacher steps still to record
    takeover_t: int = 0              # the step whose label the takeover is
    takeover_acts: list = field(default_factory=list)

    @property
    def t(self):
        return len(self.rows["actions"])


def rollout(pool: EnvPool, seeds, controller: Controller, reset_options=None, perturb_fn=None,
            meta_extra=None, progress=None, on_done=None, max_wait=0.01, stats=None) -> list[Episode]:
    """Run one episode per seed across the pool (see `_rollout`). Holds the shared CPU slot
    for the duration when `IMITATION_CPU_SLOT` is set (`imitation.cpu_slot`, M5c.3)."""
    from imitation.cpu_slot import cpu_slot
    with cpu_slot() as waited:
        if stats is not None:
            stats["slot_wait_s"] = stats.get("slot_wait_s", 0.0) + waited
        return _rollout(pool, seeds, controller, reset_options, perturb_fn, meta_extra, progress, on_done,
                        max_wait, stats)


def _rollout(pool, seeds, controller, reset_options=None, perturb_fn=None, meta_extra=None, progress=None,
             on_done=None, max_wait=0.01, stats=None) -> list[Episode]:
    """Run one episode per seed across the pool; returns Episodes in seed order.

    Event-driven: each env gets its next command the moment its last one returns,
    so one env stalled in a slow expert IK solve never holds up the others. Envs
    that need a new chunk wait (at most `max_wait` s, or until nothing else is in
    flight) so their policy calls go out as one batch.

    reset_options(seed) -> dict of env reset options (e.g. a harder cloth pose).
    perturb_fn(seed, rng) -> Perturbation or None.
    on_done(ep) is called as each episode finishes, e.g. to persist it at once.
    stats: a dict to accumulate a time breakdown into (M5c.1 profiling): worker-side
    teacher / physics / render seconds (summed over workers), main-process plan seconds,
    main-process wait seconds, wall seconds, and step count.
    """
    seeds = list(seeds)
    t_start = time.perf_counter()
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
        controller.slot_seeds = {i: active[i].seed for i in need}   # for per-seed controllers
        images = [active[i].img for i in need] if controller.needs_images else None
        tp = time.perf_counter()
        plans = controller.plan(need, obs_hist, labels, rngs, images=images)
        if stats is not None:
            stats["plan_s"] = stats.get("plan_s", 0.0) + time.perf_counter() - tp
        awaiting.clear()
        for i, p in zip(need, plans):
            s = active[i]
            if p.label is not None:
                s.label_steps.append(s.t)
                s.labels.append(np.asarray(p.label, dtype=np.float32))
            if p.actions is None:
                s.teacher_live = True
                if p.takeover:                 # the student was driving: resync, then record K real steps
                    s.teacher_stale = True
                    s.takeover_left, s.takeover_t, s.takeover_acts = p.takeover, s.t, []
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
        tw = time.perf_counter()
        ready = mp_connection.wait([pool.pipes[i] for i in inflight], timeout=timeout)
        if stats is not None:
            stats["wait_s"] = stats.get("wait_s", 0.0) + time.perf_counter() - tw
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
                awaiting[i], label_s = payload
                if stats is not None:
                    stats["label_s"] = stats.get("label_s", 0.0) + label_s
            elif cmd == "step":
                obs, r, term, trunc, info, executed = payload
                timing = info.pop("timing", None)
                if stats is not None and timing is not None:
                    for key, v in zip(("teacher_s", "physics_s", "render_s"), timing):
                        stats[key] = stats.get(key, 0.0) + v
                    stats["steps"] = stats.get("steps", 0) + 1
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
                if s.takeover_left > 0 and step_actor[i] == ACTOR_TEACHER:
                    s.takeover_acts.append(np.asarray(executed, dtype=np.float32))
                    s.takeover_left -= 1
                    if s.takeover_left == 0 or term or trunc:
                        s.label_steps.append(s.takeover_t)
                        s.labels.append(_pad_chunk(s.takeover_acts, controller.label_horizon))
                        s.takeover_left, s.teacher_live = 0, False
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
    if stats is not None:
        stats["wall_s"] = stats.get("wall_s", 0.0) + time.perf_counter() - t_start
    return [done[k] for k in range(len(seeds))]


def _pad_chunk(actions, horizon):
    """A chunk label from executed actions; an episode that ends inside it is padded as ScriptedTeacher.label_chunk
    pads: joints still, gripper commands kept."""
    chunk = np.zeros((horizon, len(actions[0])), dtype=np.float32)
    chunk[:len(actions)] = actions
    if len(actions) < horizon:
        chunk[len(actions):, [5, 11]] = actions[-1][[5, 11]]
    return chunk


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
