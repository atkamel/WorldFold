"""Lockstep scheduler for B envs that share one physics step (milestone V, vectorised Isaac env). No Isaac imports.

A batch (e.g. isaac.isaac_env.IsaacClothFoldBatch) owns B sub-envs whose physics advances together in one
`batch.advance()`. A rollout worker serves each sub-env from its own *slot thread* (one per driver pipe) running
the ordinary synchronous env code; each control step inside it calls `batch.sync(i)`. With a `Lockstep`
attached, `sync` parks the slot thread, and the scheduler -- on the main thread -- calls `advance()` once every
*active* slot (one with an episode in flight) is parked. A slot thread holds the *baton* (the condition's lock)
while it runs env code, so one thread touches the scene at a time; calls that must run on the main thread (Kit,
USD writes) go through `call_main`. Design and deadlock argument: docs/superpowers/specs/2026-10-04-isaac-vec-design.md.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class BatchBase:
    """What a batch shares with the scheduler: `advance()` steps every copy once; `sync(i)` is sub-env i's share
    of a step; `on_main(fn)` runs fn on the scheduler's (main) thread. Without a scheduler both run at once."""

    scheduler = None

    def advance(self):
        raise NotImplementedError

    def keep_alive(self):
        """Tick whatever needs ticking while no physics runs (Kit's hang detector)."""

    def sync(self, i):
        if self.scheduler is None:
            self.advance()
        else:
            self.scheduler.sync(i)

    def on_main(self, fn, *args, **kwargs):
        if self.scheduler is None:
            return fn(*args, **kwargs)
        return self.scheduler.call_main(fn, *args, **kwargs)


class Lockstep:
    def __init__(self, advance, keep_alive=None, tick_s=30.0):
        self.baton = threading.Condition()
        self._advance = advance
        self._keep_alive = keep_alive
        self._tick_s = tick_s
        self.active: set[int] = set()
        self.parked: set[int] = set()
        self.gen = 0
        self.steps = 0
        self.advance_s = 0.0          # time inside advance() (the shared physics step), for profiling
        self._failed: dict[int, BaseException] = {}     # generation -> the exception its advance raised
        self._calls: deque = deque()
        self._closed = False
        self._main = threading.current_thread()   # run() must be called from this thread

    # ---- slot side (baton held) ----
    def set_active(self, i, flag):
        (self.active.add if flag else self.active.discard)(i)
        self.baton.notify_all()

    def sync(self, i):
        """Park slot i until the next global step has run. Called with the baton held."""
        g = self.gen
        self.parked.add(i)
        self.baton.notify_all()
        while self.gen == g and not self._closed:
            self.baton.wait()
        if g in self._failed:
            raise RuntimeError(f"the batch physics step failed: {self._failed[g]!r}") from self._failed[g]
        if self.gen == g:
            raise RuntimeError("the lockstep scheduler closed while a slot was waiting for physics")

    def call_main(self, fn, *args, **kwargs):
        """Run fn on the scheduler's thread and return its result. Called with the baton held."""
        if self._main is None or threading.current_thread() is self._main:
            return fn(*args, **kwargs)
        box = {}
        self._calls.append((fn, args, kwargs, box))
        self.baton.notify_all()
        while "done" not in box:
            self.baton.wait()
        if "error" in box:
            raise box["error"]
        return box["value"]

    # ---- scheduler side ----
    def ready(self):
        return bool(self.parked) and self.active <= self.parked

    def run(self):
        """Serve steps and main-thread calls until close(). Runs on the thread that owns the scene."""
        self._main = threading.current_thread()
        last = time.monotonic()
        with self.baton:
            while True:
                if self._calls:
                    fn, args, kwargs, box = self._calls.popleft()
                    try:
                        box["value"] = fn(*args, **kwargs)
                    except BaseException as e:      # noqa: BLE001 -- handed to the calling slot
                        box["error"] = e
                    box["done"] = True
                    self.baton.notify_all()
                    continue
                if self.ready():
                    t0 = time.perf_counter()
                    try:
                        self._advance()
                    except BaseException as e:      # noqa: BLE001 -- re-raised in every parked slot
                        self._failed[self.gen] = e
                    self.advance_s += time.perf_counter() - t0
                    self.parked.clear()
                    self.gen += 1
                    self.steps += 1
                    last = time.monotonic()
                    self.baton.notify_all()
                    continue
                if self._closed:
                    break
                self.baton.wait(timeout=self._tick_s)
                if self._keep_alive is not None and time.monotonic() - last >= self._tick_s and not self.ready() \
                        and not self._calls:
                    self._keep_alive()
                    last = time.monotonic()

    def close(self):
        with self.baton:
            self._closed = True
            self.baton.notify_all()
