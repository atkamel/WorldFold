"""Speed patches for the lehome_solution policy server (applied in the Modal image, env-gated).

Every patch is OFF unless its env var is set, and the OFF path is the original code, so one
image serves both the baseline and the fast server for A/B runs.

  TEACHER_FAST_OUT=1   (C) one device->host transfer of all model outputs, then numpy slicing.
                           Replaces ~150-240 per-row GPU slice ops per model call (~1 ms each).
  TEACHER_FLAT_STATE=1 (F) module_jit passes the model state as a pre-flattened leaf list instead
                           of an nnx.State pytree, so JAX stops re-walking the 3B-param tree every call.
  TEACHER_BUCKETS=1    (D) pad main and retry batches to sizes 1/2/4/6/8/12/16/24/32 -> few compiled shapes.
  TEACHER_RETRY=0      (E) disable the best-of-N retry pass (quality trade-off; default on).

Usage (inside the image):  python fast_server_patch.py /opt/lehome_solution
Fails loudly if the upstream code no longer matches (pinned: lehome_solution main + openpi c23745b5).
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-fast-patch]"


def patch(rel, edits):
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        print(f"already patched: {rel}")
        return
    for old, new in edits:
        n = s.count(old)
        assert n == 1, f"{rel}: expected 1 match, found {n} for:\n{old[:200]}"
        s = s.replace(old, new)
    p.write_text(MARK + "\n" + s)
    print(f"patched: {rel}")


# ---------------- F: pre-flattened state in openpi's module_jit ----------------
patch("openpi/src/openpi/shared/nnx_utils.py", [
    ("import flax.nnx as nnx\nimport jax\n",
     "import os as _os\n\nimport flax.nnx as nnx\nimport jax\n"),
    ("""    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper
""",
     """    if _os.environ.get("TEACHER_FLAT_STATE") == "1":
        leaves, treedef = jax.tree_util.tree_flatten(state)

        def fun_flat(leaves, *args, **kwargs):
            return fun(jax.tree_util.tree_unflatten(treedef, leaves), *args, **kwargs)

        jitted_flat = jax.jit(fun_flat, *jit_args, **jit_kwargs)

        @functools.wraps(meth)
        def wrapper_flat(*args: P.args, **kwargs: P.kwargs) -> R:
            return jitted_flat(leaves, *args, **kwargs)

        return wrapper_flat

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper
"""),
])

# ---------------- C: single host transfer of outputs ----------------
patch("src/lehome_solution/policies/pi_modified_policy.py", [
    ('"""Minimal Policy subclass', 'import os as _os\n"""Minimal Policy subclass'),
    ("""        BN = B * N
        if N > 1:
            state_tiled = jnp.repeat(batched_inputs["state"], N, axis=0)
        else:
            state_tiled = batched_inputs["state"]
        results = []
""",
     """        BN = B * N
        if N > 1:
            state_tiled = jnp.repeat(batched_inputs["state"], N, axis=0)
        else:
            state_tiled = batched_inputs["state"]
        if _os.environ.get("TEACHER_FAST_OUT") == "1":
            # one device->host transfer for everything, then numpy slicing (no per-row GPU slice ops)
            (state_tiled, actions, value_pred, checkpoint_pred, garment_type_pred, completion_pred,
             ttc_pred, keypoint_distances_pred, wm_flow_preds) = jax.device_get(
                (state_tiled, actions, value_pred, checkpoint_pred, garment_type_pred, completion_pred,
                 ttc_pred, keypoint_distances_pred, wm_flow_preds))
        results = []
"""),
])

# ---------------- D + E: batch buckets and retry switch ----------------
patch("src/lehome_solution/shared/eval_wrapper.py", [
    ("class LeHomePolicyWrapper", '''import os as _os


def _pad_bucket(obs_l, ia_l, ov_l):
    """TEACHER_BUCKETS=1: pad to a fixed set of batch sizes so JAX compiles few shapes."""
    if _os.environ.get("TEACHER_BUCKETS") != "1":
        return obs_l, ia_l, ov_l
    n = len(obs_l)
    b = next((s for s in (1, 2, 4, 6, 8, 12, 16, 24, 32) if n <= s), n)
    k = b - n
    return obs_l + [obs_l[-1]] * k, ia_l + [ia_l[-1]] * k, ov_l + [ov_l[-1]] * k


class LeHomePolicyWrapper'''),
    ("""            outputs = self.policy.infer_batched(
                obs_list, ia_list, overrides_list, num_candidates=N_batch,
                cfg_disabled=self.config.cfg_disabled,
            )
""",
     """            _ol, _il, _vl = _pad_bucket(obs_list, ia_list, overrides_list)
            outputs = self.policy.infer_batched(
                _ol, _il, _vl, num_candidates=N_batch,
                cfg_disabled=self.config.cfg_disabled,
            )
            outputs = outputs[: B * N_batch]
"""),
    ("            if retry_needed:\n",
     '            if retry_needed and _os.environ.get("TEACHER_RETRY", "1") != "0":\n'),
    ("""                    retry_out = self.policy.infer_batched(
                        retry_obs, retry_ia, retry_overrides, num_candidates=R,
                        cfg_disabled=self.config.cfg_disabled,
                    )
""",
     """                    _ro, _ri, _rv = _pad_bucket(retry_obs, retry_ia, retry_overrides)
                    retry_out = self.policy.infer_batched(
                        _ro, _ri, _rv, num_candidates=R,
                        cfg_disabled=self.config.cfg_disabled,
                    )
"""),
])
print("PATCH_OK")
