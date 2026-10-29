"""Present so pytest puts the repo root on sys.path.

`imitation`, `cloth_fold_rl` and `cloth_angles` are imported as top-level
packages with no install step, so the tests only resolve them when the root is
importable. An empty conftest at the root is what guarantees that.
"""
