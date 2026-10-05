"""Split a directory tree into chunk dirs of at most LIMIT bytes each, so the final image can COPY them as separate
layers (GitHub's registry rejects layers over 10 GB, and an upload must finish within 10 minutes).
    python3 pack.py <src_dir> <chunks_dir> <limit_mb> <n_chunks>
Moves files into <chunks_dir>/NN/<absolute path>; whatever is left (empty dirs, small leftovers) goes to <chunks_dir>/skel.
"""
import os, shutil, sys

src, out, limit, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) * 2**20, int(sys.argv[4])


def size(p):
    if os.path.islink(p) or not os.path.isdir(p):
        return os.lstat(p).st_size
    return sum(size(os.path.join(p, c)) for c in os.listdir(p))


def items(p):
    """Largest units under the limit: a whole dir if it fits, else its children; files over the limit stand alone."""
    s = size(p)
    if s <= limit or os.path.islink(p) or not os.path.isdir(p):
        return [(s, p)]
    return [it for c in sorted(os.listdir(p)) for it in items(os.path.join(p, c))]


its = sorted(items(src), reverse=True)
bins = []                                            # first-fit decreasing
for s, p in its:
    for b in bins:
        if b[0] + s <= limit:
            b[0] += s; b[1].append(p); break
    else:
        bins.append([s, [p]])
assert len(bins) <= n, f"{len(bins)} chunks needed, Dockerfile has {n}: raise n_chunks"
for k in range(n):
    os.makedirs(f"{out}/{k:02d}", exist_ok=True)
for k, (s, paths) in enumerate(bins):
    for p in paths:
        dst = f"{out}/{k:02d}{p}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(p, dst)
    print(f"chunk {k:02d}: {s / 2**30:5.2f} GB, {len(paths)} items")
os.makedirs(f"{out}/skel{os.path.dirname(src)}", exist_ok=True)
shutil.move(src, f"{out}/skel{src}")                 # the remaining skeleton: empty dirs + leftovers
print(f"{len(bins)} chunks + skel ({size(out + '/skel') / 2**20:.0f} MB)")
