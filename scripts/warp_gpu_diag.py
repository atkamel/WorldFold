"""What can one process allocate on a WATcloud GPU shard? (docs/warp_port.md phase 2)

Prints the CUDA/MPS/SLURM environment, then allocates growing blocks with torch
and with Warp until one fails. Run inside a job:
    cloud/watcloud/watcloud.sh run --gpu --time 0:20:00 -- python scripts/warp_gpu_diag.py
"""
import os
import subprocess

import torch
import warp as wp

for k, v in sorted(os.environ.items()):
    if any(t in k for t in ("CUDA", "MPS", "NVIDIA", "GRES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "GPU")):
        print(f"env {k}={v}")
print(subprocess.run(["nvidia-smi", "-q", "-d", "MEMORY"], capture_output=True, text=True).stdout.strip()[:1500])
print(subprocess.run(["bash", "-lc", "ulimit -a; cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /proc/self/cgroup"],
                     capture_output=True, text=True).stdout.strip()[:1200])

wp.init()
dev = wp.get_device("cuda:0")
print(f"warp: free {dev.free_memory / 2**30:.2f} of {dev.total_memory / 2**30:.2f} GiB; torch free/total "
      f"{[x / 2**30 for x in torch.cuda.mem_get_info()]}")
held = []
for gib in (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8):
    try:
        held.append(torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda"))
        print(f"torch cumulative {sum(t.numel() for t in held) / 2**30:.2f} GiB ok")
    except Exception as e:
        print(f"torch +{gib} GiB FAILED: {str(e)[:120]}")
        break
del held
torch.cuda.empty_cache()
held = []
for gib in (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8):
    try:
        held.append(wp.empty(int(gib * 2**30), dtype=wp.uint8, device=dev))
        print(f"warp cumulative {sum(a.size for a in held) / 2**30:.2f} GiB ok")
    except Exception as e:
        print(f"warp +{gib} GiB FAILED: {str(e)[:120]}")
        break
# single big blocks (the CCD workspace is one allocation)
for gib in (1, 2, 4):
    try:
        a = wp.empty(int(gib * 2**30), dtype=wp.uint8, device=dev); del a
        print(f"warp single {gib} GiB ok")
    except Exception as e:
        print(f"warp single {gib} GiB FAILED: {str(e)[:120]}")
