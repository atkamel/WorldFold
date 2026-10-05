"""One teleop session on RunPod, hands-off: pod from our prebuilt image -> code copied -> Isaac teleop server ->
you play -> demo logs copied to the laptop -> pod DELETED -> billing verified back to zero.
    RUNPOD_ENV_FILE=...\\.env python play.py            (from teacher/runpod)
The session ends when the teleop server ends: you quit the client, 5 min without input, or MAX_S. Any failure
before you can play also deletes the pod (after copying the logs). Ctrl+C here = end now (same cleanup).
"""
import io, os, subprocess, sys, tarfile, time
import rp

HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER = os.path.dirname(HERE)
KEY = os.path.expanduser("~/.ssh/runpod_worldfold")
OUT = os.path.join(TEACHER, "results", "runpod")
MAX_S = int(os.environ.get("MAX_S", "5400"))       # teleop server's own hard stop (90 min)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def pod_info(pid):
    d = rp.gql("query($id: String!) { pod(input: {podId: $id}) { desiredStatus runtime { uptimeInSeconds "
               "ports { ip isIpPublic privatePort publicPort type } } } }", {"id": pid})["pod"]
    ports = {p["privatePort"]: (p["ip"], p["publicPort"]) for p in ((d.get("runtime") or {}).get("ports") or [])
             if p["isIpPublic"] and p["type"] == "tcp"}
    return d, ports


def ssh(host, port, cmd, timeout=60, input_bytes=None):
    return subprocess.run(["ssh", "-i", KEY, "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                           "-o", "BatchMode=yes", f"root@{host}", cmd], input=input_bytes, capture_output=True, timeout=timeout)


def code_tarball():
    """teacher/ without results and caches (~25 MB), extracted on the pod as /workspace/WorldFold/teacher."""
    buf = io.BytesIO()
    skip = ("results", "__pycache__", ".pytest_cache")
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(TEACHER, arcname="teacher",
               filter=lambda ti: None if any(f"/{s}" in ti.name or ti.name.endswith(s) for s in skip) else ti)
    return buf.getvalue()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    pid, host, sp = None, None, None
    try:
        wait_min = float(os.environ.get("WAIT_MIN", "10" if os.environ.get("RUNPOD_REGION", "").lower() == "montreal" else "0"))
        while True:                                   # Montreal only: retry for a while instead of falling back to the US
            try:
                d = rp.deploy(); break
            except SystemExit as e:
                if time.time() - t0 > wait_min * 60:
                    raise RuntimeError(f"no suitable GPU free ({'Montreal' if wait_min else 'North America'}): {e}")
                log("no suitable GPU free yet; retrying in 60 s"); time.sleep(60)
        pid = d["id"]
        log(f"pod {pid}: waiting for it to download the image and boot (first pull of 10 GB on this host)")
        while True:                                   # boot: image pull + start.sh (sshd)
            info, ports = pod_info(pid)
            if info.get("desiredStatus") == "EXITED":   # e.g. the image could not be pulled
                raise RuntimeError("RunPod exited the pod before it ran (image pull / start failed)")
            if 22 in ports and 7777 in ports:
                host, sp = ports[22]
                if ssh(host, sp, "echo ok", timeout=20).returncode == 0:
                    break
            if time.time() - t0 > 1800:
                raise RuntimeError("pod not reachable after 30 min")
            time.sleep(10)
        tport = ports[7777][1]
        log(f"pod up after {time.time() - t0:.0f} s; copying our code")
        r = ssh(host, sp, "mkdir -p /workspace/WorldFold && tar xzf - -C /workspace/WorldFold && echo CODE_OK", input_bytes=code_tarball())
        if b"CODE_OK" not in r.stdout:
            raise RuntimeError("code copy failed: " + r.stderr.decode()[-300:])
        try:   # the detached job can keep the ssh channel open: don't wait for it, the log below confirms the start
            ssh(host, sp, f"cd /workspace && (KEEP_POD=1 MAX_S={MAX_S} setsid nohup bash /workspace/WorldFold/teacher/runpod/"
                          f"run_isaac.sh teleop > /workspace/teleop.log 2>&1 < /dev/null &) ; echo STARTED", timeout=20)
        except subprocess.TimeoutExpired:
            pass
        if ssh(host, sp, "sleep 3; test -s /workspace/teleop.log && echo OK", timeout=30).stdout.strip() != b"OK":
            raise RuntimeError("teleop did not start (no /workspace/teleop.log)")
        log("Isaac starting (scene build ~3-4 min)")
        shown = False
        while True:
            r = ssh(host, sp, "grep -E 'TELEOP|FATAL|Traceback|Error:|isaac exit|session took' /workspace/teleop.log | tail -5", timeout=30)
            out = r.stdout.decode(errors="ignore")
            if "TELEOP listening" in out and not shown:
                log("READY - connect from another terminal in mujoco_live:")
                print(f"\n    python teleop_client.py {host} {tport}\n", flush=True)
                shown = True
            if any(s in out for s in ("FATAL", "isaac exit", "TELEOP end")):   # a Traceback alone may be harmless
                log("session over:\n" + out.strip())
                break
            time.sleep(10)
    except KeyboardInterrupt:
        log("Ctrl+C - ending the session")
    except Exception as e:
        log(f"FAILED: {e!r}")
    finally:
        if host:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            r = ssh(host, sp, "cd /workspace && tar czf - teleop.log results 2>/dev/null", timeout=300)
            if r.stdout:
                p = os.path.join(OUT, f"{stamp}_teleop_session.tgz")
                open(p, "wb").write(r.stdout)
                log(f"logs + server-side demo copied to {p}")
        if pid:
            rp.terminate(pid)
            time.sleep(5)
            rp.status()
        log(f"total pod time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
