"""Minimal RunPod API helper (GraphQL). The key comes from RUNPOD_API_KEY or a .env file named by RUNPOD_ENV_FILE;
it is never printed.
    python rp.py status                 # balance, spend/hr, every pod with status, cost, uptime, GPU use
    python rp.py pubkey <file.pub>      # add an SSH public key to the account (keeps existing keys)
    python rp.py ports <pod_id>         # public IP + port mappings (SSH 22, teleop 7777)
    python rp.py terminate <pod_id>     # delete a pod (stops all billing for it)
    python rp.py registry               # save the GitHub read-only token (GHCR_PULL_TOKEN in the env file) in RunPod
    python rp.py deploy [image]         # one 4090 pod; default image = our prebuilt Isaac image (ghcr, private)
"""
import json, os, sys, urllib.request

URL = "https://api.runpod.io/graphql"


def env_value(name):
    v = os.environ.get(name, "").strip()
    if not v and os.environ.get("RUNPOD_ENV_FILE"):
        for line in open(os.environ["RUNPOD_ENV_FILE"], encoding="utf-8"):
            if line.strip().startswith(name):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
    return v.strip("<>")    # a pasted "<token>" (placeholder brackets kept) is still the token


def key():
    k = env_value("RUNPOD_API_KEY")
    if not k:
        sys.exit("no RunPod key (set RUNPOD_API_KEY or RUNPOD_ENV_FILE)")
    return k


IMAGE = "ghcr.io/cimurghe/worldfold-isaac:5.1"   # built from teacher/runpod/image/ (private: needs the registry auth)
REGISTRY_NAME = "ghcr-worldfold"


def registry():
    """Store the GitHub read:packages token in RunPod (once). The token is read from GHCR_PULL_TOKEN, never printed."""
    tok = env_value("GHCR_PULL_TOKEN")
    if not tok:
        sys.exit("no GHCR_PULL_TOKEN in the env file")
    have = gql("query { myself { containerRegistryCreds { id name } } }")["myself"].get("containerRegistryCreds") or []
    for c in have:
        if c["name"] == REGISTRY_NAME:
            if "--replace" not in sys.argv:
                print(f"registry auth already saved: {c['id']} (--replace to save the token again)"); return c["id"]
            gql("mutation($id: String!) { deleteRegistryAuth(registryAuthId: $id) }", {"id": c["id"]})
            print("old registry auth deleted")
    d = gql("mutation($n: String!, $u: String!, $p: String!) { saveRegistryAuth(input: {name: $n, username: $u, password: $p}) "
            "{ id name } }", {"n": REGISTRY_NAME, "u": "Cimurghe", "p": tok})["saveRegistryAuth"]
    print(f"registry auth saved: {d['id']}")
    return d["id"]


def registry_id():
    have = gql("query { myself { containerRegistryCreds { id name } } }")["myself"].get("containerRegistryCreds") or []
    return next((c["id"] for c in have if c["name"] == REGISTRY_NAME), None)


def gql(query, variables=None):
    req = urllib.request.Request(URL, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key()}",
                                          "User-Agent": "worldfold-teacher"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    if out.get("errors"):
        sys.exit("API error: " + json.dumps(out["errors"])[:500])
    return out["data"]


def status():
    d = gql("""query { myself { clientBalance currentSpendPerHr pubKey
              pods { id name desiredStatus costPerHr gpuCount machine { gpuDisplayName }
                     runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } } } } }""")["myself"]
    print(f"balance ${d['clientBalance']:.2f} | spending now ${d['currentSpendPerHr'] or 0:.3f}/hr | "
          f"SSH keys on account: {len([l for l in (d.get('pubKey') or '').splitlines() if l.strip()])}")
    if not d["pods"]:
        print("pods: none")
    for p in d["pods"]:
        rt = p.get("runtime") or {}
        gpus = rt.get("gpus") or []
        print(f"pod {p['id']} '{p['name']}' {p['desiredStatus']} ${p['costPerHr']}/hr {(p.get('machine') or {}).get('gpuDisplayName')} "
              f"up {rt.get('uptimeInSeconds')} s gpu {[g.get('gpuUtilPercent') for g in gpus]}%")
    return d


def add_pubkey(path):
    new = open(path).read().strip()
    cur = gql("query { myself { pubKey } }")["myself"].get("pubKey") or ""
    if new in cur:
        print("key already on the account"); return
    gql("mutation($k: String!) { updateUserSettings(input: {pubKey: $k}) { id } }", {"k": (cur.strip() + "\n" + new).strip()})
    print("SSH public key added (existing keys kept)")


def ports(pod_id):
    d = gql("query($id: String!) { pod(input: {podId: $id}) { id desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } }",
            {"id": pod_id})["pod"]
    for p in (d.get("runtime") or {}).get("ports") or []:
        print(f"{p['type']} {p['privatePort']} -> {p['ip']}:{p['publicPort']} public={p['isIpPublic']}")


MTL_DCS = ["CA-MTL-1", "CA-MTL-2", "CA-MTL-3", "CA-MTL-4"]   # ~25-30 ms from Waterloo (estimate); US-IL measured 49 ms
NA_DCS = ["CA-MTL-1", "CA-MTL-2", "CA-MTL-3", "CA-MTL-4", "US-IL-1", "US-DE-1", "US-NC-1", "US-GA-1", "US-GA-2", "US-KS-2", "US-TX-3", "US-TX-4", "US-CA-2", "US-WA-1"]


# RTX cards Isaac 5.1 runs on, cheapest first. Not Blackwell (5090, RTX PRO 4500/6000): the image's torch is cu126.
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S", "NVIDIA RTX A6000",
        "NVIDIA A40", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4080", "NVIDIA GeForce RTX 4080 SUPER", "NVIDIA RTX A5000"]


def deploy(image=IMAGE, name="worldfold-isaac", gpus=GPUS, disk=60):
    for gpu in gpus:
        try:
            return deploy_one(image, name, gpu, disk)
        except SystemExit as e:
            print(f"{gpu}: {e}")
    sys.exit("none of " + ", ".join(gpus) + " available in North America right now")


def deploy_one(image=IMAGE, name="worldfold-isaac", gpu="NVIDIA GeForce RTX 4090", disk=60):
    """One on-demand 4090 pod, North America (closest to Ontario first), CUDA 13 hosts only (Isaac 5.1 needs driver 580+).
    Our image: Isaac + LeHome preinstalled (no setup on the clock); its start.sh runs SSH with the account key."""
    q = """mutation($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) {
             id costPerHr machine { gpuDisplayName } } }"""
    base = dict(cloudType="ALL", gpuCount=1, gpuTypeId=gpu, containerDiskInGb=disk, volumeInGb=0, name=name,
                imageName=image, ports="22/tcp,7777/tcp", startSsh=True, allowedCudaVersions=["13.0"])
    if image.startswith("ghcr.io/"):
        rid = registry_id()
        if not rid:
            sys.exit("private image: run `rp.py registry` first")
        base["containerRegistryAuthId"] = rid
    dcs = MTL_DCS if os.environ.get("RUNPOD_REGION", "").lower() == "montreal" else NA_DCS
    for dc in dcs:
        try:
            d = gql(q, {"in": dict(base, dataCenterId=dc)})["podFindAndDeployOnDemand"]
            if d:
                print(f"POD {d['id']} in {dc}: {d['machine']['gpuDisplayName']} ${d['costPerHr']}/hr")
                return d
        except SystemExit as e:
            print(f"{dc}: {str(e)[:160]}")
    sys.exit(f"no {gpu} with CUDA 13 free in North America")


def terminate(pod_id):
    gql("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})
    print(f"terminated {pod_id}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"status": lambda: status(), "pubkey": lambda: add_pubkey(sys.argv[2]), "ports": lambda: ports(sys.argv[2]),
     "terminate": lambda: terminate(sys.argv[2]), "registry": lambda: registry(),
     "deploy": lambda: deploy(*(sys.argv[2:3] or [IMAGE]))}[cmd]()
