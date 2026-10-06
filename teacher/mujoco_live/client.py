"""Send one JSON command to the live MuJoCo server and print the reply.
Usage: python client.py '{"op":"obs"}'"""
import json, socket, sys

cmd = sys.argv[1]
s = socket.create_connection(("127.0.0.1", 7788), timeout=600)
s.sendall((json.dumps(json.loads(cmd)) + "\n").encode())
buf = b""
while not buf.endswith(b"\n"):
    c = s.recv(1 << 20)
    if not c:
        break
    buf += c
r = json.loads(buf.decode())
obs = r.pop("obs", r if "landmarks_cm" in r else None)
if obs is r:
    r = {}
if r:
    print(json.dumps(r))
if obs:
    mp = obs.pop("map", None)
    print("score:", json.dumps(obs["score"]))
    print("tips:", obs["tips_cm"], "holding:", obs["holding"], "t:", obs["t"])
    print("landmarks:", json.dumps(obs["landmarks_cm"]))
    if mp:
        print(f"map (1 char = 1 cm, x from {mp['x_from_cm']} cm, far side on top):")
        print("\n".join(mp["rows_top_is_far"]))
