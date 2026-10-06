"""Smoke test of the real pygame teleop client (window, JPEG decode, HUD, local scoring) for a few seconds,
without grabbing the pointer.   python smoke_client.py PORT   (a teleop server must be running)"""
import sys, time
import numpy as np
import teleop_client as tc

tc.Client.set_play = lambda self, on: setattr(self, "playing", False)   # never grab the pointer in a test
c = tc.Client("127.0.0.1", int(sys.argv[1]), fullscreen=False)
t0 = time.time()
n = 0
while time.time() - t0 < 3.0:
    c.send_input({}, [])
    c.draw()
    n += 1
    time.sleep(1 / 60)
got_frame = getattr(c, "surf", None) is not None
# feed one 'drop' through the local scorer (flat shirt from the hello -> should score 0)
flat = c.hello.get("flat_cloth_cm")
if c.tris is not None:
    from neat_metric import fold_score
    s = fold_score(np.array(flat or [[0, 0, 0]]) / 100, c.tris, c.hello["flat_area_m2"], c.hello["thickness"]) if flat else None
print(f"client ran {n} frames of its loop; received a server frame: {got_frame}; server fps {c.srv_fps:.0f}; "
      f"lag {None if c.lag is None else round(c.lag * 1000)} ms; window {c.screen.get_size()}; "
      f"flat area {c.hello.get('flat_area_m2')}")
c.close()
