"""Wire format between the teleop client (laptop) and server (local MuJoCo or remote Isaac).

Every message = 8-byte header (json length, payload length, big-endian uint32) + json + binary payload.
  client -> server  {"t": "input", "mice": {handle: [dx, dy, wheel, left, right, events]}, "keys": [...],
                     "ext": {handle: is_usb_mouse}, "ct": client_time,
                     "tune": [["adjust", key, +1|-1] | ["reset", key] | ["set", key, value] | ["apply_cloth"]],
                     "got": newest state frame number fully received (-1 = none yet)}
                                                                                  (~60 per second; tune, got optional)
                    {"t": "bye"}
  server -> client  {"t": "hello", "backend", "dt", "tris", "thickness", "flat_area_m2", "frame_size",
                     "tune": [{key, label, kind, unit, scale, default}]}           (tuning.py)
                    {"t": "frame", "hud": {..., "tune": {key: value}, "cloth_pending"}, "echo", "sim_t", "fps"} + JPEG
                    {"t": "state", "hud", "echo", "sim_t", "fps"} + state payload      (hello "view": "state": the laptop
                     draws; payload = 12 float32 joint angles [L5, Lgrip, R5, Rgrip] + cloth points int16 x3 in units of
                     hello "quant" metres, table frame; hello "origin" = table frame -> Isaac world offset)
                     Compressed state (when the client's inputs carry "got"): header also has "fid" (frame number),
                     "base" (frame it is a difference from, -1 = full frame), "codec" ("blosc2" | "raw"); payload =
                     codec(payload int16 lanes - base frame's lanes). See StateEncoder / StateDecoder.
                    {"t": "log", "recs": [demo-log records]}      (the laptop keeps its own copy of the demo log; a
                     grab/drop record's cloth is pack_cloth's compact form, or "local" when the laptop draws from the
                     state stream: it already has the cloth and fills in its own)
                    {"t": "end", "why": ...}
"""
import json
import struct

HDR = struct.Struct("!II")


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def send(sock, header, payload=b""):
    h = json.dumps(header, separators=(",", ":"), default=_default).encode()
    sock.sendall(HDR.pack(len(h), len(payload)) + h + payload)


def send_json(sock, header_json, payload=b""):
    """send() for a header that is already JSON text."""
    h = header_json.encode()
    sock.sendall(HDR.pack(len(h), len(payload)) + h + payload)


def pack_cloth(cloth_cm):
    """Cloth points in cm (N x 3, rounded to 0.01 cm by the controller) -> a small JSON-able dict, exact at that
    rounding (int16 in 0.01 cm, +-327 cm). Well under 1 ms, where a JSON list of 44k floats took ~10-24 ms with
    Python's lock held (the physics loop waited)."""
    import base64
    import numpy as np
    q = np.clip(np.round(np.asarray(cloth_cm, float) * 100), -32767, 32767).astype("<i2")
    return {"i16": base64.b64encode(q.tobytes()).decode("ascii"), "scale": 0.01}


def cloth_cm(rec):
    """A demo-log record's cloth as an (N, 3) array in cm, from either form (a JSON list in older logs, or
    pack_cloth's dict). None when the record has none."""
    import base64
    import numpy as np
    c = rec.get("cloth_cm")
    if isinstance(c, dict):
        return np.frombuffer(base64.b64decode(c["i16"]), "<i2").reshape(-1, 3) * c["scale"]
    return None if c is None or isinstance(c, str) else np.asarray(c, float)


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv(sock):
    """Returns (header dict, payload bytes) or None when the other side closed."""
    head = _recvall(sock, HDR.size)
    if head is None:
        return None
    hl, pl = HDR.unpack(head)
    h = _recvall(sock, hl)
    p = _recvall(sock, pl) if pl else b""
    if h is None or p is None:
        return None
    return json.loads(h), p


# ---- compressed state stream ---------------------------------------------------------------------------------------
# Three battle-tested ideas, mixed: Quake 3's snapshot deltas (send the difference from the newest frame the client has
# confirmed), Blosc2 (shuffle + zstd for arrays of numbers) to squeeze that difference, and VNC-style flow control (only
# `window` frames may be unconfirmed at a time, so a slow link gets fewer, always-fresh frames instead of a queue).
# The difference is taken over the payload's int16 lanes with wrap-around arithmetic, so decoding is bit-exact for
# any payload (the joint floats included).

def _compress(raw):
    try:
        import blosc2
    except ImportError:
        return raw, "raw"
    return blosc2.compress(raw, typesize=2, clevel=1, filter=blosc2.Filter.SHUFFLE, codec=blosc2.Codec.ZSTD), "blosc2"


def _decompress(body, codec):
    if codec == "raw":
        return body
    import blosc2
    return blosc2.decompress(body)


class StateEncoder:
    """Server side. ack(got) when the client confirms a frame; can_send() before encode(payload).

    How many frames may be unconfirmed adapts to the link (TCP Vegas' rule): while round trips stay near the fastest
    one seen in the last few seconds, grow one frame at a time toward what covers a round trip at the frame rate
    (+2 spare); as soon as round trips rise above that, the link is queueing frames, so shrink. A far but fast link
    gets every frame through; a slow link keeps few frames in flight and the delay stays flat.
    Between `window` and `max_window` frames."""

    def __init__(self, window=3, max_window=16):
        import time
        self._now = time.perf_counter
        self.window, self.min_window, self.max_window = window, window, max_window
        self.next, self.acked, self.sent = 0, -1, {}
        self.t_sent = {}                  # fid -> send time, until confirmed
        self.rtts = []                    # (time, round trip) of recent confirmations
        self.t_enc = []                   # times of recent frames (frame rate)

    def ack(self, got):
        got = int(got)
        if got <= self.acked:
            return
        now = self._now()
        t = self.t_sent.get(got)
        self.t_sent = {k: v for k, v in self.t_sent.items() if k > got}
        self.acked = got
        if t is None:
            return
        rtt = now - t
        self.rtts.append((now, rtt))
        self.rtts = [x for x in self.rtts if now - x[0] < 3.0]
        rtt_min = min(r for _, r in self.rtts)
        if rtt > 1.25 * rtt_min + 0.010:              # frames are waiting somewhere: fewer in flight
            self.window = max(self.min_window, self.window - 1)
        elif len(self.t_enc) > 5:
            rate = (len(self.t_enc) - 1) / max(self.t_enc[-1] - self.t_enc[0], 1e-6)
            target = min(self.max_window, int(rtt_min * rate) + 2)
            self.window = max(self.min_window, min(target, self.window + 1))

    def can_send(self):
        return self.next - 1 - self.acked < self.window

    def encode(self, payload):
        import numpy as np
        cur = np.frombuffer(payload, "<i2")
        acked = self.acked
        ref = self.sent.get(acked)
        base = acked if ref is not None and len(ref) == len(cur) else -1
        diff = cur - ref if base >= 0 else cur          # int16 - int16 wraps around: exact on decode
        body, codec = _compress(diff.tobytes())
        fid, self.next = self.next, self.next + 1
        self.sent = {k: v for k, v in self.sent.items() if k >= acked}
        self.sent[fid] = cur.copy()
        now = self._now()
        self.t_sent[fid] = now
        self.t_enc.append(now)
        self.t_enc = [x for x in self.t_enc if now - x < 1.0]
        return dict(fid=fid, base=base, codec=codec), body


class StateDecoder:
    """Client side. decode(header, payload) -> the original payload bytes; `last` = newest frame decoded."""

    def __init__(self):
        self.frames, self.last = {}, -1

    def decode(self, h, body):
        import numpy as np
        if "fid" not in h:                             # uncompressed server
            return body
        diff = np.frombuffer(_decompress(body, h["codec"]), "<i2")
        cur = diff if h["base"] < 0 else self.frames[h["base"]] + diff
        self.frames = {k: v for k, v in self.frames.items() if k >= h["base"]}
        self.frames[h["fid"]] = cur
        self.last = h["fid"]
        return cur.tobytes()
