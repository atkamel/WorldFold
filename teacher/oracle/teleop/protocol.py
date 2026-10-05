"""Wire format between the teleop client (laptop) and server (local MuJoCo or remote Isaac).

Every message = 8-byte header (json length, payload length, big-endian uint32) + json + binary payload.
  client -> server  {"t": "input", "mice": {handle: [dx, dy, wheel, left, right, events]}, "keys": [...],
                     "ext": {handle: is_usb_mouse}, "ct": client_time,
                     "tune": [["adjust", key, +1|-1] | ["reset", key] | ["set", key, value] | ["apply_cloth"]]}
                                                                                  (~60 per second; tune optional)
                    {"t": "bye"}
  server -> client  {"t": "hello", "backend", "dt", "tris", "thickness", "flat_area_m2", "frame_size",
                     "tune": [{key, label, kind, unit, scale, default}]}           (tuning.py)
                    {"t": "frame", "hud": {..., "tune": {key: value}, "cloth_pending"}, "echo", "sim_t", "fps"} + JPEG
                    {"t": "state", "hud", "echo", "sim_t", "fps"} + state payload      (hello "view": "state": the laptop
                     draws; payload = 12 float32 joint angles [L5, Lgrip, R5, Rgrip] + cloth points int16 x3 in units of
                     hello "quant" metres, table frame; hello "origin" = table frame -> Isaac world offset)
                    {"t": "log", "recs": [demo-log records]}      (the laptop keeps its own copy of the demo log)
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
