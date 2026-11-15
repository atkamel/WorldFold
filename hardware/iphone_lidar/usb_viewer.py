"""Live point cloud from an iPhone LiDAR streamed by the Record3D app over USB.

Run from the repo root:  python -m hardware.iphone_lidar.usb_viewer
Then open:               http://localhost:8000

The phone needs Record3D open with USB Streaming mode on and NOT recording: the red record
button pauses the USB stream. Windows needs Apple Mobile Device Support (ships with iTunes).
Off eduroam, wifi_viewer.html does the same over Wi-Fi with no Apple software.
"""

import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from record3d import Record3DStream

from hardware.iphone_lidar.pointcloud import depth_to_points, pack_points, scale_intrinsics

PORT = 8000
PAGE_DIR = Path(__file__).parent


class FrameStore:
    """The latest packed point cloud, shared between the phone loop and the web server."""

    def __init__(self):
        self.lock = threading.Lock()
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        self.blob = pack_points(empty_points, empty_colors)

    def set(self, blob):
        with self.lock:
            self.blob = blob

    def get(self):
        with self.lock:
            return self.blob


def make_handler(store):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

        def do_GET(self):
            if self.path == "/frame":
                blob = store.get()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(blob)
                return
            if self.path == "/":
                self.path = "/viewer.html"
            super().do_GET()

        def log_message(self, *args):
            pass

    return Handler


def start_server(store, port=PORT):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stream_phone(store, device_index):
    devices = Record3DStream.get_connected_devices()
    print("{} iPhone(s) found over USB".format(len(devices)))
    if len(devices) <= device_index:
        sys.exit("No iPhone at index {}. Plug it in, unlock it, and check Apple's device drivers are installed."
                 .format(device_index))

    new_frame = threading.Event()
    stopped = threading.Event()
    session = Record3DStream()
    session.on_new_frame = new_frame.set
    session.on_stream_stopped = stopped.set
    connected = session.connect(devices[device_index])
    if connected is False:
        sys.exit("Record3D is not serving USB. Open the app, turn on USB Streaming mode in its settings, "
                 "and make sure it is not recording.")

    while not stopped.is_set():
        # The timeout keeps Ctrl+C responsive while the phone is not sending.
        got_frame = new_frame.wait(timeout=0.5)
        if not got_frame:
            continue
        new_frame.clear()

        depth = session.get_depth_frame()
        rgb = session.get_rgb_frame()
        coeffs = session.get_intrinsic_mat()
        # Record3D reports intrinsics for the color image; the depth image is smaller.
        fx, fy, cx, cy = scale_intrinsics(
            coeffs.fx, coeffs.fy, coeffs.tx, coeffs.ty,
            from_width=rgb.shape[1], to_width=depth.shape[1],
        )
        points, colors = depth_to_points(depth, rgb, fx, fy, cx, cy)
        store.set(pack_points(points, colors))

    print("The phone stopped streaming.")


def main():
    device_index = 0
    if len(sys.argv) > 1:
        device_index = int(sys.argv[1])
    store = FrameStore()
    start_server(store)
    print("Viewer: http://localhost:{}".format(PORT))
    stream_phone(store, device_index)


if __name__ == "__main__":
    main()
