"""Live point cloud from an iPhone LiDAR streamed by the Record3D app over USB.

Run from the repo root:  python -m hardware.iphone_lidar.usb_viewer
Then open:               http://localhost:8000

Phone and driver setup: see camera.py. Off eduroam, wifi_viewer.html does the same over
Wi-Fi with no Apple software. The SO-101 control panel (hardware/so101/ui.py) shows the
same cloud next to the arms.
"""

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from hardware.iphone_lidar.camera import IphoneCamera
from hardware.iphone_lidar.pointcloud import depth_to_points, pack_points, scale_intrinsics

PORT = 8000
PAGE_DIR = Path(__file__).parent


def frame_blob(camera):
    """The camera's latest frame as packed points for viewer.html (no points when there is no frame)."""
    frame = camera.latest()
    if frame is None:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        return pack_points(empty_points, empty_colors)
    rgb, depth, intrinsics = frame
    # Record3D reports intrinsics for the color image; the depth image is smaller.
    fx, fy, cx, cy = scale_intrinsics(
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
        from_width=rgb.shape[1], to_width=depth.shape[1],
    )
    points, colors = depth_to_points(depth, rgb, fx, fy, cx, cy)
    return pack_points(points, colors)


def make_handler(camera):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

        def do_GET(self):
            if self.path == "/frame":
                blob = frame_blob(camera)
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


def main():
    device_index = 0
    if len(sys.argv) > 1:
        device_index = int(sys.argv[1])
    camera = IphoneCamera()
    if not camera.start(device_index):
        sys.exit(camera.status)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(camera))
    print("Viewer: http://localhost:{}  (Ctrl+C to stop)".format(PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
