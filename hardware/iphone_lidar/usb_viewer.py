"""Live point cloud from an iPhone LiDAR streamed by the Record3D app over USB.

Run from the repo root:  python -m hardware.iphone_lidar.usb_viewer
Then open:               http://localhost:8000

The phone needs Record3D open with USB Streaming mode on and NOT recording: the red record
button pauses the USB stream. Windows needs Apple Mobile Device Support. Off eduroam,
wifi_viewer.html does the same over Wi-Fi with no Apple software. The SO-101 control panel
(hardware/so101/ui.py) shows the same cloud next to the arms.
"""

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hardware.iphone_lidar.pointcloud import pack_frame, pack_no_points
from hardware.so101.rgbd import Phone

PORT = 8000
PAGE_DIR = Path(__file__).parent


def frame_blob(phone):
    frame = phone.latest()
    if frame is None:
        return pack_no_points()
    return pack_frame(frame.rgb, frame.depth, frame.K)


def make_handler(phone):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

        def do_GET(self):
            if self.path == "/frame":
                blob = frame_blob(phone)
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
    phone = Phone(device_index)
    phone.start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(phone))
    print("Viewer: http://localhost:{}  (Ctrl+C to stop)".format(PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
