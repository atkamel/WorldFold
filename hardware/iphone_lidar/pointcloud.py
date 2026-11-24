"""Turn a depth image and its camera intrinsics into a colored 3D point cloud.

Points are in the camera frame: x right, y down, z forward, in meters.
"""

import struct

import numpy as np

# Calibration knobs, in meters. LiDAR depth outside this range is dropped.
MIN_DEPTH = 0.1
MAX_DEPTH = 3.0


def scale_intrinsics(fx, fy, cx, cy, from_width, to_width):
    """Intrinsics of the same camera after its image is resized from from_width to to_width."""
    scale = to_width / from_width
    return fx * scale, fy * scale, cx * scale, cy * scale


def sample_colors(rgb, height, width):
    """Nearest-neighbour resize of an HxWx3 color image to height x width."""
    rgb_height = rgb.shape[0]
    rgb_width = rgb.shape[1]
    rows = np.arange(height) * rgb_height // height
    cols = np.arange(width) * rgb_width // width
    picked_rows = rgb[rows]
    return picked_rows[:, cols]


def depth_to_points(depth, rgb, fx, fy, cx, cy, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH):
    """Back-project a depth image through the pinhole model.

    depth: HxW float32, meters. rgb: color image of the same view, any resolution.
    fx, fy, cx, cy: intrinsics at the depth image's resolution.
    Returns (points Nx3 float32, colors Nx3 uint8) for the pixels with a depth in range.
    """
    height = depth.shape[0]
    width = depth.shape[1]

    pixel_x = np.arange(width, dtype=np.float32)
    pixel_y = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(pixel_x, pixel_y)

    x = (grid_x - cx) * depth / fx
    y = (grid_y - cy) * depth / fy
    points = np.stack([x, y, depth], axis=2).reshape(-1, 3)
    colors = sample_colors(rgb, height, width).reshape(-1, 3)

    flat_depth = depth.reshape(-1)
    # NaN compares false, so pixels with no depth reading are dropped here too.
    in_range = (flat_depth >= min_depth) & (flat_depth <= max_depth)
    return points[in_range].astype(np.float32), colors[in_range]


def pack_points(points, colors):
    """Bytes for the browser: int32 point count, then xyz float32 triples, then rgb uint8 triples."""
    count = points.shape[0]
    header = struct.pack("<i", count)
    return header + points.astype("<f4").tobytes() + colors.astype(np.uint8).tobytes()


def pack_frame(rgb, depth, intrinsics):
    """A camera frame as packed points. intrinsics: (fx, fy, cx, cy) of the color image."""
    # Record3D reports intrinsics for the color image; the depth image is smaller.
    fx, fy, cx, cy = scale_intrinsics(
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
        from_width=rgb.shape[1], to_width=depth.shape[1],
    )
    points, colors = depth_to_points(depth, rgb, fx, fy, cx, cy)
    return pack_points(points, colors)


def pack_no_points():
    """What the viewer gets while there is no camera frame."""
    empty_points = np.zeros((0, 3), dtype=np.float32)
    empty_colors = np.zeros((0, 3), dtype=np.uint8)
    return pack_points(empty_points, empty_colors)
