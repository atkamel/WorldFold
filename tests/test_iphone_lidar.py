import struct

import numpy as np

from hardware.iphone_lidar.pointcloud import depth_to_points, pack_frame, pack_points, scale_intrinsics


def test_pixel_at_principal_point_lies_on_the_optical_axis():
    depth = np.full((4, 6), 1.5, dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    points, colors = depth_to_points(depth, rgb, fx=10.0, fy=10.0, cx=2.0, cy=1.0)
    assert points.shape == (24, 3)
    index = 1 * 6 + 2
    np.testing.assert_allclose(points[index], [0.0, 0.0, 1.5], atol=1e-6)


def test_pixel_off_centre_follows_the_pinhole_model():
    depth = np.full((4, 6), 2.0, dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    points, colors = depth_to_points(depth, rgb, fx=10.0, fy=20.0, cx=2.0, cy=1.0)
    index = 3 * 6 + 5
    np.testing.assert_allclose(points[index], [(5 - 2) * 2.0 / 10.0, (3 - 1) * 2.0 / 20.0, 2.0], atol=1e-6)


def test_points_outside_the_depth_range_are_dropped():
    depth = np.array([[0.0, 0.5, 5.0, np.nan]], dtype=np.float32)
    rgb = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]], dtype=np.uint8)
    points, colors = depth_to_points(depth, rgb, fx=1.0, fy=1.0, cx=0.0, cy=0.0, min_depth=0.1, max_depth=3.0)
    assert points.shape == (1, 3)
    assert colors.tolist() == [[2, 2, 2]]


def test_color_image_larger_than_depth_is_sampled_down():
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[0:2, 2:4] = [255, 0, 0]
    points, colors = depth_to_points(depth, rgb, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    assert colors.shape == (4, 3)
    assert colors[1].tolist() == [255, 0, 0]
    assert colors[0].tolist() == [0, 0, 0]


def test_intrinsics_scale_with_image_width():
    fx, fy, cx, cy = scale_intrinsics(fx=1000.0, fy=1000.0, cx=480.0, cy=360.0, from_width=960, to_width=240)
    assert (fx, fy, cx, cy) == (250.0, 250.0, 120.0, 90.0)


def test_packed_frame_round_trips():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    colors = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    blob = pack_points(points, colors)
    count = struct.unpack("<i", blob[0:4])[0]
    assert count == 2
    unpacked_points = np.frombuffer(blob, dtype="<f4", count=6, offset=4).reshape(2, 3)
    unpacked_colors = np.frombuffer(blob, dtype=np.uint8, count=6, offset=4 + 24).reshape(2, 3)
    np.testing.assert_array_equal(unpacked_points, points)
    np.testing.assert_array_equal(unpacked_colors, colors)


def test_packed_camera_frame_uses_intrinsics_scaled_to_the_depth_image():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    blob = pack_frame(rgb, depth, (2.0, 2.0, 2.0, 2.0))
    count = struct.unpack("<i", blob[0:4])[0]
    assert count == 4
    points = np.frombuffer(blob, dtype="<f4", count=12, offset=4).reshape(4, 3)
    np.testing.assert_allclose(points[3], [0.0, 0.0, 1.0], atol=1e-6)
