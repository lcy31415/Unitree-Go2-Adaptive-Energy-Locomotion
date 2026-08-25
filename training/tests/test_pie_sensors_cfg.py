"""Configuration tests for PIE ray-caster sensors."""

from __future__ import annotations

import math

import pytest
import torch

from unitree_rl_lab.tasks.locomotion.robots.go2.pie_sensors_cfg import (
    PIE_DEPTH_CAMERA_ROT_XYZW,
    PIE_DEPTH_HEIGHT,
    PIE_DEPTH_HORIZONTAL_APERTURE,
    PIE_DEPTH_RAW_WIDTH,
    PIE_FOOT_ORDER,
    make_pie_depth_camera_cfg,
    make_pie_foot_scanner_cfg,
    make_pie_height_scanner_cfg,
)


def quaternion_rotation_matrix_xyzw(quaternion: tuple[float, float, float, float]) -> torch.Tensor:
    x, y, z, w = quaternion
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_depth_camera_intrinsics_pose_and_rate() -> None:
    cfg = make_pie_depth_camera_cfg()
    assert cfg.prim_path == "{ENV_REGEX_NS}/Robot/base"
    assert cfg.pattern_cfg.width == PIE_DEPTH_RAW_WIDTH
    assert cfg.pattern_cfg.height == PIE_DEPTH_HEIGHT
    assert cfg.pattern_cfg.horizontal_aperture == pytest.approx(PIE_DEPTH_HORIZONTAL_APERTURE)
    assert cfg.update_period == pytest.approx(0.1)
    assert cfg.data_types == ["distance_to_camera"]
    assert cfg.offset.convention == "opengl"
    assert cfg.offset.pos == (0.345, 0.0, 0.07)

    rotation = quaternion_rotation_matrix_xyzw(PIE_DEPTH_CAMERA_ROT_XYZW)
    optical_forward = rotation @ torch.tensor([0.0, 0.0, -1.0])
    expected = torch.tensor([math.cos(math.radians(20.0)), 0.0, -math.sin(math.radians(20.0))])
    torch.testing.assert_close(optical_forward, expected, atol=1.0e-6, rtol=1.0e-6)


def test_height_grid_has_198_rays() -> None:
    cfg = make_pie_height_scanner_cfg()
    assert cfg.prim_path == "{ENV_REGEX_NS}/Robot/base"
    ray_starts, ray_directions = cfg.pattern_cfg.func(cfg.pattern_cfg, "cpu")
    assert ray_starts.shape == (198, 3)
    assert ray_directions.shape == (198, 3)
    torch.testing.assert_close(ray_directions, torch.tensor([0.0, 0.0, -1.0]).expand_as(ray_directions))
    assert cfg.ray_alignment == "yaw"


def test_four_foot_scanners_have_world_downward_four_ray_grids() -> None:
    for foot_name in PIE_FOOT_ORDER:
        cfg = make_pie_foot_scanner_cfg(foot_name)
        ray_starts, ray_directions = cfg.pattern_cfg.func(cfg.pattern_cfg, "cpu")
        assert ray_starts.shape == (4, 3)
        assert ray_directions.shape == (4, 3)
        assert cfg.ray_alignment == "world"
        assert cfg.max_distance == pytest.approx(0.6)
        assert cfg.prim_path == f"{{ENV_REGEX_NS}}/Robot/{foot_name}_foot"

    with pytest.raises(ValueError, match="Unknown Go2 foot"):
        make_pie_foot_scanner_cfg("XX")
