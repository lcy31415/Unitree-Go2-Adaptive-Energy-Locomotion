"""Reusable Isaac Lab ray-caster configurations for Unitree Go2 PIE."""

from __future__ import annotations

import math

from isaaclab.sensors import RayCasterCameraCfg, RayCasterCfg, patterns


PIE_DEPTH_RAW_WIDTH = 106
PIE_DEPTH_HEIGHT = 60
PIE_DEPTH_CROP_LEFT = 10
PIE_DEPTH_CROP_RIGHT = 10
PIE_DEPTH_WIDTH = PIE_DEPTH_RAW_WIDTH - PIE_DEPTH_CROP_LEFT - PIE_DEPTH_CROP_RIGHT
PIE_DEPTH_HORIZONTAL_FOV_DEG = 87.0
PIE_DEPTH_CUTOFF_DISTANCE = 3.0
PIE_DEPTH_MIN_DISTANCE = 0.05
PIE_DEPTH_UPDATE_PERIOD = 0.1
PIE_DEPTH_FOCAL_LENGTH = 24.0
PIE_DEPTH_HORIZONTAL_APERTURE = 2.0 * PIE_DEPTH_FOCAL_LENGTH * math.tan(
    math.radians(PIE_DEPTH_HORIZONTAL_FOV_DEG) / 2.0
)

# Source MJLab quaternion reordered from wxyz to xyzw. In the OpenGL camera
# convention it maps local -Z onto robot (+X, 0, -sin(20 deg)).
PIE_DEPTH_CAMERA_ROT_XYZW = (0.4055798, -0.4055798, -0.5792280, 0.5792280)

PIE_FOOT_ORDER = ("FR", "FL", "RR", "RL")
PIE_FOOT_SENSOR_NAMES = tuple(f"pie_{foot.lower()}_foot_scanner" for foot in PIE_FOOT_ORDER)


def make_pie_depth_camera_cfg(update_period: float = PIE_DEPTH_UPDATE_PERIOD) -> RayCasterCameraCfg:
    """Create the 106x60, 87-degree PIE front depth camera."""
    return RayCasterCameraCfg(
        # Track the existing rigid-body prim. The PhysX ray-caster backend
        # does not materialize configured child sensor Xforms before its
        # initialization callback in the Isaac Lab version used here.
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.345, 0.0, 0.07),
            rot=PIE_DEPTH_CAMERA_ROT_XYZW,
            convention="opengl",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=PIE_DEPTH_FOCAL_LENGTH,
            horizontal_aperture=PIE_DEPTH_HORIZONTAL_APERTURE,
            width=PIE_DEPTH_RAW_WIDTH,
            height=PIE_DEPTH_HEIGHT,
        ),
        data_types=["distance_to_camera"],
        depth_clipping_behavior="none",
        max_distance=PIE_DEPTH_CUTOFF_DISTANCE,
        debug_vis=False,
    )


def make_pie_height_scanner_cfg(update_period: float = 0.02) -> RayCasterCfg:
    """Create the inclusive 18x11 privileged terrain-height grid."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.7, 1.0)),
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
        # Rays start 20 m above the body to avoid terrain penetration. The
        # observation itself is clipped and normalized at 5 m.
        max_distance=100.0,
        debug_vis=False,
    )


def make_pie_foot_scanner_cfg(foot_name: str, update_period: float = 0.02) -> RayCasterCfg:
    """Create a small world-downward grid attached to one Go2 foot."""
    if foot_name not in PIE_FOOT_ORDER:
        raise ValueError(f"Unknown Go2 foot {foot_name!r}; expected one of {PIE_FOOT_ORDER}.")
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}_foot",
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.02, 0.02)),
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
        max_distance=0.6,
        debug_vis=False,
    )


def make_pie_base_clearance_scanner_cfg(update_period: float = 0.02) -> RayCasterCfg:
    """Create a compact world-downward scanner directly below the Go2 base."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.02, 0.02)),
        mesh_prim_paths=["/World/ground"],
        update_period=update_period,
        max_distance=0.8,
        debug_vis=False,
    )
