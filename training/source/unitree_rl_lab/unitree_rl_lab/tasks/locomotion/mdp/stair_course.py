"""MDP terms for a directed, finite stair traversal course."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def _tensor(value):
    """Return the torch view used by both tensor and Warp-backed Isaac Lab data."""
    return getattr(value, "torch", value)


def _course_displacement(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    root_pos_w = _tensor(asset.data.root_pos_w)
    env_origins = _tensor(env.scene.terrain.env_origins).to(root_pos_w.device)
    return root_pos_w[:, :2] - env_origins[:, :2]


def stair_course_complete(
    env,
    finish_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return true once the robot reaches the +x landing beyond the stairs."""
    if finish_distance <= 0.0:
        raise ValueError("finish_distance must be positive.")
    return _course_displacement(env, asset_cfg)[:, 0] >= finish_distance


def stair_lateral_deviation(
    env,
    max_deviation: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate robots that leave the directed course laterally."""
    if max_deviation <= 0.0:
        raise ValueError("max_deviation must be positive.")
    return _course_displacement(env, asset_cfg)[:, 1].abs() > max_deviation


def stair_heading_error(
    env,
    max_heading_error: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate robots whose body heading turns too far from world +x."""
    if not 0.0 < max_heading_error <= torch.pi:
        raise ValueError("max_heading_error must be in (0, pi].")
    asset = env.scene[asset_cfg.name]
    root_quat_w = _tensor(asset.data.root_quat_w)
    forward_b = torch.zeros((root_quat_w.shape[0], 3), device=root_quat_w.device, dtype=root_quat_w.dtype)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(root_quat_w, forward_b)
    heading = torch.atan2(forward_w[:, 1], forward_w[:, 0])
    return heading.abs() > max_heading_error


def stair_course_completion_reward(
    env,
    finish_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-shot terminal bonus signal for successful course completion."""
    return stair_course_complete(env, finish_distance, asset_cfg).float()
