from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def adaptive_energy_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    asset_cfg=None,
    minimum_expected_progress: float = 1.0,
    minimum_tracking_fraction: float = 0.8,
    minimum_tracking_fraction_for_hold: float | None = None,
    move_up_distance_fraction: float = 0.5,
    move_down_expected_fraction: float = 0.5,
) -> dict[str, torch.Tensor | float]:
    """Promote terrain using survival, physical tracking error and command-aligned progress.

    The command term accumulates statistics across command resampling boundaries,
    so an episode with opposite commands cannot pass merely because its final
    Euclidean displacement happens to be large.
    """
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    terrain = env.scene.terrain
    command_term = env.command_manager.get_term(command_name)
    episode_steps = command_term._episode_steps[env_ids]
    valid = episode_steps > 0

    if not torch.any(valid):
        return {
            "mean_level": torch.mean(terrain.terrain_levels.float()),
            "max_level": torch.max(terrain.terrain_levels.float()),
            "move_up_fraction": 0.0,
            "move_down_fraction": 0.0,
            "hold_fraction": 0.0,
            "tracking_success": 0.0,
            "survival_rate": 0.0,
        }

    valid_env_ids = env_ids[valid]
    steps = episode_steps[valid].unsqueeze(1)
    mean_error = command_term._episode_velocity_abs_error_sum[valid_env_ids] / steps
    tracking_fraction = command_term._episode_within_tolerance_steps[valid_env_ids] / steps.squeeze(1)
    if minimum_tracking_fraction_for_hold is None:
        minimum_tracking_fraction_for_hold = minimum_tracking_fraction
    if not 0.0 <= minimum_tracking_fraction_for_hold <= minimum_tracking_fraction <= 1.0:
        raise ValueError(
            "Terrain tracking fractions must satisfy "
            "0 <= hold <= promotion <= 1."
        )
    tracking_success = tracking_fraction >= minimum_tracking_fraction
    tracking_failure = tracking_fraction < minimum_tracking_fraction_for_hold

    episode_length = env.episode_length_buf[valid_env_ids]
    survived = episode_length >= env.max_episode_length - 1
    progress = command_term._episode_command_progress[valid_env_ids]
    expected_progress = command_term._episode_expected_progress[valid_env_ids]
    moving = expected_progress >= minimum_expected_progress
    terrain_half_length = terrain.cfg.terrain_generator.size[0] * move_up_distance_fraction
    progress_success = torch.where(moving, progress >= terrain_half_length, torch.ones_like(moving))

    move_up = survived & tracking_success & progress_success
    progress_failure = moving & (progress < expected_progress * move_down_expected_fraction)
    move_down = (~move_up) & ((~survived) | tracking_failure | progress_failure)
    hold = ~(move_up | move_down)
    terrain.update_env_origins(valid_env_ids, move_up, move_down)

    return {
        "mean_level": torch.mean(terrain.terrain_levels.float()),
        "max_level": torch.max(terrain.terrain_levels.float()),
        "move_up_fraction": torch.mean(move_up.float()),
        "move_down_fraction": torch.mean(move_down.float()),
        "hold_fraction": torch.mean(hold.float()),
        "tracking_success": torch.mean(tracking_success.float()),
        "tracking_fraction": torch.mean(tracking_fraction),
        "survival_rate": torch.mean(survived.float()),
        "mean_progress": torch.mean(progress),
        "mean_error_vx": torch.mean(mean_error[:, 0]),
        "mean_error_vy": torch.mean(mean_error[:, 1]),
        "mean_error_yaw": torch.mean(mean_error[:, 2]),
    }


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_x_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    expansion_step: float = 0.1,
    success_ratio: float = 0.8,
) -> torch.Tensor:
    """Expand only the longitudinal command range after successful episodes.

    Unlike :func:`lin_vel_cmd_levels`, this curriculum never changes lateral
    or yaw commands. This keeps the adaptive-energy baseline strictly limited
    to straight-line velocity tracking.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * success_ratio:
            delta_command = torch.tensor([-expansion_step, expansion_step], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def velocity_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    linear_reward_term_name: str = "track_lin_vel_xy",
    angular_reward_term_name: str = "track_ang_vel_z",
    expansion_step: float = 0.1,
    linear_success_ratio: float = 0.8,
    angular_success_ratio: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Expand longitudinal, lateral, and yaw command ranges together."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    linear_term = env.reward_manager.get_term_cfg(linear_reward_term_name)
    angular_term = env.reward_manager.get_term_cfg(angular_reward_term_name)
    linear_reward = (
        torch.mean(env.reward_manager._episode_sums[linear_reward_term_name][env_ids]) / env.max_episode_length_s
    )
    angular_reward = (
        torch.mean(env.reward_manager._episode_sums[angular_reward_term_name][env_ids]) / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        linear_success = linear_reward > linear_term.weight * linear_success_ratio
        angular_success = angular_reward > angular_term.weight * angular_success_ratio
        if linear_success and angular_success:
            delta = torch.tensor([-expansion_step, expansion_step], device=env.device)
            for range_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                current = torch.tensor(getattr(ranges, range_name), device=env.device)
                limits = getattr(limit_ranges, range_name)
                setattr(ranges, range_name, torch.clamp(current + delta, limits[0], limits[1]).tolist())

    return {
        "lin_vel_x": torch.tensor(ranges.lin_vel_x[1], device=env.device),
        "lin_vel_y": torch.tensor(ranges.lin_vel_y[1], device=env.device),
        "ang_vel_z": torch.tensor(ranges.ang_vel_z[1], device=env.device),
    }


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
