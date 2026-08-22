from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
