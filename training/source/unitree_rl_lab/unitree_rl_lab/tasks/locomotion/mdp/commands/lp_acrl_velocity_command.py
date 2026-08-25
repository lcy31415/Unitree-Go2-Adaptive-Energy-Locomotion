"""Episode-fixed velocity commands assigned by the LP-ACRL curriculum."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm
from isaaclab.utils.configclass import configclass


class LPACRLVelocityCommand(UniformVelocityCommand):
    """Hold a curriculum-assigned command for one complete episode."""

    cfg: LPACRLVelocityCommandCfg

    def __init__(self, cfg: LPACRLVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.metrics.clear()
        self.task_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._pending_task_ids = torch.full_like(self.task_ids, -1)
        self._pending_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._has_pending_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Whole-episode statistics consumed by the rough-terrain curriculum
        # before CommandManager.reset() clears them.
        self._episode_velocity_abs_error_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_within_tolerance_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_command_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_expected_progress = torch.zeros(self.num_envs, device=self.device)
        self._skip_metrics_once = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def assign(self, env_ids: torch.Tensor, task_ids: torch.Tensor, commands: torch.Tensor) -> None:
        self._pending_task_ids[env_ids] = task_ids
        self._pending_commands[env_ids] = commands
        self._has_pending_command[env_ids] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # The curriculum owns task metrics; omit generic command error logs.
        if env_ids is None:
            env_ids = slice(None)
        extras = CommandTerm.reset(self, env_ids)
        self._skip_metrics_once[env_ids] = True
        self._episode_velocity_abs_error_sum[env_ids] = 0.0
        self._episode_steps[env_ids] = 0
        self._episode_within_tolerance_steps[env_ids] = 0
        self._episode_command_progress[env_ids] = 0.0
        self._episode_expected_progress[env_ids] = 0.0
        return extras

    def _update_metrics(self) -> None:
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        root_lin_vel_b = getattr(root_lin_vel_b, "torch", root_lin_vel_b)
        root_ang_vel_b = getattr(root_ang_vel_b, "torch", root_ang_vel_b)
        valid = ~self._skip_metrics_once

        error = torch.stack(
            (
                torch.abs(root_lin_vel_b[:, 0] - self.vel_command_b[:, 0]),
                torch.abs(root_lin_vel_b[:, 1] - self.vel_command_b[:, 1]),
                torch.abs(root_ang_vel_b[:, 2] - self.vel_command_b[:, 2]),
            ),
            dim=1,
        )
        self._episode_velocity_abs_error_sum[valid] += error[valid]
        self._episode_steps[valid] += 1

        forward_tolerance = torch.maximum(
            torch.full_like(self.vel_command_b[:, 0], self.cfg.forward_error_abs),
            self.cfg.forward_error_rel * torch.abs(self.vel_command_b[:, 0]),
        )
        lateral_tolerance = torch.maximum(
            torch.full_like(self.vel_command_b[:, 1], self.cfg.lateral_error_abs),
            self.cfg.lateral_error_rel * torch.abs(self.vel_command_b[:, 1]),
        )
        yaw_tolerance = torch.maximum(
            torch.full_like(self.vel_command_b[:, 2], self.cfg.angular_error_abs),
            self.cfg.angular_error_rel * torch.abs(self.vel_command_b[:, 2]),
        )
        within_tolerance = (
            (error[:, 0] <= forward_tolerance)
            & (error[:, 1] <= lateral_tolerance)
            & (error[:, 2] <= yaw_tolerance)
        )
        self._episode_within_tolerance_steps[valid] += within_tolerance[valid].long()

        planar_command = self.vel_command_b[:, :2]
        command_speed = torch.linalg.norm(planar_command, dim=1)
        command_direction = planar_command / command_speed.clamp_min(1.0e-6).unsqueeze(1)
        aligned_speed = torch.sum(root_lin_vel_b[:, :2] * command_direction, dim=1).clamp_min(0.0)
        moving = command_speed > self.cfg.zero_command_threshold
        self._episode_command_progress[valid] += torch.where(
            moving[valid], aligned_speed[valid] * self._env.step_dt, 0.0
        )
        self._episode_expected_progress[valid] += torch.where(
            moving[valid], command_speed[valid] * self._env.step_dt, 0.0
        )
        self._skip_metrics_once[:] = False

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        if not bool(torch.all(self._has_pending_command[env_ids])):
            missing = env_ids[~self._has_pending_command[env_ids]]
            raise RuntimeError(f"LP-ACRL did not assign commands for env IDs {missing[:8].tolist()}.")
        self.vel_command_b[env_ids] = self._pending_commands[env_ids]
        self.task_ids[env_ids] = self._pending_task_ids[env_ids]
        self._has_pending_command[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False


@configclass
class LPACRLVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for episode-fixed LP-ACRL velocity commands."""

    class_type: type[LPACRLVelocityCommand] = LPACRLVelocityCommand
    forward_error_abs: float = 0.1
    forward_error_rel: float = 0.033
    lateral_error_abs: float = 0.15
    lateral_error_rel: float = 0.1
    angular_error_abs: float = 0.2
    angular_error_rel: float = 0.1
    zero_command_threshold: float = 0.2
