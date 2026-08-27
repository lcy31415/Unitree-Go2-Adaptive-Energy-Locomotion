"""Gait-free velocity commands with the Adaptive Energy reward-threshold curriculum."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.configclass import configclass

from .velocity_command import UniformLevelVelocityCommandCfg


class RewardThresholdCurriculum:
    """Cartesian command grid with staged activation and mixed replay sampling.

    Weights retain the reference curriculum's incremental activation rule, but
    their magnitudes no longer determine sampling probability. Sampling mixes
    the current frontier, every active cell, and the initial low-speed cells so
    that saturated easy cells cannot dilute frontier training.
    """

    def __init__(
        self,
        device: str,
        limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        num_bins: tuple[int, int, int],
        initial_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ):
        self.device = device
        self.limits = torch.tensor(limits, device=device, dtype=torch.float)
        self.num_bins = num_bins
        self.bin_widths = (self.limits[:, 1] - self.limits[:, 0]) / torch.tensor(
            num_bins, device=device, dtype=torch.float
        )

        axes = [
            torch.linspace(
                self.limits[index, 0] + 0.5 * self.bin_widths[index],
                self.limits[index, 1] - 0.5 * self.bin_widths[index],
                steps=count,
                device=device,
            )
            for index, count in enumerate(num_bins)
        ]
        self.grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        self.grid = torch.where(
            torch.isclose(self.grid, torch.zeros_like(self.grid), atol=1.0e-6, rtol=0.0),
            torch.zeros_like(self.grid),
            self.grid,
        )
        self.weights = torch.zeros(self.grid.shape[0], device=device)

        # All staged axes use symmetric, odd-sized grids so that a true zero
        # center exists. Before an axis is unlocked, both its bin center and
        # its within-bin sampling noise are constrained to exactly zero.
        self.zero_center_masks = torch.isclose(
            self.grid, torch.zeros(1, 3, device=device), atol=1.0e-6, rtol=0.0
        )

        initial_ranges_tensor = torch.tensor(initial_ranges, device=device, dtype=torch.float)
        initially_active = torch.logical_and(
            self.grid >= initial_ranges_tensor[:, 0], self.grid <= initial_ranges_tensor[:, 1]
        ).all(dim=1)
        if not torch.any(initially_active):
            raise ValueError("The initial command ranges do not contain any curriculum cell centers.")
        self.initially_active = initially_active.clone()
        self.weights[initially_active] = 1.0

    def allowed_mask(self, stage: int) -> torch.Tensor:
        """Return cells allowed by the staged ``vx -> yaw -> vy`` schedule."""
        if stage == 0:
            return torch.logical_and(self.zero_center_masks[:, 1], self.zero_center_masks[:, 2])
        if stage == 1:
            return self.zero_center_masks[:, 1]
        return torch.ones(self.grid.shape[0], dtype=torch.bool, device=self.device)

    def frontier_mask(self, stage: int, frontier_bin_count: int) -> torch.Tensor:
        """Return the highest active band on the axis expanded by this stage."""
        if frontier_bin_count < 1:
            raise ValueError("frontier_bin_count must be at least one.")
        active = torch.logical_and(self.weights > 0.0, self.allowed_mask(stage))
        if not torch.any(active):
            return active

        # Stage 0 prioritizes the positive forward-speed frontier. Yaw and
        # lateral motion are symmetric, so their frontiers use absolute rate.
        axis = (0, 2, 1)[min(stage, 2)]
        coordinate = self.grid[:, axis] if stage == 0 else torch.abs(self.grid[:, axis])
        frontier_coordinate = torch.max(coordinate[active])
        band_width = (frontier_bin_count - 1) * self.bin_widths[axis] + 1.0e-6
        return torch.logical_and(active, coordinate >= frontier_coordinate - band_width)

    def sample(
        self,
        batch_size: int,
        stage: int,
        sampling_probabilities: tuple[float, float, float],
        frontier_bin_count: int,
        max_abs_vx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix curriculum sources while respecting per-environment speed limits."""
        probabilities = torch.tensor(sampling_probabilities, device=self.device, dtype=torch.float)
        if torch.any(probabilities < 0.0) or not torch.isclose(
            probabilities.sum(), torch.tensor(1.0, device=self.device), atol=1.0e-6
        ):
            raise ValueError("Mixed curriculum sampling probabilities must be non-negative and sum to one.")

        allowed = self.allowed_mask(stage)
        active = torch.logical_and(self.weights > 0.0, allowed)
        if not torch.any(active):
            raise RuntimeError(f"The staged velocity curriculum has no active cells at stage {stage}.")
        frontier = self.frontier_mask(stage, frontier_bin_count)
        replay = torch.logical_and(self.initially_active, allowed)
        source_masks = (frontier, active, replay)

        # Source ids: 0=frontier, 1=all active uniformly, 2=initial low-speed replay.
        source_ids = torch.multinomial(probabilities, batch_size, replacement=True)
        bin_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        for source_id, source_mask in enumerate(source_masks):
            sample_indices = torch.nonzero(source_ids == source_id, as_tuple=False).squeeze(-1)
            if sample_indices.numel() == 0:
                continue
            if max_abs_vx is None:
                speed_limits = (None,)
            else:
                speed_limits = torch.unique(max_abs_vx[sample_indices])
            for speed_limit in speed_limits:
                if speed_limit is None:
                    limited_indices = sample_indices
                    candidate_mask = source_mask
                else:
                    limited_indices = sample_indices[
                        torch.isclose(max_abs_vx[sample_indices], speed_limit)
                    ]
                    candidate_mask = source_mask & (self.grid[:, 0].abs() <= speed_limit + 1.0e-6)
                candidates = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
                if candidates.numel() == 0:
                    fallback = active
                    if speed_limit is not None:
                        fallback = fallback & (self.grid[:, 0].abs() <= speed_limit + 1.0e-6)
                    candidates = torch.nonzero(fallback, as_tuple=False).squeeze(-1)
                if candidates.numel() == 0:
                    raise RuntimeError("No active velocity cell satisfies the terrain speed limit.")
                candidate_indices = torch.randint(
                    candidates.numel(), (limited_indices.numel(),), device=self.device
                )
                bin_ids[limited_indices] = candidates[candidate_indices]

        noise = torch.rand(batch_size, 3, device=self.device) - 0.5
        if stage == 0:
            noise[:, 1:] = 0.0
        elif stage == 1:
            noise[:, 1] = 0.0
        commands = self.grid[bin_ids] + noise * self.bin_widths
        if stage == 0:
            commands[:, 1:] = 0.0
        elif stage == 1:
            commands[:, 1] = 0.0
        commands = torch.maximum(torch.minimum(commands, self.limits[:, 1]), self.limits[:, 0])
        if max_abs_vx is not None:
            commands[:, 0] = torch.clamp(commands[:, 0], min=-max_abs_vx, max=max_abs_vx)
        return commands, bin_ids, source_ids

    def update(
        self,
        bin_ids: torch.Tensor,
        success: torch.Tensor,
        local_range: tuple[float, float, float],
        weight_increment: float,
        stage: int,
    ) -> torch.Tensor:
        """Increase weights around cells that pass the physical-error test."""
        successful_bins = bin_ids[success]
        if successful_bins.numel() == 0:
            return success

        unique_bins, counts = torch.unique(successful_bins, return_counts=True)

        # This exactly mirrors RewardThresholdCurriculum.update(): first add
        # 0.2 to each successful cell, then add 0.2 to its full Cartesian
        # neighborhood. Because that neighborhood includes the cell itself,
        # a successful cell receives at least 0.4 before clipping at 1.0.
        allowed = self.allowed_mask(stage)
        self.weights[unique_bins] += weight_increment
        ranges = torch.tensor(local_range, device=self.device)
        neighbors = (
            torch.abs(self.grid.unsqueeze(0) - self.grid[unique_bins].unsqueeze(1))
            <= ranges.view(1, 1, 3)
        ).all(dim=2)
        neighbors = torch.logical_and(neighbors, allowed.unsqueeze(0))
        neighbor_increments = neighbors.T.float() @ counts.float()
        self.weights += weight_increment * neighbor_increments
        # Cells on locked command axes must remain inactive until the
        # corresponding stage is entered.
        self.weights[torch.logical_not(allowed)] = 0.0
        self.weights.clamp_(0.0, 1.0)
        return success


class RewardThresholdVelocityCommand(UniformVelocityCommand):
    """One gait-free ``(v_x, v_y, yaw_rate)`` reward-threshold curriculum."""

    cfg: RewardThresholdVelocityCommandCfg

    def __init__(self, cfg: RewardThresholdVelocityCommandCfg, env):
        super().__init__(cfg, env)

        limits = (
            cfg.limit_ranges.lin_vel_x,
            cfg.limit_ranges.lin_vel_y,
            cfg.limit_ranges.ang_vel_z,
        )
        initial_ranges = (cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z)
        self.curriculum = RewardThresholdCurriculum(env.device, limits, cfg.num_bins, initial_ranges)

        self.bin_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._velocity_abs_error_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self._segment_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_velocity_abs_error_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_within_tolerance_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_command_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_expected_progress = torch.zeros(self.num_envs, device=self.device)
        self._skip_metrics_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage = 0
        self._stage_success_count = 0
        self._terrain_level_ema: float | None = None
        self._terrain_gate_open = cfg.terrain_gate_min_mean_level is None

        if not 0.0 <= cfg.terrain_level_ema_decay < 1.0:
            raise ValueError("terrain_level_ema_decay must be in [0, 1).")
        if (
            cfg.terrain_gate_close_mean_level is not None
            and cfg.terrain_gate_min_mean_level is not None
            and cfg.terrain_gate_close_mean_level >= cfg.terrain_gate_min_mean_level
        ):
            raise ValueError("Terrain gate close threshold must be below its open threshold.")
        if not 0.0 <= cfg.yaw_recovery_sampling_probability < 1.0:
            raise ValueError("yaw_recovery_sampling_probability must be in [0, 1).")
        vx_min, vx_max = cfg.yaw_recovery_forward_range
        yaw_min, yaw_max = cfg.yaw_recovery_abs_yaw_range
        if not 0.0 <= vx_min <= vx_max or not 0.0 <= yaw_min <= yaw_max:
            raise ValueError("Yaw-recovery command ranges must be ordered and non-negative.")
        if cfg.terrain_conditioned_max_abs_vx is not None and any(
            value <= 0.0 for value in cfg.terrain_conditioned_max_abs_vx
        ):
            raise ValueError("Every terrain-conditioned forward-speed limit must be positive.")

        self.metrics["curriculum_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_active_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_mean_weight"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_stage_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_error_vx"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_error_vy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_error_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sample_frontier"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sample_active_uniform"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sample_low_speed_replay"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sample_yaw_recovery"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["terrain_gate_open"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["terrain_level_ema"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_eligible_max_level"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        extras = super().reset(env_ids)
        # CommandManager.compute() runs once after an environment reset. At
        # that point there has been no transition under the newly sampled
        # command, so exclude that state from the next 10-second score window.
        self._skip_metrics_once[env_ids] = True
        # Terrain curriculum is computed before CommandManager.reset(), so the
        # just-finished episode has already consumed these accumulators.
        self._episode_velocity_abs_error_sum[env_ids] = 0.0
        self._episode_steps[env_ids] = 0
        self._episode_within_tolerance_steps[env_ids] = 0
        self._episode_command_progress[env_ids] = 0.0
        self._episode_expected_progress[env_ids] = 0.0
        return extras

    def _update_metrics(self):
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        root_lin_vel_b = getattr(root_lin_vel_b, "torch", root_lin_vel_b)
        root_ang_vel_b = getattr(root_ang_vel_b, "torch", root_ang_vel_b)

        valid = torch.logical_not(self._skip_metrics_once)
        velocity_abs_error = torch.stack(
            (
                torch.abs(root_lin_vel_b[:, 0] - self.vel_command_b[:, 0]),
                torch.abs(root_lin_vel_b[:, 1] - self.vel_command_b[:, 1]),
                torch.abs(root_ang_vel_b[:, 2] - self.vel_command_b[:, 2]),
            ),
            dim=1,
        )
        self._velocity_abs_error_sum[valid] += velocity_abs_error[valid]
        self._segment_steps[valid] += 1
        self._episode_velocity_abs_error_sum[valid] += velocity_abs_error[valid]
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
            (velocity_abs_error[:, 0] <= forward_tolerance)
            & (velocity_abs_error[:, 1] <= lateral_tolerance)
            & (velocity_abs_error[:, 2] <= yaw_tolerance)
        )
        self._episode_within_tolerance_steps[valid] += within_tolerance[valid].long()

        planar_command = self.vel_command_b[:, :2]
        command_speed = torch.linalg.norm(planar_command, dim=1)
        command_direction = planar_command / command_speed.clamp_min(1.0e-6).unsqueeze(1)
        command_aligned_speed = torch.sum(root_lin_vel_b[:, :2] * command_direction, dim=1).clamp_min(0.0)
        moving = command_speed > self.cfg.zero_command_threshold
        self._episode_command_progress[valid] += torch.where(
            moving[valid], command_aligned_speed[valid] * self._env.step_dt, 0.0
        )
        self._episode_expected_progress[valid] += torch.where(
            moving[valid], command_speed[valid] * self._env.step_dt, 0.0
        )
        self._skip_metrics_once[:] = False

        # Preserve the standard command diagnostics.
        super()._update_metrics()

    def _update_curriculum(self, env_ids: torch.Tensor):
        valid = torch.logical_and(self.bin_ids[env_ids] >= 0, self._segment_steps[env_ids] > 0)
        valid_env_ids = env_ids[valid]
        if valid_env_ids.numel() == 0:
            return

        terrain_gate_open = True
        eligible_max_level = self.cfg.curriculum_update_max_terrain_level
        if self.cfg.terrain_gate_min_mean_level is not None:
            terrain = getattr(self._env.scene, "terrain", None)
            terrain_levels = getattr(terrain, "terrain_levels", None)
            if terrain_levels is None:
                terrain_gate_open = False
            else:
                current_mean_level = torch.mean(terrain_levels.float()).item()
                if self._terrain_level_ema is None:
                    self._terrain_level_ema = current_mean_level
                else:
                    decay = self.cfg.terrain_level_ema_decay
                    self._terrain_level_ema = (
                        decay * self._terrain_level_ema + (1.0 - decay) * current_mean_level
                    )
                if self._terrain_gate_open:
                    close_level = self.cfg.terrain_gate_close_mean_level
                    if close_level is not None and self._terrain_level_ema < close_level:
                        self._terrain_gate_open = False
                elif self._terrain_level_ema >= self.cfg.terrain_gate_min_mean_level:
                    self._terrain_gate_open = True
                terrain_gate_open = self._terrain_gate_open

                if self.cfg.curriculum_update_level_margin is not None:
                    dynamic_max = math.floor(self._terrain_level_ema) + self.cfg.curriculum_update_level_margin
                    terrain_max = int(torch.max(terrain_levels).item())
                    configured_max = (
                        self.cfg.curriculum_update_max_terrain_level
                        if self.cfg.curriculum_update_max_terrain_level is not None
                        else terrain_max
                    )
                    eligible_max_level = min(configured_max, max(0, dynamic_max))
        if self._terrain_level_ema is not None:
            self.metrics["terrain_level_ema"][:] = self._terrain_level_ema
        if eligible_max_level is not None:
            self.metrics["curriculum_eligible_max_level"][:] = float(eligible_max_level)
        self.metrics["terrain_gate_open"][:] = float(terrain_gate_open)
        if not terrain_gate_open:
            return

        if eligible_max_level is not None:
            terrain_levels = self._env.scene.terrain.terrain_levels[valid_env_ids]
            eligible = terrain_levels <= eligible_max_level
            valid_env_ids = valid_env_ids[eligible]
            if valid_env_ids.numel() == 0:
                return

        expected_steps = self.cfg.resampling_time_range[0] / self._env.step_dt
        segment_steps = self._segment_steps[valid_env_ids]
        mean_abs_error = self._velocity_abs_error_sum[valid_env_ids] / segment_steps.unsqueeze(1)
        commands = self.vel_command_b[valid_env_ids]

        forward_tolerance = torch.maximum(
            torch.full_like(commands[:, 0], self.cfg.forward_error_abs),
            self.cfg.forward_error_rel * torch.abs(commands[:, 0]),
        )
        lateral_tolerance = torch.maximum(
            torch.full_like(commands[:, 1], self.cfg.lateral_error_abs),
            self.cfg.lateral_error_rel * torch.abs(commands[:, 1]),
        )
        yaw_tolerance = torch.maximum(
            torch.full_like(commands[:, 2], self.cfg.angular_error_abs),
            self.cfg.angular_error_rel * torch.abs(commands[:, 2]),
        )
        completed_segment = segment_steps.float() >= self.cfg.minimum_segment_fraction * expected_steps
        linear_success = torch.logical_and(
            mean_abs_error[:, 0] <= forward_tolerance,
            mean_abs_error[:, 1] <= lateral_tolerance,
        )
        angular_success = mean_abs_error[:, 2] <= yaw_tolerance

        success = torch.logical_and(
            torch.logical_and(linear_success, angular_success), completed_segment
        )
        success = self.curriculum.update(
            self.bin_ids[valid_env_ids],
            success,
            self.cfg.local_range,
            self.cfg.weight_increment,
            self._stage,
        )

        self._update_stage(commands, success)

        allowed = self.curriculum.allowed_mask(self._stage)
        self.metrics["curriculum_success"][valid_env_ids] = success.float()
        self.metrics["curriculum_active_fraction"][valid_env_ids] = (
            self.curriculum.weights[allowed] > 0.0
        ).float().mean()
        self.metrics["curriculum_mean_weight"][valid_env_ids] = self.curriculum.weights[allowed].mean()
        self.metrics["curriculum_stage"][:] = float(self._stage)
        self.metrics["curriculum_stage_progress"][:] = (
            self._stage_success_count / self.cfg.stage_transition_successes if self._stage < 2 else 1.0
        )
        self.metrics["curriculum_error_vx"][valid_env_ids] = mean_abs_error[:, 0]
        self.metrics["curriculum_error_vy"][valid_env_ids] = mean_abs_error[:, 1]
        self.metrics["curriculum_error_yaw"][valid_env_ids] = mean_abs_error[:, 2]

    def _update_stage(self, commands: torch.Tensor, success: torch.Tensor) -> None:
        """Advance globally after repeated successful frontier segments."""
        if self._stage == 0:
            at_frontier = commands[:, 0] >= self.cfg.linear_stage_threshold
        elif self._stage == 1:
            at_frontier = torch.abs(commands[:, 2]) >= self.cfg.angular_stage_threshold
        else:
            return

        self._stage_success_count += int(torch.sum(torch.logical_and(success, at_frontier)).item())
        if self._stage_success_count >= self.cfg.stage_transition_successes:
            self._stage += 1
            self._stage_success_count = 0

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._update_curriculum(env_ids)

        max_abs_vx = None
        if self.cfg.terrain_conditioned_max_abs_vx is not None:
            terrain_levels = self._env.scene.terrain.terrain_levels[env_ids].long()
            envelope = torch.tensor(
                self.cfg.terrain_conditioned_max_abs_vx, device=self.device, dtype=torch.float
            )
            terrain_levels = terrain_levels.clamp(min=0, max=envelope.numel() - 1)
            max_abs_vx = envelope[terrain_levels]

        commands, bin_ids, source_ids = self.curriculum.sample(
            len(env_ids),
            self._stage,
            (
                self.cfg.frontier_sampling_probability,
                self.cfg.active_sampling_probability,
                self.cfg.replay_sampling_probability,
            ),
            self.cfg.frontier_bin_count,
            max_abs_vx=max_abs_vx,
        )

        recovery_probability = self.cfg.yaw_recovery_sampling_probability
        recovery = torch.rand(len(env_ids), device=self.device) < recovery_probability
        if torch.any(recovery):
            count = int(torch.sum(recovery).item())
            vx_min, vx_max = self.cfg.yaw_recovery_forward_range
            yaw_min, yaw_max = self.cfg.yaw_recovery_abs_yaw_range
            recovery_vx = vx_min + torch.rand(count, device=self.device) * (vx_max - vx_min)
            if max_abs_vx is not None:
                recovery_vx = torch.minimum(recovery_vx, max_abs_vx[recovery])
            recovery_yaw = yaw_min + torch.rand(count, device=self.device) * (yaw_max - yaw_min)
            recovery_sign = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                -torch.ones(count, device=self.device),
                torch.ones(count, device=self.device),
            )
            commands[recovery, 0] = recovery_vx
            commands[recovery, 1] = 0.0
            commands[recovery, 2] = recovery_yaw * recovery_sign
            # Recovery anchors train the policy but do not expand command bins.
            bin_ids[recovery] = -1
            source_ids[recovery] = 3
        self.vel_command_b[env_ids] = commands
        self.bin_ids[env_ids] = bin_ids
        self.metrics["sample_frontier"][env_ids] = (source_ids == 0).float()
        self.metrics["sample_active_uniform"][env_ids] = (source_ids == 1).float()
        self.metrics["sample_low_speed_replay"][env_ids] = (source_ids == 2).float()
        self.metrics["sample_yaw_recovery"][env_ids] = (source_ids == 3).float()
        self._velocity_abs_error_sum[env_ids] = 0.0
        self._segment_steps[env_ids] = 0

        # Match the reference command post-processing: tiny planar commands
        # become exactly zero, while yaw commands remain untouched.
        planar_motion = torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=1) > self.cfg.zero_command_threshold
        self.vel_command_b[env_ids, :2] *= planar_motion.unsqueeze(1)
        self.is_standing_env[env_ids] = False


@configclass
class RewardThresholdVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Configuration for :class:`RewardThresholdVelocityCommand`."""

    class_type: type[RewardThresholdVelocityCommand] = RewardThresholdVelocityCommand

    num_bins: tuple[int, int, int] = (21, 1, 21)
    """Number of bins for ``(v_x, v_y, yaw_rate)``."""

    local_range: tuple[float, float, float] = (0.55, 0.55, 0.55)
    """Half-width of the Cartesian neighborhood unlocked after success."""

    weight_increment: float = 0.2
    frontier_sampling_probability: float = 0.4
    """Probability of sampling the highest active band on the current stage axis."""
    active_sampling_probability: float = 0.4
    """Probability of sampling uniformly from every currently active cell."""
    replay_sampling_probability: float = 0.2
    """Probability of replaying the initially active low-speed cells."""
    frontier_bin_count: int = 2
    """Number of outer command-bin layers included in the frontier band."""
    forward_error_abs: float = 0.1
    """Minimum allowed mean absolute forward-velocity error in m/s."""
    forward_error_rel: float = 0.033
    """Relative forward-velocity error allowed above the absolute floor."""
    lateral_error_abs: float = 0.15
    """Minimum allowed mean absolute lateral-velocity error in m/s."""
    lateral_error_rel: float = 0.1
    """Relative lateral-velocity error allowed above the absolute floor."""
    angular_error_abs: float = 0.2
    """Minimum allowed mean absolute yaw-rate error in rad/s."""
    angular_error_rel: float = 0.1
    """Relative yaw-rate error allowed above the absolute floor."""
    minimum_segment_fraction: float = 0.95
    """Fraction of the 10-second segment that must be completed successfully."""
    linear_stage_threshold: float = 2.5
    """Successful absolute forward speed required before yaw is unlocked."""
    angular_stage_threshold: float = 2.5
    """Successful absolute yaw rate required before lateral speed is unlocked."""
    stage_transition_successes: int = 20
    """Number of successful frontier segments required for each stage change."""
    zero_command_threshold: float = 0.2
    terrain_gate_min_mean_level: float | None = None
    """Mean terrain level required before command-bin weights can expand."""
    terrain_gate_close_mean_level: float | None = None
    """Lower EMA threshold that closes an already-open terrain gate."""
    terrain_level_ema_decay: float = 0.0
    """EMA decay for the global mean terrain level; zero uses the latest value."""
    curriculum_update_max_terrain_level: int | None = None
    """Absolute ceiling on terrain levels allowed to expand command bins."""
    curriculum_update_level_margin: int | None = None
    """If set, dynamic ceiling is ``floor(mean-level EMA) + margin``."""
    terrain_conditioned_max_abs_vx: tuple[float, ...] | None = None
    """Maximum absolute forward command for each terrain level."""
    yaw_recovery_sampling_probability: float = 0.0
    """Fraction of commands replaced by non-zero-yaw recovery anchors."""
    yaw_recovery_forward_range: tuple[float, float] = (0.3, 1.0)
    yaw_recovery_abs_yaw_range: tuple[float, float] = (0.2, 0.6)


class MultiTerrainRewardThresholdVelocityCommand(RewardThresholdVelocityCommand):
    """Independent reward-threshold velocity curriculum for every terrain family."""

    cfg: MultiTerrainRewardThresholdVelocityCommandCfg

    def __init__(self, cfg: MultiTerrainRewardThresholdVelocityCommandCfg, env):
        super().__init__(cfg, env)
        if not cfg.terrain_family_names:
            raise ValueError("terrain_family_names must not be empty.")
        if cfg.terrain_columns_per_family < 1:
            raise ValueError("terrain_columns_per_family must be positive.")
        if len(cfg.terrain_family_max_abs_vx) != len(cfg.terrain_family_names):
            raise ValueError("Every terrain family needs one forward-speed envelope.")
        if any(not envelope or any(value <= 0.0 for value in envelope) for envelope in cfg.terrain_family_max_abs_vx):
            raise ValueError("Every family speed envelope must contain positive values.")

        limits = (
            cfg.limit_ranges.lin_vel_x,
            cfg.limit_ranges.lin_vel_y,
            cfg.limit_ranges.ang_vel_z,
        )
        initial_ranges = (cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z)
        self.curricula = [
            RewardThresholdCurriculum(env.device, limits, cfg.num_bins, initial_ranges)
            for _ in cfg.terrain_family_names
        ]
        # Retain the public attribute for tooling that expects a command term
        # to expose one representative curriculum.
        self.curriculum = self.curricula[0]
        family_count = len(cfg.terrain_family_names)
        self._family_stages = [0] * family_count
        self._family_stage_success_counts = [0] * family_count
        self._family_terrain_level_ema: list[float | None] = [None] * family_count
        self._family_terrain_gate_open = [cfg.terrain_gate_min_mean_level is None] * family_count
        self._family_eligible_max_level: list[int | None] = [
            cfg.curriculum_update_max_terrain_level
        ] * family_count
        self._family_last_gate_step = [-1] * family_count

        for family_name in cfg.terrain_family_names:
            for metric_name in (
                "curriculum_stage",
                "curriculum_stage_progress",
                "curriculum_active_fraction",
                "curriculum_mean_weight",
                "terrain_gate_open",
                "terrain_level_ema",
                "curriculum_eligible_max_level",
                "curriculum_error_vx",
                "curriculum_error_vy",
                "curriculum_error_yaw",
            ):
                self.metrics[f"{family_name}/{metric_name}"] = torch.zeros(
                    self.num_envs, device=self.device
                )

    def _family_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        terrain_types = self._env.scene.terrain.terrain_types[env_ids].long()
        return torch.div(
            terrain_types,
            self.cfg.terrain_columns_per_family,
            rounding_mode="floor",
        ).clamp(max=len(self.cfg.terrain_family_names) - 1)

    def _update_family_gate(self, family_index: int) -> tuple[bool, int | None]:
        current_step = int(self._env.common_step_counter)
        if self._family_last_gate_step[family_index] == current_step:
            return (
                self._family_terrain_gate_open[family_index],
                self._family_eligible_max_level[family_index],
            )
        self._family_last_gate_step[family_index] = current_step

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        family_env_ids = all_env_ids[self._family_ids(all_env_ids) == family_index]
        if family_env_ids.numel() == 0:
            self._family_terrain_gate_open[family_index] = False
            return False, self._family_eligible_max_level[family_index]

        terrain_levels = self._env.scene.terrain.terrain_levels[family_env_ids].float()
        current_mean = float(torch.mean(terrain_levels).item())
        previous = self._family_terrain_level_ema[family_index]
        decay = self.cfg.terrain_level_ema_decay
        ema = current_mean if previous is None else decay * previous + (1.0 - decay) * current_mean
        self._family_terrain_level_ema[family_index] = ema

        gate_open = self._family_terrain_gate_open[family_index]
        if self.cfg.terrain_gate_min_mean_level is None:
            gate_open = True
        elif gate_open:
            close_level = self.cfg.terrain_gate_close_mean_level
            if close_level is not None and ema < close_level:
                gate_open = False
        elif ema >= self.cfg.terrain_gate_min_mean_level:
            gate_open = True
        self._family_terrain_gate_open[family_index] = gate_open

        eligible_max = self.cfg.curriculum_update_max_terrain_level
        if self.cfg.curriculum_update_level_margin is not None:
            terrain_max = max(len(self.cfg.terrain_family_max_abs_vx[family_index]) - 1, 0)
            configured_max = terrain_max if eligible_max is None else eligible_max
            eligible_max = min(
                configured_max,
                max(0, math.floor(ema) + self.cfg.curriculum_update_level_margin),
            )
        self._family_eligible_max_level[family_index] = eligible_max
        self._publish_family_state(family_index)
        return gate_open, eligible_max

    def _publish_family_state(self, family_index: int) -> None:
        name = self.cfg.terrain_family_names[family_index]
        stage = self._family_stages[family_index]
        curriculum = self.curricula[family_index]
        allowed = curriculum.allowed_mask(stage)
        progress = (
            self._family_stage_success_counts[family_index] / self.cfg.stage_transition_successes
            if stage < 2
            else 1.0
        )
        self.metrics[f"{name}/curriculum_stage"][:] = float(stage)
        self.metrics[f"{name}/curriculum_stage_progress"][:] = float(progress)
        self.metrics[f"{name}/curriculum_active_fraction"][:] = (
            curriculum.weights[allowed] > 0.0
        ).float().mean()
        self.metrics[f"{name}/curriculum_mean_weight"][:] = curriculum.weights[allowed].mean()
        self.metrics[f"{name}/terrain_gate_open"][:] = float(
            self._family_terrain_gate_open[family_index]
        )
        ema = self._family_terrain_level_ema[family_index]
        self.metrics[f"{name}/terrain_level_ema"][:] = 0.0 if ema is None else ema
        eligible = self._family_eligible_max_level[family_index]
        self.metrics[f"{name}/curriculum_eligible_max_level"][:] = (
            0.0 if eligible is None else float(eligible)
        )
        family_count = len(self.cfg.terrain_family_names)
        self.metrics["curriculum_stage"][:] = sum(self._family_stages) / family_count
        self.metrics["curriculum_stage_progress"][:] = sum(
            (
                self._family_stage_success_counts[index] / self.cfg.stage_transition_successes
                if self._family_stages[index] < 2
                else 1.0
            )
            for index in range(family_count)
        ) / family_count
        self.metrics["terrain_gate_open"][:] = sum(self._family_terrain_gate_open) / family_count
        known_emas = [value for value in self._family_terrain_level_ema if value is not None]
        self.metrics["terrain_level_ema"][:] = (
            sum(known_emas) / len(known_emas) if known_emas else 0.0
        )
        known_eligible = [
            value for value in self._family_eligible_max_level if value is not None
        ]
        self.metrics["curriculum_eligible_max_level"][:] = (
            sum(known_eligible) / len(known_eligible) if known_eligible else 0.0
        )
        active_fractions = []
        mean_weights = []
        for index, family_curriculum in enumerate(self.curricula):
            family_allowed = family_curriculum.allowed_mask(self._family_stages[index])
            active_fractions.append(
                float((family_curriculum.weights[family_allowed] > 0.0).float().mean().item())
            )
            mean_weights.append(float(family_curriculum.weights[family_allowed].mean().item()))
        self.metrics["curriculum_active_fraction"][:] = sum(active_fractions) / family_count
        self.metrics["curriculum_mean_weight"][:] = sum(mean_weights) / family_count

    def _update_curriculum(self, env_ids: torch.Tensor):
        family_ids = self._family_ids(env_ids)
        for family_index, _ in enumerate(self.cfg.terrain_family_names):
            family_reset_ids = env_ids[family_ids == family_index]
            if family_reset_ids.numel() == 0:
                continue
            gate_open, eligible_max = self._update_family_gate(family_index)
            valid = torch.logical_and(
                self.bin_ids[family_reset_ids] >= 0,
                self._segment_steps[family_reset_ids] > 0,
            )
            valid_env_ids = family_reset_ids[valid]
            if not gate_open or valid_env_ids.numel() == 0:
                continue
            if eligible_max is not None:
                levels = self._env.scene.terrain.terrain_levels[valid_env_ids]
                valid_env_ids = valid_env_ids[levels <= eligible_max]
                if valid_env_ids.numel() == 0:
                    continue

            expected_steps = self.cfg.resampling_time_range[0] / self._env.step_dt
            segment_steps = self._segment_steps[valid_env_ids]
            mean_abs_error = self._velocity_abs_error_sum[valid_env_ids] / segment_steps.unsqueeze(1)
            commands = self.vel_command_b[valid_env_ids]
            forward_tolerance = torch.maximum(
                torch.full_like(commands[:, 0], self.cfg.forward_error_abs),
                self.cfg.forward_error_rel * torch.abs(commands[:, 0]),
            )
            lateral_tolerance = torch.maximum(
                torch.full_like(commands[:, 1], self.cfg.lateral_error_abs),
                self.cfg.lateral_error_rel * torch.abs(commands[:, 1]),
            )
            yaw_tolerance = torch.maximum(
                torch.full_like(commands[:, 2], self.cfg.angular_error_abs),
                self.cfg.angular_error_rel * torch.abs(commands[:, 2]),
            )
            completed = segment_steps.float() >= self.cfg.minimum_segment_fraction * expected_steps
            success = (
                (mean_abs_error[:, 0] <= forward_tolerance)
                & (mean_abs_error[:, 1] <= lateral_tolerance)
                & (mean_abs_error[:, 2] <= yaw_tolerance)
                & completed
            )
            stage = self._family_stages[family_index]
            self.curricula[family_index].update(
                self.bin_ids[valid_env_ids],
                success,
                self.cfg.local_range,
                self.cfg.weight_increment,
                stage,
            )
            self._update_family_stage(family_index, commands, success)
            name = self.cfg.terrain_family_names[family_index]
            self.metrics[f"{name}/curriculum_error_vx"][:] = mean_abs_error[:, 0].mean()
            self.metrics[f"{name}/curriculum_error_vy"][:] = mean_abs_error[:, 1].mean()
            self.metrics[f"{name}/curriculum_error_yaw"][:] = mean_abs_error[:, 2].mean()
            self.metrics["curriculum_success"][valid_env_ids] = success.float()
            self.metrics["curriculum_error_vx"][valid_env_ids] = mean_abs_error[:, 0]
            self.metrics["curriculum_error_vy"][valid_env_ids] = mean_abs_error[:, 1]
            self.metrics["curriculum_error_yaw"][valid_env_ids] = mean_abs_error[:, 2]
            self._publish_family_state(family_index)

    def _update_family_stage(
        self, family_index: int, commands: torch.Tensor, success: torch.Tensor
    ) -> None:
        stage = self._family_stages[family_index]
        if stage == 0:
            at_frontier = commands[:, 0] >= self.cfg.linear_stage_threshold
        elif stage == 1:
            at_frontier = torch.abs(commands[:, 2]) >= self.cfg.angular_stage_threshold
        else:
            return
        self._family_stage_success_counts[family_index] += int(
            torch.sum(success & at_frontier).item()
        )
        if self._family_stage_success_counts[family_index] >= self.cfg.stage_transition_successes:
            self._family_stages[family_index] += 1
            self._family_stage_success_counts[family_index] = 0

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._update_curriculum(env_ids)
        family_ids = self._family_ids(env_ids)
        commands = torch.empty(len(env_ids), 3, device=self.device)
        bin_ids = torch.full((len(env_ids),), -1, dtype=torch.long, device=self.device)
        source_ids = torch.empty(len(env_ids), dtype=torch.long, device=self.device)

        for family_index, _ in enumerate(self.cfg.terrain_family_names):
            local_indices = torch.nonzero(family_ids == family_index, as_tuple=False).squeeze(-1)
            if local_indices.numel() == 0:
                continue
            family_env_ids = env_ids[local_indices]
            levels = self._env.scene.terrain.terrain_levels[family_env_ids].long()
            envelope = torch.tensor(
                self.cfg.terrain_family_max_abs_vx[family_index],
                device=self.device,
                dtype=torch.float,
            )
            max_abs_vx = envelope[levels.clamp(min=0, max=envelope.numel() - 1)]
            family_commands, family_bins, family_sources = self.curricula[family_index].sample(
                local_indices.numel(),
                self._family_stages[family_index],
                (
                    self.cfg.frontier_sampling_probability,
                    self.cfg.active_sampling_probability,
                    self.cfg.replay_sampling_probability,
                ),
                self.cfg.frontier_bin_count,
                max_abs_vx=max_abs_vx,
            )
            commands[local_indices] = family_commands
            bin_ids[local_indices] = family_bins
            source_ids[local_indices] = family_sources

        recovery = torch.rand(len(env_ids), device=self.device) < self.cfg.yaw_recovery_sampling_probability
        if torch.any(recovery):
            count = int(recovery.sum().item())
            vx_min, vx_max = self.cfg.yaw_recovery_forward_range
            yaw_min, yaw_max = self.cfg.yaw_recovery_abs_yaw_range
            recovery_vx = vx_min + torch.rand(count, device=self.device) * (vx_max - vx_min)
            recovery_yaw = yaw_min + torch.rand(count, device=self.device) * (yaw_max - yaw_min)
            signs = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                -torch.ones(count, device=self.device),
                torch.ones(count, device=self.device),
            )
            # Respect every family/level envelope for the injected anchors too.
            recovery_env_ids = env_ids[recovery]
            recovery_family_ids = family_ids[recovery]
            for family_index in range(len(self.cfg.terrain_family_names)):
                selected = recovery_family_ids == family_index
                if not torch.any(selected):
                    continue
                levels = self._env.scene.terrain.terrain_levels[recovery_env_ids[selected]].long()
                envelope = torch.tensor(
                    self.cfg.terrain_family_max_abs_vx[family_index],
                    device=self.device,
                    dtype=torch.float,
                )
                limits = envelope[levels.clamp(min=0, max=envelope.numel() - 1)]
                recovery_vx[selected] = torch.minimum(recovery_vx[selected], limits)
            commands[recovery, 0] = recovery_vx
            commands[recovery, 1] = 0.0
            commands[recovery, 2] = recovery_yaw * signs
            bin_ids[recovery] = -1
            source_ids[recovery] = 3

        self.vel_command_b[env_ids] = commands
        self.bin_ids[env_ids] = bin_ids
        self.metrics["sample_frontier"][env_ids] = (source_ids == 0).float()
        self.metrics["sample_active_uniform"][env_ids] = (source_ids == 1).float()
        self.metrics["sample_low_speed_replay"][env_ids] = (source_ids == 2).float()
        self.metrics["sample_yaw_recovery"][env_ids] = (source_ids == 3).float()
        self._velocity_abs_error_sum[env_ids] = 0.0
        self._segment_steps[env_ids] = 0
        planar_motion = (
            torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=1)
            > self.cfg.zero_command_threshold
        )
        self.vel_command_b[env_ids, :2] *= planar_motion.unsqueeze(1)
        self.is_standing_env[env_ids] = False

    def state_dict(self) -> dict:
        """Serialize every family curriculum and the current terrain allocation."""
        terrain = self._env.scene.terrain
        return {
            "version": 1,
            "family_names": tuple(self.cfg.terrain_family_names),
            "weights": [curriculum.weights.detach().cpu() for curriculum in self.curricula],
            "stages": list(self._family_stages),
            "stage_success_counts": list(self._family_stage_success_counts),
            "terrain_level_ema": list(self._family_terrain_level_ema),
            "terrain_gate_open": list(self._family_terrain_gate_open),
            "eligible_max_level": list(self._family_eligible_max_level),
            "terrain_levels": terrain.terrain_levels.detach().cpu(),
            "terrain_types": terrain.terrain_types.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore family states and remap saved levels when env count changes."""
        if tuple(state["family_names"]) != tuple(self.cfg.terrain_family_names):
            raise ValueError("Checkpoint terrain families do not match the current task.")
        for curriculum, weights in zip(self.curricula, state["weights"], strict=True):
            if curriculum.weights.shape != weights.shape:
                raise ValueError("Checkpoint velocity grid shape does not match the current task.")
            curriculum.weights.copy_(weights.to(self.device))
        self._family_stages = [int(value) for value in state["stages"]]
        self._family_stage_success_counts = [int(value) for value in state["stage_success_counts"]]
        self._family_terrain_level_ema = [
            None if value is None else float(value) for value in state["terrain_level_ema"]
        ]
        self._family_terrain_gate_open = [bool(value) for value in state["terrain_gate_open"]]
        self._family_eligible_max_level = list(state["eligible_max_level"])

        terrain = self._env.scene.terrain
        saved_levels = state.get("terrain_levels")
        saved_types = state.get("terrain_types")
        if saved_levels is not None and saved_types is not None:
            saved_levels = saved_levels.to(self.device)
            saved_types = saved_types.to(self.device)
            current_ids = torch.arange(self.num_envs, device=self.device)
            current_families = self._family_ids(current_ids)
            saved_families = torch.div(
                saved_types.long(), self.cfg.terrain_columns_per_family, rounding_mode="floor"
            ).clamp(max=len(self.cfg.terrain_family_names) - 1)
            for family_index in range(len(self.cfg.terrain_family_names)):
                current = current_ids[current_families == family_index]
                candidates = saved_levels[saved_families == family_index]
                if current.numel() > 0 and candidates.numel() > 0:
                    repeats = (current.numel() + candidates.numel() - 1) // candidates.numel()
                    terrain.terrain_levels[current] = candidates.repeat(repeats)[: current.numel()]
            terrain.env_origins[:] = terrain.terrain_origins[
                terrain.terrain_levels, terrain.terrain_types
            ]
        for family_index in range(len(self.cfg.terrain_family_names)):
            self._publish_family_state(family_index)

    def resample_current_episodes(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.bin_ids[:] = -1
        self._segment_steps[:] = 0
        self._resample_command(env_ids)

    def csv_snapshot(self) -> tuple[tuple[str, ...], list[tuple]]:
        rows = []
        for family_index, name in enumerate(self.cfg.terrain_family_names):
            curriculum = self.curricula[family_index]
            for bin_id in range(curriculum.grid.shape[0]):
                rows.append(
                    (
                        name,
                        bin_id,
                        *curriculum.grid[bin_id].detach().cpu().tolist(),
                        float(curriculum.weights[bin_id].item()),
                        self._family_stages[family_index],
                    )
                )
        return ("terrain_family", "bin_id", "vx", "vy", "yaw", "weight", "stage"), rows


@configclass
class MultiTerrainRewardThresholdVelocityCommandCfg(RewardThresholdVelocityCommandCfg):
    """Configuration for independent per-terrain reward-threshold curricula."""

    class_type: type[MultiTerrainRewardThresholdVelocityCommand] = (
        MultiTerrainRewardThresholdVelocityCommand
    )
    terrain_family_names: tuple[str, ...] = ()
    terrain_columns_per_family: int = 1
    terrain_family_max_abs_vx: tuple[tuple[float, ...], ...] = ()
