"""Gait-free velocity commands with the Adaptive Energy reward-threshold curriculum."""

from __future__ import annotations

from collections.abc import Sequence

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix frontier, active-uniform, and low-speed replay samples."""
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
            candidates = torch.nonzero(source_mask, as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                candidates = torch.nonzero(active, as_tuple=False).squeeze(-1)
            candidate_indices = torch.randint(candidates.numel(), (sample_indices.numel(),), device=self.device)
            bin_ids[sample_indices] = candidates[candidate_indices]

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
        self.metrics["terrain_gate_open"] = torch.zeros(self.num_envs, device=self.device)

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
        if self.cfg.terrain_gate_min_mean_level is not None:
            terrain = getattr(self._env.scene, "terrain", None)
            terrain_levels = getattr(terrain, "terrain_levels", None)
            if terrain_levels is None:
                terrain_gate_open = False
            else:
                terrain_gate_open = bool(
                    torch.mean(terrain_levels.float()).item() >= self.cfg.terrain_gate_min_mean_level
                )
        self.metrics["terrain_gate_open"][:] = float(terrain_gate_open)
        if not terrain_gate_open:
            return

        if self.cfg.curriculum_update_max_terrain_level is not None:
            terrain_levels = self._env.scene.terrain.terrain_levels[valid_env_ids]
            eligible = terrain_levels <= self.cfg.curriculum_update_max_terrain_level
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

        commands, bin_ids, source_ids = self.curriculum.sample(
            len(env_ids),
            self._stage,
            (
                self.cfg.frontier_sampling_probability,
                self.cfg.active_sampling_probability,
                self.cfg.replay_sampling_probability,
            ),
            self.cfg.frontier_bin_count,
        )
        self.vel_command_b[env_ids] = commands
        self.bin_ids[env_ids] = bin_ids
        self.metrics["sample_frontier"][env_ids] = (source_ids == 0).float()
        self.metrics["sample_active_uniform"][env_ids] = (source_ids == 1).float()
        self.metrics["sample_low_speed_replay"][env_ids] = (source_ids == 2).float()
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
    curriculum_update_max_terrain_level: int | None = None
    """Highest terrain level whose successful segments may expand command bins."""
