from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def _adaptive_reward_weights(
    command: torch.Tensor,
    transition_speed: float = 1.7,
    energy_weight_decay: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return speed-dependent ``(Rlin, Rang, Renergy)`` weights.

    Above ``transition_speed``, the energy weight decreases by
    ``energy_weight_decay`` per additional m/s. All released energy weight is
    transferred to linear-velocity completion; the angular weight remains at
    0.2. Clamping the energy weight at zero keeps all weights non-negative and
    their sum one.
    """
    command_speed = torch.linalg.norm(command[:, :2], dim=1)
    released_weight = torch.clamp(
        energy_weight_decay * (command_speed - transition_speed), min=0.0, max=0.4
    )
    linear_weight = 0.4 + released_weight
    angular_weight = torch.full_like(linear_weight, 0.2)
    energy_weight = 0.4 - released_weight
    return linear_weight, angular_weight, energy_weight


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def adaptive_energy_tracking_lin_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    tracking_sigma: float = 0.25,
    transition_speed: float = 1.7,
    energy_weight_decay: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Original dynamically weighted planar velocity-tracking reward.

    This matches ``corl_rewards.py``: the summed squared planar error is
    divided by the fixed ``tracking_sigma`` rather than command magnitude.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    squared_error = torch.square(asset.data.root_lin_vel_b[:, :2] - command[:, :2])
    raw_reward = torch.exp(-torch.sum(squared_error, dim=1) / tracking_sigma)
    linear_weight, _, _ = _adaptive_reward_weights(command, transition_speed, energy_weight_decay)
    return linear_weight * raw_reward


def adaptive_energy_tracking_ang_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    tracking_sigma_yaw: float = 0.25,
    transition_speed: float = 1.7,
    energy_weight_decay: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Original dynamically weighted yaw-rate tracking reward."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    squared_error = torch.square(asset.data.root_ang_vel_b[:, 2] - command[:, 2])
    raw_reward = torch.exp(-squared_error / tracking_sigma_yaw)
    _, angular_weight, _ = _adaptive_reward_weights(command, transition_speed, energy_weight_decay)
    return angular_weight * raw_reward


def adaptive_energy_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    energy_sigma_lin: float = 1000.0,
    energy_sigma_ang: float = 500.0,
    energy_clip_lin: float = 0.2,
    energy_clip_rot: float = 0.2,
    transition_speed: float = 1.7,
    energy_weight_decay: float = 0.3,
) -> torch.Tensor:
    """Dynamically weighted distance-averaged energy-efficiency reward."""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    base_lin_vel = robot.data.root_lin_vel_b
    base_ang_vel = robot.data.root_ang_vel_b

    power = torch.sum(torch.abs(joint_vel) * torch.abs(torque), dim=1)
    divider_lin = energy_sigma_lin * torch.clamp(torch.abs(base_lin_vel[:, 0]), min=energy_clip_lin)
    divider_ang = energy_sigma_ang * torch.clamp(torch.abs(base_ang_vel[:, 2]), min=energy_clip_rot)
    raw_reward = torch.exp(-power / (divider_lin + divider_ang))
    _, _, energy_weight = _adaptive_reward_weights(command, transition_speed, energy_weight_decay)
    return energy_weight * raw_reward


class adaptive_energy_reward_residual(ManagerTermBase):
    """Nonlinear attenuation residual of the adaptive-energy reward.

    Isaac Lab sums reward terms after multiplying each by the control time step.
    Linear tracking, angular tracking, and energy efficiency are registered as
    separate terms for diagnostics. This term returns the remaining nonlinear
    attenuation, making their sum exactly

    ``(w_lin(v_cmd) * R_lin + w_ang(v_cmd) * R_ang + w_energy(v_cmd) * R_energy) * exp(-R_aux)``.

    The implementation follows ``go1_gym/envs/rewards/corl_rewards.py`` and the
    active coefficients in ``AdaptiveGo1Config``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot: Articulation = env.scene["robot"]
        num_joints = robot.data.joint_pos.shape[1]
        num_actions = env.action_manager.total_action_dim

        self._previous_joint_vel = torch.zeros(env.num_envs, num_joints, device=env.device)
        self._previous_target = torch.zeros(env.num_envs, num_actions, device=env.device)
        self._previous_previous_target = torch.zeros_like(self._previous_target)
        self._previous_previous_action = torch.zeros_like(self._previous_target)
        self._last_foot_contacts: torch.Tensor | None = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_vel[env_ids] = 0.0
        self._previous_target[env_ids] = 0.0
        self._previous_previous_target[env_ids] = 0.0
        self._previous_previous_action[env_ids] = 0.0
        if self._last_foot_contacts is not None:
            self._last_foot_contacts[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        feet_asset_cfg: SceneEntityCfg,
        feet_sensor_cfg: SceneEntityCfg,
        collision_sensor_cfg: SceneEntityCfg,
        tracking_sigma: float = 0.25,
        tracking_sigma_yaw: float = 0.25,
        energy_sigma_lin: float = 1000.0,
        energy_sigma_ang: float = 500.0,
        energy_clip_lin: float = 0.2,
        energy_clip_rot: float = 0.2,
        transition_speed: float = 1.7,
        energy_weight_decay: float = 0.3,
        sigma_rew_neg: float = 0.02,
    ) -> torch.Tensor:
        robot: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
        command = env.command_manager.get_command(command_name)

        joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
        joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
        torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b

        # Motion rewards, Eq. (3).
        lin_squared_error = torch.square(base_lin_vel[:, :2] - command[:, :2])
        tracking_lin = torch.exp(-torch.sum(lin_squared_error, dim=1) / tracking_sigma)
        yaw_squared_error = torch.square(base_ang_vel[:, 2] - command[:, 2])
        tracking_ang = torch.exp(-yaw_squared_error / tracking_sigma_yaw)

        # Distance-averaged energy reward, Eq. (4), including the numerical
        # clamps used by the released implementation.
        power = torch.sum(torch.abs(joint_vel) * torch.abs(torque), dim=1)
        divider_lin = energy_sigma_lin * torch.clamp(torch.abs(base_lin_vel[:, 0]), min=energy_clip_lin)
        divider_ang = energy_sigma_ang * torch.clamp(torch.abs(base_ang_vel[:, 2]), min=energy_clip_rot)
        energy_reward = torch.exp(-power / (divider_lin + divider_ang))

        linear_weight, angular_weight, energy_weight = _adaptive_reward_weights(
            command, transition_speed, energy_weight_decay
        )

        # Fixed auxiliary penalties from AdaptiveGo1Config.
        lin_vel_z = torch.square(base_lin_vel[:, 2])
        ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        orientation = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
        torques = torch.sum(torch.square(torque), dim=1)
        dof_vel = torch.sum(torch.square(joint_vel), dim=1)
        dof_acc = torch.sum(torch.square((self._previous_joint_vel[:, asset_cfg.joint_ids] - joint_vel) / env.step_dt), dim=1)

        lower_violation = -(joint_pos - robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]).clip(max=0.0)
        upper_violation = (joint_pos - robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]).clip(min=0.0)
        dof_pos_limits = torch.sum(lower_violation + upper_violation, dim=1)

        current_action = env.action_manager.action
        previous_action = env.action_manager.prev_action
        action_rate = torch.sum(torch.square(current_action - previous_action), dim=1)
        action_term = env.action_manager.get_term("JointPositionAction")
        current_target = action_term.processed_actions
        smoothness_1 = torch.sum(
            torch.square(current_target - self._previous_target) * (previous_action != 0), dim=1
        )
        smoothness_2 = torch.sum(
            torch.square(current_target - 2 * self._previous_target + self._previous_previous_target)
            * (previous_action != 0)
            * (self._previous_previous_action != 0),
            dim=1,
        )

        forces = contact_sensor.data.net_forces_w
        foot_contact = forces[:, feet_sensor_cfg.body_ids, 2] > 1.0
        if self._last_foot_contacts is None:
            self._last_foot_contacts = torch.zeros_like(foot_contact)
        filtered_contact = torch.logical_or(foot_contact, self._last_foot_contacts)
        foot_vel_xy = robot.data.body_lin_vel_w[:, feet_asset_cfg.body_ids, :2]
        feet_slip = torch.sum(filtered_contact * torch.sum(torch.square(foot_vel_xy), dim=2), dim=1)

        collision_forces = forces[:, collision_sensor_cfg.body_ids, :]
        collisions = torch.sum((torch.linalg.norm(collision_forces, dim=-1) > 0.1).float(), dim=1)

        negative_aux = (
            -0.02 * lin_vel_z
            -0.001 * ang_vel_xy
            -0.04 * feet_slip
            -5.0 * collisions
            -10.0 * dof_pos_limits
            -0.0001 * torques
            -0.0001 * dof_vel
            -2.5e-7 * dof_acc
            -0.1 * smoothness_1
            -0.1 * smoothness_2
            -0.01 * action_rate
            -5.0 * orientation
        )

        weighted_lin = linear_weight * tracking_lin
        weighted_ang = angular_weight * tracking_ang
        weighted_energy = energy_weight * energy_reward
        positive_reward = weighted_lin + weighted_ang + weighted_energy
        total_reward = positive_reward * torch.exp(negative_aux * env.step_dt / sigma_rew_neg)

        # Update the state after evaluating the current transition.
        self._previous_joint_vel[:] = robot.data.joint_vel
        self._previous_previous_target[:] = self._previous_target
        self._previous_target[:] = current_target
        self._previous_previous_action[:] = previous_action
        self._last_foot_contacts[:] = foot_contact

        # Motion and energy terms are separate RewardTerms for diagnostics and
        # curriculum accounting. Their sum with this residual is unchanged.
        return (
            total_reward
            - weighted_lin
            - weighted_ang
            - weighted_energy
        )


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
