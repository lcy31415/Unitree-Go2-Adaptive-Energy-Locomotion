"""Stairs-focused flood-fill curriculum task with a flat-terrain side wing.

Terrain families: 45% stairs_up, 45% stairs_down, 10% flat. Each family owns
one joint (level x |vx| x |yaw|) grid whose cells flood-fill outward from the
easy corner as mastery accumulates. The flat wing keeps its 0-5 m/s / 0-5 rad/s
axes. Stair commands are forward-only at 0.2-1.5 m/s with exactly zero yaw.
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_pie_env_cfg import (
    AdaptiveEnergyPIEEnvCfg,
    AdaptiveEnergyPIEEventsCfg,
    AdaptiveEnergyPIESceneCfg,
)
from .adaptive_energy_pie_terrain_cfg import (
    ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
    ADAPTIVE_ENERGY_PIE_NUM_LEVELS,
    ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES,
)


# flat, stairs_up, stairs_down, slope_up, slope_down, random_rough, obstacles
FLOODFILL_FAMILY_WEIGHTS = (0.10, 0.45, 0.45, 0.0, 0.0, 0.0, 0.0)
FLOODFILL_ACTIVE_FAMILIES = (0, 1, 2)

# Five forward-speed bins cover the complete 0.2--1.5 m/s range at every
# terrain level. Stair yaw is no longer a curriculum axis: its single bin is
# sampled as exactly zero, while load_state_dict migrates old 10 x 5 x 5 state.
_STAIR_VX_AXES = tuple(0.2 + (1.3 / 5.0) * index for index in range(6))
_STAIR_YAW_AXES = (0.0, 1.0e-6)
_FLAT_VX_AXES = tuple(round(0.5 * index, 1) for index in range(11))  # 0.0 .. 5.0
_FLAT_YAW_AXES = tuple(round(0.5 * index, 1) for index in range(11))  # 0.0 .. 5.0

FLOODFILL_GRIDS = {
    "flat": {
        "levels": 1,
        "vx_edges": _FLAT_VX_AXES,
        "yaw_edges": _FLAT_YAW_AXES,
        "tracking_tolerances": (0.10, 0.15, 0.15),
        "activation_neighbors": 1,
    },
    "stairs_up": {
        "levels": ADAPTIVE_ENERGY_PIE_NUM_LEVELS,
        "vx_edges": _STAIR_VX_AXES,
        "yaw_edges": _STAIR_YAW_AXES,
        "tracking_tolerances": (0.25, 0.25, 0.30),
        # Traversal is the primary stair criterion. Tracking remains a relaxed
        # hold condition so a policy cannot unlock cells by merely sprinting
        # across the course while ignoring its command.
        "success_tracking_fraction": 0.45,
        "minimum_progress": 3.2,
        "forward_only": True,
        "zero_yaw_probability": 1.0,
        "activation_neighbors": 1,
    },
    "stairs_down": {
        "levels": ADAPTIVE_ENERGY_PIE_NUM_LEVELS,
        "vx_edges": _STAIR_VX_AXES,
        "yaw_edges": _STAIR_YAW_AXES,
        "tracking_tolerances": (0.25, 0.25, 0.30),
        "success_tracking_fraction": 0.45,
        "minimum_progress": 3.2,
        "forward_only": True,
        "zero_yaw_probability": 1.0,
        "activation_neighbors": 1,
    },
}


@configclass
class AdaptiveEnergyStairsPIEFloodFillCommandsCfg:
    """Episode-fixed commands assigned by the flood grid."""

    base_velocity = mdp.FloodFillVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.FloodFillVelocityCommandCfg.Ranges(
            # The broad range is used only as UniformVelocityCommand metadata;
            # FloodGridFamily supplies the actual family-specific commands.
            lin_vel_x=(-5.0, 5.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-5.0, 5.0),
        ),
    )


@configclass
class AdaptiveEnergyStairsPIEFloodFillCurriculumCfg:
    flood_grid = CurrTerm(
        func=mdp.FloodGridCurriculum,
        params={
            "command_name": "base_velocity",
            "terrain_family_names": ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES,
            "active_family_indices": FLOODFILL_ACTIVE_FAMILIES,
            "columns_per_family": ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
            "family_grids": FLOODFILL_GRIDS,
        },
    )


@configclass
class AdaptiveEnergyStairsPIEFloodFillEnvCfg(AdaptiveEnergyPIEEnvCfg):
    """Stairs + flat flood-fill PIE locomotion."""

    scene: AdaptiveEnergyPIESceneCfg = AdaptiveEnergyPIESceneCfg(num_envs=512, env_spacing=2.5)
    commands: AdaptiveEnergyStairsPIEFloodFillCommandsCfg = AdaptiveEnergyStairsPIEFloodFillCommandsCfg()
    curriculum: AdaptiveEnergyStairsPIEFloodFillCurriculumCfg = AdaptiveEnergyStairsPIEFloodFillCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.allocate_terrain_families.params["family_weights"] = FLOODFILL_FAMILY_WEIGHTS

        self.events.reset_base.params["pose_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.radians(5.0), math.radians(5.0)),
        }
        self.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class AdaptiveEnergyStairsPIEFloodFillPlayEnvCfg(AdaptiveEnergyStairsPIEFloodFillEnvCfg):
    """Small deterministic visualization configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(FLOODFILL_ACTIVE_FAMILIES)
        # One deterministic environment for flat, stairs-up and stairs-down.
        self.events.allocate_terrain_families.params["family_weights"] = (
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.events.push_robot = None
        self.curriculum = None
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
