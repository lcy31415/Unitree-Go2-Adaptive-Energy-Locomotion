import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.adaptive_energy_env_cfg:AdaptiveEnergyEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.adaptive_energy_env_cfg:AdaptiveEnergyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:AdaptiveEnergyPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-Flat-PIE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_flat_pie_env_cfg:AdaptiveEnergyFlatPIEEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_flat_pie_env_cfg:AdaptiveEnergyFlatPIEPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.pie_cfg:AdaptiveEnergyFlatPIERunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-Flat-LPACRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_flat_lpacrl_env_cfg:AdaptiveEnergyFlatLPACRLEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_flat_lpacrl_env_cfg:AdaptiveEnergyFlatLPACRLPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "AdaptiveEnergyFlatLPACRLPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-Flat-LPACRL-PIE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_flat_lpacrl_pie_env_cfg:AdaptiveEnergyFlatLPACRLPIEEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_flat_lpacrl_pie_env_cfg:AdaptiveEnergyFlatLPACRLPIEPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.pie_cfg:AdaptiveEnergyFlatLPACRLPIERunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-Terrain-LPACRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_terrain_lpacrl_env_cfg:AdaptiveEnergyTerrainLPACRLEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_terrain_lpacrl_env_cfg:AdaptiveEnergyTerrainLPACRLPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "AdaptiveEnergyTerrainLPACRLPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-LPACRL-PIE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_lpacrl_pie_env_cfg:AdaptiveEnergyLPACRLPIEEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_lpacrl_pie_env_cfg:AdaptiveEnergyLPACRLPIEPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.pie_cfg:AdaptiveEnergyLPACRLPIERunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-PIE-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_pie_stairs_env_cfg:AdaptiveEnergyPIEStairsEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_pie_stairs_env_cfg:AdaptiveEnergyPIEStairsPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.pie_cfg:AdaptiveEnergyPIEStairsRunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-stairs-PIE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_stairs_pie_env_cfg:AdaptiveEnergyStairsPIEEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_stairs_pie_env_cfg:AdaptiveEnergyStairsPIEPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.pie_cfg:AdaptiveEnergyStairsPIERunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-stairs-PIE-FloodFill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_stairs_pie_floodfill_env_cfg:AdaptiveEnergyStairsPIEFloodFillEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_stairs_pie_floodfill_env_cfg:AdaptiveEnergyStairsPIEFloodFillPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.pie_cfg:AdaptiveEnergyStairsPIEFloodFillRunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-Adaptive-Energy-PIE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_pie_env_cfg:AdaptiveEnergyPIEEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.adaptive_energy_pie_env_cfg:AdaptiveEnergyPIEPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.pie_cfg:AdaptiveEnergyPIERunnerCfg"
        ),
    },
)
