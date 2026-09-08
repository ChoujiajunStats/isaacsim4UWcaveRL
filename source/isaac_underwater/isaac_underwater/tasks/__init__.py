"""Gym registration kept free of simulator imports for launcher compatibility."""

import gymnasium as gym


gym.register(
    id="Isaac-Underwater-PointNav-Direct-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.rsl_rl_ppo_cfg:UnderwaterPointNavPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-PointNav-IMU-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavImuEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.rsl_rl_ppo_cfg:UnderwaterPointNavPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-PointNav-Perception-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavPerceptionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.rsl_rl_ppo_cfg:UnderwaterPointNavPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-PointNav-Stereo-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavStereoEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.rsl_rl_ppo_cfg:UnderwaterPointNavPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-Cave-Perception-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterCavePerceptionEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.rsl_rl_ppo_cfg:UnderwaterPointNavPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-Cave-Stereo-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterCaveStereoEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.rsl_rl_ppo_cfg:UnderwaterPointNavPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-Cave-VisualPilot-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterCaveVisualPilotEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.visual_ppo_cfg:UnderwaterVisualPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-Cave-Explore-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterCaveExploreEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.visual_ppo_cfg:UnderwaterExplorePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-Cave-Entry-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_underwater.tasks.direct.pointnav_env:UnderwaterCaveEntryEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.visual_ppo_cfg:UnderwaterCaveEntryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Underwater-Cave-Navigation-v0",
    entry_point="isaac_underwater.tasks.direct.pointnav_env:UnderwaterPointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaac_underwater.tasks.direct.pointnav_env:UnderwaterMultiCaveNavigationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaac_underwater.tasks.direct.agents.visual_ppo_cfg:"
            "UnderwaterMultiCaveNavigationPPORunnerCfg"
        ),
    },
)
