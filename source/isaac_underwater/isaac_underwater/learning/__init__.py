"""Project-specific learning modules and RSL-RL registration helpers."""

from .visual_actor_critic import StereoVisualActorCriticRecurrent, register_rsl_rl_extensions

__all__ = ["StereoVisualActorCriticRecurrent", "register_rsl_rl_extensions"]
