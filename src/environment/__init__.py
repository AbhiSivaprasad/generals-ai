from gymnasium.envs.registration import register

from src.environment.environment import GeneralsEnvironment

register(
     id="generals-v0",
     entry_point=GeneralsEnvironment,
     max_episode_steps=1000,
)