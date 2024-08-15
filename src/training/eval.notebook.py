# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: generals
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from src.agents.random_agent import RandomAgent
from src.agents.curiousgeorge_agent import CuriousGeorgeAgent
from src.training.dqn.dqn import DQN
from src.environment.logger import Logger
from src.environment.environment import GeneralsEnvironment

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### Eval checkpoints against RandomAgent

# %%
N_ROWS = 2
N_COLUMNS = 2
env = GeneralsEnvironment(
    players=[],
    board_x_size=N_COLUMNS,
    board_y_size=N_ROWS,
)

# %%
CHECKPOINT_DIR = Path("./resources/checkpoints")


# %%
def eval_checkpoint(checkpoint_path: Path, log_dir: Path, num_games: int = 100):
    model = DQN.load_checkpoint(checkpoint_path, device=device)
    agent = CuriousGeorgeAgent(model, train=False)
    random_agent = RandomAgent(player_index=0)
    env.players = [random_agent, agent]

    metrics = []
    for i in range(num_games):
        logger = Logger()
        metrics.append(simulate_game(env, logger))

    # compute overall stats
    overall_metrics = {}
    overall_metrics["duration"] = np.mean(m[1]["duration"] for m in metrics)
    overall_metrics["reward"] = np.mean(m[1]["reward"] for m in metrics)
    overall_metrics["win_rate"] = np.mean(m[1]["won"] for m in metrics)
    return overall_metrics


def simulate_game(env, logger: Logger):
    """Simulate a game between players and return the total reward collected, duration, and which player won"""
    # Initialize the environment and get its state
    state, _ = env.reset(logger=logger)
    convert_agent_dict_to_tensor(state, device=device)
    num_players = len(env.unwrapped.players)
    metrics = defaultdict(dict)
    for t in count():
        actions = {
            agent_name: agent.move(state[agent_name], env).item()
            for agent_name, agent in zip(
                env.unwrapped.agent_name_by_player_index, env.unwrapped.players
            )
        }
        state, rewards, terminated, truncated, info = env.step(actions)
        convert_agent_dict_to_tensor(state, device=device)
        convert_agent_dict_to_tensor(actions, dtype=torch.long, device=device)
        truncated = list(truncated.values())[0]
        terminated = list(terminated.values())[0]
        done = terminated or truncated

        # update metrics
        for agent_name, reward in rewards.items():
            metrics[agent_name]["reward"] = (
                metrics[agent_name].get("reward", 0) + reward
            )
            metrics[agent_name]["duration"] = metrics[agent_name].get("duration", 0) + 1

        if done:
            terminal_status = {
                env.unwrapped.player_index_by_agent_name[agent_name]: main_reward * 0.5
                + 0.5
                for agent_name, main_reward in env.unwrapped.get_main_rewards().items()
            }
            metrics["won"] = terminal_status


# %%
eval_checkpoint()
