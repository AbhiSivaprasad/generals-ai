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
from src.training.utils import convert_agent_dict_to_tensor
from collections import defaultdict
from itertools import count
import matplotlib.pyplot as plt
from pathlib import Path
import re

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### Eval checkpoints against RandomAgent

# %%
N_ROWS = 2
N_COLUMNS = 2

# %%
LOG_DIR = Path("./resources/replays/vs_random/")
CHECKPOINT_DIR = Path("./resources/checkpoints")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# %%
def eval_checkpoint(checkpoint_path: Path, log_dir: Path, num_games: int = 100):
    model = DQN.load_checkpoint(checkpoint_path, device=device)
    agent = CuriousGeorgeAgent(model, train=False)
    random_agent = RandomAgent(player_index=0)
    env = GeneralsEnvironment(
        players=[random_agent, agent],
        board_x_size=N_COLUMNS,
        board_y_size=N_ROWS,
    )

    metrics = []
    for i in range(num_games):
        logger = Logger()
        metrics.append(simulate_game(env, logger))
        logger.write(log_dir / f"{i}.json")

    # compute overall stats
    overall_metrics = {}
    overall_metrics["duration"] = np.mean([m[1]["duration"] for m in metrics])
    overall_metrics["reward"] = np.mean([m[1]["reward"] for m in metrics])
    overall_metrics["win_rate"] = np.mean([m[1]["won"] for m in metrics])
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
            agent_name: agent.move(state[agent_name], env)
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
            metrics[env.unwrapped.player_index_by_agent_name[agent_name]]["reward"] = (
                metrics[env.unwrapped.player_index_by_agent_name[agent_name]].get(
                    "reward", 0
                )
                + reward
            )

        if done:
            terminal_status = {
                env.unwrapped.player_index_by_agent_name[agent_name]: main_reward * 0.5
                + 0.5
                for agent_name, main_reward in env.unwrapped.get_main_rewards().items()
            }
            for name, status in terminal_status.items():
                metrics[name]["won"] = status
                metrics[name]["duration"] = t
            return metrics


# %%
metrics = eval_checkpoint(
    checkpoint_path=CHECKPOINT_DIR / "checkpoint_50.pth", log_dir=LOG_DIR, num_games=100
)

# %%
metrics


# %%
def get_checkpoint_paths(directory):
    return sorted(
        [f for f in directory.glob("checkpoint_*.pth")],
        key=lambda x: int(re.findall(r"\d+", x.stem)[-1]),
    )


def evaluate_checkpoints(checkpoint_paths, num_games=100):
    results = []
    for path in checkpoint_paths:
        print(f"Evaluating {path}")
        metrics = eval_checkpoint(
            checkpoint_path=path, log_dir=LOG_DIR, num_games=num_games
        )
        step = int(re.findall(r"\d+", path.stem)[-1])
        metrics["step"] = step
        results.append(metrics)
    return results


def plot_metrics(results):
    metrics = ["duration", "reward", "win_rate"]
    fig, axs = plt.subplots(
        len(metrics), 1, figsize=(10, 5 * len(metrics)), sharex=True
    )

    for i, metric in enumerate(metrics):
        steps = [r["step"] for r in results]
        values = [r[metric] for r in results]
        axs[i].plot(steps, values, marker="o")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_title(f"{metric.capitalize()} vs Checkpoint")
        axs[i].grid(True)

    axs[-1].set_xlabel("Checkpoint Step")
    plt.tight_layout()
    plt.show()


# %%
checkpoint_paths = get_checkpoint_paths(CHECKPOINT_DIR)
results = evaluate_checkpoints(checkpoint_paths)
plot_metrics(results)
