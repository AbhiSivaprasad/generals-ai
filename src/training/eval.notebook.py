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
import re, os
from typing import List
from tqdm import tqdm
from src.utils.scheduler import ConstantHyperParameterSchedule
from src.environment.board import Board
from src.environment.tile import TileType, Tile
from src.training.input import convert_state_to_array, convert_tile_to_array
from src.environment.action import Action, Direction

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model = DQN.load_checkpoint(
    Path("/home/ubuntu/generals-ai/resources/checkpoints/checkpoint_1200.pth")
).to(device)

# %% [markdown]
# ### Probe probability

# %% [markdown]
# Handcraft a board state with an optimal move, and track model's performance over time on board state

# %%
board = Board(num_rows=2, num_cols=2)

# %%
board_state = [
    [(TileType.GENERAL, 0, 1), (TileType.NORMAL, None, None)],
    [(TileType.NORMAL, 0, 4), (TileType.GENERAL, 1, 1)],
]

# %%
grid = []
for y, row_state in enumerate(board_state):
    row_tiles = []
    for x, tile_state in enumerate(row_state):
        tile = Tile(board, x, y)
        tile.type = tile_state[0]
        tile.player_index = tile_state[1]
        tile.army = tile_state[2]
        row_tiles.append(tile)

        if tile.type == TileType.GENERAL:
            board.generals[tile.player_index] = tile
        elif tile.type == TileType.CITY:
            board.cities[tile.player_index].append(tile)

    grid.append(row_tiles)

# %%
board.set_grid(grid)

# %%
state = convert_state_to_array(board, 2, False)

# %%
state = torch.tensor(state, dtype=torch.float32, device=device)

# %%
optimal_action = Action(do_nothing=False, startx=0, starty=1, direction=Direction.RIGHT)
optimal_action_index = optimal_action.to_index(n_columns=board.num_cols)

# %%
state_action_values = model(state)[0]
state_action_values = state_action_values / state_action_values.norm()

# %%
state_action_values[optimal_action_index].item()

# %% [markdown]
# ### Eval checkpoints against each other

# %%
N_ROWS = 2
N_COLUMNS = 2


# %%
def simulate_game(env, logger: Logger):
    """Simulate a game between players and return the total reward collected, duration, and which player won"""
    # Initialize the environment and get its state
    state, info = env.reset(logger=logger)
    convert_agent_dict_to_tensor(state, device=device)
    num_agents = len(env.unwrapped.agents)
    metrics = defaultdict(dict)
    for t in count():
        actions = {
            agent_index: agent.move(state[agent_index], env)
            for agent_index, agent in enumerate(env.unwrapped.agents)
        }
        observation, rewards, terminated, truncated, info = env.step(actions)
        convert_agent_dict_to_tensor(rewards, device=device)
        convert_agent_dict_to_tensor(actions, dtype=torch.long, device=device)
        truncated = list(truncated.values())[0]
        terminated = list(terminated.values())[0]
        done = terminated or truncated

        # update metrics
        for agent_index, reward in rewards.items():
            metrics[agent_index]["reward"] = (
                metrics[agent_index].get("reward", 0) + reward.item()
            )

        if done:
            terminal_status = {
                agent_index: main_reward * 0.5 + 0.5
                for agent_index, main_reward in env.unwrapped.get_main_rewards().items()
            }
            for agent_index, status in terminal_status.items():
                metrics[agent_index]["won"] = status
                metrics[agent_index]["duration"] = t
            return metrics


# %%
def get_checkpoint_numbers(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")

    pattern = re.compile(r"checkpoint_(\d+)\.pth")
    checkpoint_numbers = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            checkpoint_numbers.append(number)

    checkpoint_numbers.sort()
    return checkpoint_numbers


# %%
def eval_checkpoints(
    checkpoint_dir: Path,
    checkpoint_numbers: List[int],
    base_log_dir: Path,
    num_games: int = 100,
):
    models = [
        DQN.load_checkpoint(
            checkpoint_dir / f"checkpoint_{checkpoint_number}.pth", device=device
        )
        for checkpoint_number in checkpoint_numbers
    ]
    metrics_by_checkpoint = defaultdict(lambda: defaultdict(list))
    for i in tqdm(range(len(models))):
        for j in range(i + 1, len(models)):
            agent1 = CuriousGeorgeAgent(
                models[i],
                epsilon_schedule=ConstantHyperParameterSchedule(0.1),
            )
            agent2 = CuriousGeorgeAgent(
                models[j],
                epsilon_schedule=ConstantHyperParameterSchedule(0.1),
            )
            env = GeneralsEnvironment(
                agents=[agent1, agent2],
                board_x_size=N_COLUMNS,
                board_y_size=N_ROWS,
            )
            log_dir = (
                base_log_dir / f"{checkpoint_numbers[i]}_vs_{checkpoint_numbers[j]}"
            )
            log_dir.mkdir(parents=True, exist_ok=True)

            metrics = []
            for k in range(num_games):
                logger = Logger()
                metrics.append(simulate_game(env, logger))
                logger.write(log_dir / f"{k}.json")

            # compute overall stats
            metrics_by_checkpoint[checkpoint_numbers[i]]["duration"].extend(
                [m[0]["duration"] for m in metrics]
            )
            metrics_by_checkpoint[checkpoint_numbers[i]]["reward"].extend(
                [m[0]["reward"] for m in metrics]
            )
            metrics_by_checkpoint[checkpoint_numbers[i]]["win_rate"].extend(
                [m[0]["won"] for m in metrics]
            )
            metrics_by_checkpoint[checkpoint_numbers[j]]["duration"].extend(
                [m[1]["duration"] for m in metrics]
            )
            metrics_by_checkpoint[checkpoint_numbers[j]]["reward"].extend(
                [m[1]["reward"] for m in metrics]
            )
            metrics_by_checkpoint[checkpoint_numbers[j]]["win_rate"].extend(
                [m[1]["won"] for m in metrics]
            )

    overall_metrics = defaultdict(dict)
    for checkpoint_number, metrics in metrics_by_checkpoint.items():
        for metric_name, values in metrics.items():
            overall_metrics[checkpoint_number][metric_name] = np.mean(values)

    return overall_metrics


# %%
checkpoint_dir = Path("./resources/checkpoints")
checkpoint_numbers = get_checkpoint_numbers(checkpoint_dir)
checkpoint_numbers = [c for c in checkpoint_numbers if c % 4000 == 0]

# %%
overall_metrics = eval_checkpoints(
    checkpoint_dir=checkpoint_dir,
    checkpoint_numbers=checkpoint_numbers,
    base_log_dir=Path("./resources/replays/tournament/"),
    num_games=15,
)

# %%
overall_metrics

# %% [markdown]
# ### Eval checkpoints against RandomAgent

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
        agents=[random_agent, agent],
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

# %%
