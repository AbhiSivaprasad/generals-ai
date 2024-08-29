# -*- coding: utf-8 -*-
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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import wandb
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from pathlib import Path

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# %%
from src.environment.environment import GeneralsEnvironment
from src.environment.probes.probe0 import ProbeZeroEnvironment
from src.agents.random_agent import RandomAgent
from src.agents.curiousgeorge_agent import CuriousGeorgeAgent
from src.training.models.dqn.dqn import DQN
from src.training.models.fc_network import FCNetwork
from src.training.input import (
    get_input_channel_dimension_size,
)
from src.training.models.dqn.replay_memory import ReplayMemory, Transition
from src.environment.logger import Logger
from src.training.utils import convert_agent_dict_to_tensor
from src.utils.scheduler import ExponentialHyperParameterSchedule
from src.utils.utils import delete_directory_contents
from src.environment.action import Action
from src.environment.probes.probe1 import ProbeOneEnvironment
from src.environment.probes.probe2 import ProbeTwoEnvironment
from src.environment.probes.probe3 import ProbeThreeEnvironment

# %%
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
BATCH_SIZE = 128  # replay buffer sample size
GAMMA = 0
EPS_START = 0.9
EPS_END = 0
EPS_DECAY = 5000  # higher means slower exponential decay
TAU = 0.005  # update rate of target network
LR = 1e-4

# %%
N_ROWS = 2
N_COLUMNS = 2
FOG_OF_WAR = False
NORMAL_TILE_INCREMENT_FREQUENCY = 2
INPUT_CHANNELS = get_input_channel_dimension_size(FOG_OF_WAR)

# %%
# DQN params
N_HIDDEN_CONV_LAYERS = 0
N_HIDDEN_CONV_CHANNELS = 32
KERNEL_SIZE = 2

# FC params
N_HIDDEN_LAYERS = 3
N_HIDDEN_DIMENSION = 128

# %%
N_ACTIONS = 4 * N_ROWS * N_COLUMNS + 1
AUXILIARY_REWARD_WEIGHT = 0.1


# %%
def get_dqn_network():
    return DQN(
        n_rows=N_ROWS,
        n_columns=N_COLUMNS,
        kernel_size=KERNEL_SIZE,
        input_channels=INPUT_CHANNELS,
        n_actions=N_ACTIONS,
        n_hidden_conv_layers=N_HIDDEN_CONV_LAYERS,
        n_hidden_conv_channels=N_HIDDEN_CONV_CHANNELS,
    ).to(device)


def get_fc_network():
    return FCNetwork(
        n_rows=N_ROWS,
        n_columns=N_COLUMNS,
        n_actions=N_ACTIONS,
        n_input_channels=INPUT_CHANNELS,
        n_hidden_dim=N_HIDDEN_DIMENSION,
        n_hidden_layers=N_HIDDEN_LAYERS,
    ).to(device)


def get_network(type: str):
    if type == "dqn":
        return get_dqn_network()
    elif type == "fc":
        return get_fc_network()
    else:
        raise ValueError("Invalid network type")


# %%
def get_model_params(model_type):
    if model_type == "dqn":
        return {
            "kernel_size": KERNEL_SIZE,
            "n_hidden_conv_layers": N_HIDDEN_CONV_LAYERS,
            "n_hidden_conv_channels": N_HIDDEN_CONV_CHANNELS,
        }
    elif model_type == "fc":
        return {
            "n_hidden_dim": N_HIDDEN_DIMENSION,
            "n_hidden_layers": N_HIDDEN_LAYERS,
        }
    else:
        raise ValueError("Invalid network type")


# %%
model_type = "fc"
policy_net = get_network(model_type)
target_net = get_network(model_type)
target_net.load_state_dict(policy_net.state_dict())

# %%
# env = GeneralsEnvironment(
env = ProbeThreeEnvironment(
    agents=[
        CuriousGeorgeAgent(
            0,
            policy_net=policy_net,
            epsilon_schedule=ExponentialHyperParameterSchedule(
                initial_value=EPS_START, final_value=EPS_END, decay_rate=EPS_DECAY
            ),
        ),
        # CuriousGeorgeAgent(
        #     policy_net=policy_net,
        #     epsilon_schedule=ExponentialHyperParameterSchedule(
        #         initial_value=EPS_START, final_value=EPS_END, decay_rate=EPS_DECAY
        #     ),
        # ),
        RandomAgent(1),
    ],
    board_x_size=N_COLUMNS,
    board_y_size=N_ROWS,
    auxiliary_reward_weight=AUXILIARY_REWARD_WEIGHT,
    normal_tile_increment_frequency=NORMAL_TILE_INCREMENT_FREQUENCY,
)

# %%
run = wandb.init(
    project="generals",
    config={
        "learning_rate": LR,
        "n_rows": N_ROWS,
        "n_columns": N_COLUMNS,
        "tau": TAU,
        "gamma": GAMMA,
        "batch_size": BATCH_SIZE,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay": EPS_DECAY,
        "normal_tile_increment_frequency": NORMAL_TILE_INCREMENT_FREQUENCY,
        **get_model_params(model_type),
    },
)

# %%
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# %%
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# %%
steps_done = 0


# %%
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    # look up our prediction for value of next state if its not a final state
    non_final_next_states = [s for s in batch.next_state if s is not None]
    if len(non_final_next_states) == 0:
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
    else:
        non_final_next_states = torch.stack(non_final_next_states)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                target_net(non_final_next_states).max(1).values
            )

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # metrics
    loss = loss.detach().item()
    mean_reward = reward_batch.abs().mean().item()
    qvalue_magnitude = state_action_values.abs().mean().item()
    return loss, mean_reward, qvalue_magnitude


# %%
LOG_DIR = Path("resources/replays")
LOG_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR = Path("resources/checkpoints")
LOG_DIR.mkdir(exist_ok=True, parents=True)

# delete old logs and checkpoints
delete_directory_contents(LOG_DIR)
delete_directory_contents(CHECKPOINT_DIR)

# %%
num_episodes = 50000
log_interval = 200
checkpoint_interval = 200
global_step = 0
for i_episode in range(num_episodes):
    logger = Logger()
    # Initialize the environment and get its state
    state, info = env.reset(logger=logger)
    convert_agent_dict_to_tensor(state, device=device)
    num_agents = len(env.unwrapped.agents)
    for t in count():
        actions_with_info = {
            agent_index: agent.move(state[agent_index], env)
            for agent_index, agent in enumerate(env.unwrapped.agents)
        }
        actions = {
            agent_index: action
            for agent_index, (action, _) in actions_with_info.items()
        }
        agent_infos = {
            agent_index: agent_info
            for agent_index, (_, agent_info) in actions_with_info.items()
        }

        # check whether agent 0 took a legal move before taking the action
        _, action_info = actions_with_info[0]
        is_action_legal = env.unwrapped.game_master.board.is_action_valid(
            Action.from_index(
                action_info["best_action"], n_columns=env.unwrapped.board_x_size
            ),
            player_index=0,
        )

        # take action
        print(agent_infos)
        observation, rewards, terminated, truncated, info = env.step(actions, agent_infos)
        convert_agent_dict_to_tensor(rewards, device=device)
        convert_agent_dict_to_tensor(actions, dtype=torch.long, device=device)
        truncated = list(truncated.values())[0]
        terminated = list(terminated.values())[0]
        done = terminated or truncated

        # next state is none if the game is terminated
        if terminated:
            next_state = {agent_name: None for agent_name in state.keys()}
        else:
            convert_agent_dict_to_tensor(observation, device=device)
            next_state = observation

        # Store the transitions in memory
        for agent_name in state.keys():
            memory.push(
                state[agent_name],
                actions[agent_name],
                next_state[agent_name],
                rewards[agent_name],
            )

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        metrics = optimize_model()
        if metrics is not None:
            minibatch_loss, minibatch_reward_magnitude, minibatch_qvalue_magnitude = (
                metrics
            )
            wandb.log(
                {
                    "loss": minibatch_loss,
                    "reward_magnitude": minibatch_reward_magnitude,
                    "qvalue_magnitude": minibatch_qvalue_magnitude,
                },
                step=global_step,
            )

        # log other metrics
        wandb.log(
            {
                "legal_move": int(is_action_legal),
                "epsilon": env.unwrapped.agents[0].epsilon,
            },
            step=global_step,
        )

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        global_step += 1

        if done:
            wandb.log({"duration": t, "episode": i_episode}, step=global_step)
            if i_episode % checkpoint_interval == 0:
                policy_net.save_checkpoint(CHECKPOINT_DIR, i_episode)
            if i_episode % log_interval == 0:
                logger.write(LOG_DIR / f"{i_episode}.json")
            break

# %%
# !find resources/replays -mindepth 1 -print0 | xargs -0 -P $(nproc) rm -rf

# %%
