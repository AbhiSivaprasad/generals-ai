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
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# %%
from src.environment.environment import GeneralsEnvironment
from src.agents.random_agent import RandomAgent
from src.training.dqn.dqn import DQN
from src.training.input import (
    get_input_channel_dimension_size,
    convert_action_index_to_action,
)
from src.training.dqn.replay_memory import ReplayMemory, Transition

# %%
gym.register(
    id="Generals-v0",
    entry_point=GeneralsEnvironment,
    nondeterministic=True,
    kwargs={"players": [RandomAgent(0), RandomAgent(1)]},
)

# %%
env = gym.make("Generals-v0")

# %%
# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# %%
# if GPU is to be used
device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")

# %%
BATCH_SIZE = 128  # replay buffer sample size
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000  # higher means slower exponential decay
TAU = 0.005  # update rate of target network
LR = 1e-4

# %%
N_ROWS = 3
N_COLUMNS = 3
FOG_OF_WAR = False
INPUT_CHANNELS = get_input_channel_dimension_size(FOG_OF_WAR)
N_ACTIONS = env.action_space.n

# %%
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# %%
policy_net = DQN(N_ROWS, N_COLUMNS, INPUT_CHANNELS, N_ACTIONS).to(device)
target_net = DQN(N_ROWS, N_COLUMNS, INPUT_CHANNELS, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())

# %%
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


# %%
steps_done = 0


# %%
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # argmax returns the indices of the maximum values along the specified dimension
            return policy_net(state).argmax(dim=1).squeeze()
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        ).squeeze()


# %%
episode_durations = []


# %%
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# %%
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

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


# %%
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# %%
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    num_players = len(env.players)
    state = torch.tensor(state, dtype=torch.float32, device=device)
    for t in count():
        raw_actions = [select_action(state[j]) for j in range(num_players)]
        actions = [
            convert_action_index_to_action(action.item(), n_columns=N_COLUMNS)
            for action in raw_actions
        ]
        observation, rewards, terminated, truncated, _ = env.step(actions)
        rewards = torch.tensor(rewards, device=device)
        done = terminated or truncated

        # next state is none if the game is terminated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)

        # Store the transitions in memory
        for j in range(num_players):
            memory.push(state[j], raw_actions[j], next_state[j], rewards[j])

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

# %%
print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()

# %%

# %%
