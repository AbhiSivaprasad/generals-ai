from pathlib import Path
import unittest

import gymnasium as gym
import numpy as np

from src.agents.random_agent import RandomAgent
from src.environment.environment import GeneralsEnvironment
import argparse

from src.models.dqn_cnn import DQN
from src.training.step import optimize_step
from src.utils.replay_buffer import ListBuffer, ReplayBuffer

import torch
from torch import nn
from torch.optim import AdamW

import src.test.probe_envs

BATCH_SIZE = 256
BOARD_SIZE = 3
BOARD_CHANNELS = 1
BUFFER_SIZE = 10_000
GAMMA = lambda p: 0.0 if p in [1, 2, 4, 5] else 0.9
N_ACTIONS = \
    lambda p: \
        {
            1: 1, 
            2: 1, 
            3: 1, 
            4: 2,
            5: 2,
            6: 2
        }[p]

board_shape = (BOARD_SIZE, BOARD_SIZE, BOARD_CHANNELS)

def probe1_assertion(env, dqn):
    obs = torch.zeros(board_shape, dtype=torch.float32).to(device="cuda").unsqueeze(0)
    with torch.no_grad():
        # print("[INFO] Observation:", obs)
        q_val = dqn(obs).cpu().detach().numpy().flatten()[0]
        # print(q_values)
    
    assert abs(q_val - 1.0) < 1e-3, f"Expected QValue 1.0, got {q_val}"
    
def probe2_assertion(env, dqn):
    obs = torch.ones((2, *board_shape), dtype=torch.float32).to(device="cuda")
    obs[1, :, :, :] *= -1
    with torch.no_grad():
        # print("[INFO] Observation:", obs)
        q_val = dqn(obs).cpu().detach().numpy().flatten()[:2]
        # print(q_values)
    
    assert abs(q_val[0] - 1.0) < 1e-3, f"Expected QValue 1.0, got {q_val[0]}"
    assert abs(q_val[1] - (-1.0)) < 1e-3, f"Expected QValue -1.0, got {q_val[1]}"

def probe3_assertion(env, dqn):
    obs = torch.ones((2, *board_shape), dtype=torch.float32).to(device="cuda")
    obs[0, :, :, :] = 0.0
    with torch.no_grad():
        # print("[INFO] Observation:", obs)
        q_val = dqn(obs).cpu().detach().numpy().flatten()[:2]
        # print(q_values)
    
    assert abs(q_val[0] - GAMMA(3)) < 1e-3, f"Expected QValue 0.9, got {q_val[0]}"
    assert abs(q_val[1] - 1.0) < 1e-3, f"Expected QValue 1.0, got {q_val[1]}"
    
def probe4_assertion(env, dqn):
    obs = torch.zeros(board_shape, dtype=torch.float32).to(device="cuda").unsqueeze(0)
    with torch.no_grad():
        # print("[INFO] Observation:", obs)
        q_val = dqn(obs).cpu().detach().numpy().flatten()
        # print(q_values)
    
    assert np.argmax(q_val) == 1, f"Expected better action to be 1, qvalues: {q_val}"
    
def probe5_assertion(env, dqn):
    obs = torch.ones((2, *board_shape), dtype=torch.float32).to(device="cuda")
    obs[0, :, :, :] *= 0.0
    with torch.no_grad():
        # print("[INFO] Observation:", obs)
        q_val = dqn(obs).cpu().detach().numpy()
        # print(q_val)
    
    assert np.argmax(q_val[0]) == 0, f"Expected better action to be 0, qvalues: {q_val[0]}"
    assert np.argmax(q_val[1]) == 1, f"Expected better action to be 1, qvalues: {q_val[1]}"
    assert abs(q_val[0][0] - 1.0) < 1e-3, f"Expected q_val 1: {q_val[0][0]}"
    assert abs(q_val[1][1] - 1.0) < 1e-3, f"Expected q_val 1: {q_val[1][1]}"
    
def probe6_assertion(env, dqn):
    obs = torch.ones((2, *board_shape), dtype=torch.float32).to(device="cuda")
    obs[0, :, :, :] *= 0.0
    with torch.no_grad():
        # print("[INFO] Observation:", obs)
        q_val = dqn(obs).cpu().detach().numpy()
        # print(q_val)
    
    assert np.argmax(q_val[0]) == 0, f"Expected better action to be 0, qvalues: {q_val[0]}"
    assert np.argmax(q_val[1]) == 1, f"Expected better action to be 1, qvalues: {q_val[1]}"
    assert abs(q_val[0][0] - 1.9) < 1e-3, f"Expected q_val 1.9: {q_val[0][0]}"
    assert abs(q_val[1][1] - 1.9) < 1e-3, f"Expected q_val 1.9: {q_val[1][1]}"


def get_action(probe, obs, env, dqn):
    if probe in [1, 2, 3]:
        return 0
    elif probe in [4, 5, 6]:
        return 0 if np.random.random() < 0.5 else 1
    else:
        raise ValueError("Invalid probe")

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=int, default=1, help="Probe env #")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbosity")
    args = parser.parse_args()
    probe = args.probe
    
    buffer: ReplayBuffer = ListBuffer(BUFFER_SIZE, 0)
        
    env = gym.make(f"probe{probe}", n_rows=BOARD_SIZE, n_cols=BOARD_SIZE, n_channels=BOARD_CHANNELS)
    env.reset(seed=0)
    dqn = DQN(BOARD_CHANNELS, N_ACTIONS(probe), BOARD_SIZE, BOARD_SIZE).to(device="cuda")
    optimizer = AdamW(dqn.parameters(), lr=3e-4)
    
    gamma = GAMMA(probe)
    
    for i in range(1_000):
        obs, _ = env.reset()
        done = False
        while not done:
            action = get_action(probe, obs, env, dqn)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buffer.add([((obs, action, reward, next_obs, done), 0)])
            obs = next_obs
            
            if buffer.size() > 2 * BATCH_SIZE:
                data = buffer.sample(BATCH_SIZE)
                experiences, steps = tuple(map(list, zip(*data)))
                loss, step_info = optimize_step(dqn, dqn, optimizer, experiences, gamma)
                predicted_q_vals = step_info["predicted_q_vals"]
                if args.verbose:
                    print(
                        loss.item(), 
                        predicted_q_vals.min().item(), 
                        predicted_q_vals.max().item(), 
                        predicted_q_vals.mean().item(), 
                        predicted_q_vals.std().item()
                    )
    
    
    assert_fn = globals()[f"probe{probe}_assertion"]
    assert_fn(env, dqn)
    
    print()
    print()
    print("--------------------")
    print(f"Probe {probe} passed!")
    print("--------------------")
    print()
    
