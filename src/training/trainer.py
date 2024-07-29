import math
import numpy as np
from src.board_generator import BoardGenerator
from src.game_master import GameMaster
from src.graphics.board import Board
from src.logger import Logger
from src.players.deep_general import DeepGeneral

import params

class Trainer:
    def __init__(self, sess, model, target, memory, max_steps=math.inf):
        self._sess = sess
        self._model = model
        self._target = target

        self._memory = memory
        self._temp_memory = None

        self._eps = params.MAX_EPS
        self._decay_step = 0
        self.max_steps = max_steps

        self._bg = BoardGenerator()
    

    def step(self, state, action, next_state, player_id, terminal):
        # Create and add temporary SAS to memory, so we can add R later based on player_id
        SAS = (state, action, player_id, next_state, terminal)
        self._temp_memory.append(SAS)

        # Only want to train when we take a step with our player
        # otherwise we're training twice as much as we'd want to
        if (player_id != 0): return
        # Don't want to train until we have enough examples
        if (self._memory.low_size()):
            if (terminal):
                print("Still filling memory...")
            return

        # Decay epsilon
        self._decay_eps()
        self._update(terminal)

    def _update(self, terminal):
        # Sample and train from actual memory
        samples = self._memory.sample(params.BATCH_SIZE)
        states_mb, actions_mb, rewards_mb, next_states_mb, terminal_mb = map(np.array, zip(*samples))

        # Get Q values for next_state
        # Double DQN 
        next_state_Qs = self._target.predict_batch(next_states_mb, self._sess)
        next_actions = np.argmax(self._model.predict_batch(next_states_mb, self._sess), axis=1)
        max_next_state_Qs = next_state_Qs[np.arange(len(next_state_Qs)), next_actions]
        # Vanilla DQN
        # max_next_state_Qs = np.amax(self._model.predict_batch(next_states_mb, self._sess), axis=1)

        # We invert the terminal so that if it IS terminal, then we're multiply by 0
        # Since if the next_state is a terminal state, the target is just the reward
        targets_mb = rewards_mb + (1. - terminal_mb) * params.GAMMA * max_next_state_Qs

        loss = self._model.train_batch(states_mb, targets_mb, actions_mb, self._sess)

        if (terminal):
            print("Loss: " + str(loss) + ", Eps: " + str(self._eps))


    def gen_game(self, episode_number):
        self._temp_memory = []
        board = self._bg.generate_board_state(params.BOARD_WIDTH, params.BOARD_HEIGHT)

        logger = Logger(num_players=2)

        p1 = DeepGeneral(self._model, self._sess, self._eps)
        p2 = DeepGeneral(self._model, self._sess, self._eps)

        return GameMaster(board, players=[p1, p2], logger=logger)

    def convert_temp_memory(self, winner):
        SARS = [(state, action, 1, next_state, t) if winner == player_id
            else (state, action, -1, next_state, t)
            for (state, action, player_id, next_state, t) in self._temp_memory]
        self._memory.add_samples(SARS)

    def _decay_eps(self):
        self._eps = params.MIN_EPS + \
            (params.MAX_EPS - params.MIN_EPS) * np.exp(-params.DECAY_RATE * self._decay_step)
        self._decay_step += 1
