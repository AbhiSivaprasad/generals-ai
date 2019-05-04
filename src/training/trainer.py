import json

from src.board_generator import BoardGenerator
from src.game_master import GameMaster
from src.graphics.board import Board
from src.logger import Logger
from src.players.deep_general import DeepGeneral

import params

class Trainer:
    def __init__(self, sess, model, memory):
        self._sess = sess
        self._model = model

        self._memory = memory
        self._temp_memory = None

        self._eps = params.MAX_EPS
        self._decay_step = 0

        self._bg = BoardGenerator()
    

    def step(self, state, action, next_state, player_id, terminal):
        # Decay epsilon
        self._decay_eps()

        # Create and add temporary SAS to memory, so we can add R later based on player_id
        SAS = (state, action, player_id, next_state, terminal)
        self._temp_memory.append(SAS)

        # Only want to train when we take a step with our player
        # otherwise we're training twice as much as we'd want to
        if (player_id != 0): return

        # Sample and train from actual memory
        mini_batch = self._memory.sample(params.BATCH_SIZE)
        states_mb = np.array([b[0] for b in mini_batch])
        actions_mb = np.array([b[1] for b in mini_batch])
        next_states_mb = np.array([b[3] for b in mini_batch])


        # Get Q values for next_state 
        next_Q = self._model.predict_batch(next_states_mb, self._sess)

        target_mb = [r if terminal else r + params.GAMMA * np.max(next_Q)
            for (_, _, r, _, terminal), next_Q in zip(mini_batch, next_Q)]
        
        loss = self._model.train_batch(states_mb, targets_mb, actions_mb)

        if (terminal):
            print("Loss: " + str(loss))
            print("Episode Finished")
            print()


    def gen_game(self, episode_number):
        print("Starting Episode " + str(episode_number) + "...")
        self._temp_memory = []
        board = bg.generate_board_state(params.BOARD_WIDTH, params.BOARD_HEIGHT)
        logger = Logger(num_players=2)

        p1 = DeepGeneral(self._model, self._eps)
        p2 = DeepGeneral(self._model, self._eps)
        return GameMaster(board, players=[p1, p2], logger=logger)

    def convert_temp_memory(self, winner):
        SARS = [(state, action, 1, next_state, t) if winner == player_id
            else (state, action, -1, next_state, t)
            for (state, action, player_id, next_state, t) in self._temp_memory]
        self._memory.add_samples(SARS)

    def _decay_eps(self):
        self._eps = params.MIN_EPS + 
            (params.MAX_EPS - params.MIN_EPS) * np.exp(-params.DECAY_RATE * self._decay_step)
        self._decay_step += 1
