import json

from src.board_generator import BoardGenerator
from src.game_master import GameMaster
from src.graphics.board import Board
from src.logger import Logger
from src.players.deep_general import DeepGeneral


for i in range(1):
    bg = BoardGenerator()
    board = bg.generate_board_state(15, 15)

    print(board.serialize())

    logger = Logger(num_players=2)

    game_master = GameMaster(board, players=[RandomPlayer(), RandomPlayer()], logger=logger)
    game_master.play()

    with open("../../resources/replays/temp.txt", "w") as f:
        f.write(json.dumps(logger.output()))


class Trainer:
    def __init__(self, sess, model, memory, max_eps, min_eps):
        self._sess = sess
        self._model = model
        self._best_model = model
        self._memory = memory
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._eps = max_eps
        self._bg = BoardGenerator()

    def run(self):
        iterations_without_imp = 0
        while True:
            self._play_game_batch(50)
            win_pct = self._play_game_batch(10, True)
            # If wins more than 50% of games, should replace the best one
            if (win_pct > 0.5):
                self._best_player = self._player
                iterations_without_imp = 0
            else:
                iterations_without_imp += 1

            # If model can't beat best_model after certain amount of time, should stop training
            if (iterations_without_imp > 20):
                print("EARLY STOPPING")
                break

    def _play_game_batch(self, num_games, evaluate=False):
        wins = 0
        for i in range(num_games):
            board = bg.generate_board_state(15, 15)
            logger = Logger(num_players=2)

            p1 = DeepGeneral(self._model, self._eps)
            p2 = DeepGeneral(self._best_model, self._eps)
            game_master = GameMaster(board, players=[p1, p2], logger=logger)
            game_master.play()

            if logger.winner() == 1:
                wins += 1
            if not evaluate:
                SARS = logger.getSARS()
                self._memory.add_samples(SARS)
                self._replay()
                
        return wins/num_games

