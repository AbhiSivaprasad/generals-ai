import random


class RandomPlayer:
    def move(self, board):
        moves = list(board.legal_moves)
        assert(len(moves) > 0)
        return random.choice(moves)
