import random
import tensorflow as tf

class DeepGeneral:
    def __init__(self, model, eps=0):
        self._eps = eps

    def move(self, board):
        moves = list(board.legal_moves)
        assert(len(moves) > 0)
        if random.random() < self._eps:
            return random.choice(moves)
        else:
            # Have to fix this logic to properly interpret output of the model
            return np.argmax(self._model.predict_one(state, self._sess))