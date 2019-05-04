import random
import tensorflow as tf
from src.graphics.constants import *


class DeepGeneral:
    def __init__(self, model, eps=0):
        self._eps = eps

    def move(self, board):
        
        # First get the legal moves
        moves = list(board.legal_moves)
        
        assert(len(moves) > 0)
        
        # e-greedy approach
        if random.random() < self._eps:
            return random.choice(moves)
        else:
            
            # get the predicted action
            action = np.argmax(self._model.predict_one(state, self._sess))
           
            move = self.convert_action(action)
            
            if move in moves:
                return move
            else:
                return random.choice(moves)
    
    def convert_action(action):
        y = action // (board.cols * 4)
        x = (action - (y * board.cols * 4)) // 4
        dr = action - ((y * board.cols * 4) + x * 4)
        
        move = Move(x,y, x + DIRECTIONS[dr][0], y + DIRECTIONS[dr][1])
        
        return move
