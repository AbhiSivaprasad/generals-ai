import random
import tensorflow as tf
import numpy as np
from src.graphics.constants import *
from src.training.model import *
from src.move import Move


class DeepGeneral:
    def __init__(self, model, sess, eps=0):
        self._eps = eps
        self._model = model
        self._sess = sess

        # metrics
        self.legal_moves = 0
        self.illegal_moves = 0

    def move(self, board):
        
        # First get the legal moves
        moves = list(board.legal_moves)
        
        assert(len(moves) > 0)
        
        # e-greedy approach
        if random.random() < self._eps:
            return random.choice(moves)
        else:
            
            # get the predicted action
            action = np.argmax(self._model.predict_one(convert_board(board), self._sess))
           
            move = self.convert_action(action)

            if move in moves:
                self.legal_moves += 1
                return move
            else:
                # print("illegal move")
                self.illegal_moves += 1
                return random.choice(moves)
    
    def convert_action(self, action):
        y = action // (params.BOARD_WIDTH * 4)
        x = (action - (y * params.BOARD_WIDTH * 4)) // 4
        dr = action - ((y * params.BOARD_WIDTH * 4) + x * 4)
        
        move = Move(x,y, x + DIRECTIONS[dr][0], y + DIRECTIONS[dr][1])
        
        return move
