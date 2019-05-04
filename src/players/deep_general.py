import random
import tensorflow as tf

class DeepGeneral:
    def __init__(self, model, eps=0):
        self._eps = eps

    def move(self, board):
        
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        
        moves = list(board.legal_moves)
        assert(len(moves) > 0)
        if random.random() < self._eps:
            return random.choice(moves)
        else:
            action = np.argmax(self._model.predict_one(state, self._sess))
            
            y = action // (self.board_width * 4)
            x = (action - (y * self.board_width * 4)) // 4
            dr = action - ((y * self.board_width * 4) + x * 4)
            
            move = Move(x,y, x + directions[dr][0], y + directions[dr][1])
    
            return move
