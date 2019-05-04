import tensorflow as tf
import numpy as np
import params


class Model:
    def __init__(self, bw=None, bh=None):
        self._board_width = bw
        self._board_height = bh
        self._num_actions = self._board_width * self._board_height * 4

        self._input_states = None
        self._actions = None
        self._target_Q = None

        self._output = None
        self._loss = None
        self._optimizer = None
        self.var_init = None

        self._create_model()

    def _add_conv_layer(self, input_tensor, filters, kernal_size, i):
        # Architecture found from alpha zero. Does a conv layer followed by batch normal folled by relu
        conv = tf.layers.conv2d(input_tensor,
            filters=filters,
            kernel_size=kernal_size,
            kernal_initializer=tf.contrib.layers.variance_scaling_initializer(),
            padding="VALID",
            name="conv" + str(i+1))
        norm = tf.layers.batch_normalization(conv, training=self.is_training, name="norm" + str(i+1))
        relu = tf.nn.relu(norm, name="relu" + str(i+1))
        return relu

    def _create_model(self):
        self._input_states = tf.placeholder(shape=[None, self._board_width, self._board_height, params.INPUT_DEPTH],
            name="input_state", dtype=tf.float32)
        self._actions = tf.placeholder(tf.float32, [None, self._num_actions], name="actions", dtype=tf.float32)
        self._target_Q = tf.placeholder(shape=[None], name="target_Q", dtype=tf.float32)

        current_input = self._states
        for i in range(params.NUM_CONV_LAYERS):
            current_input = self._add_conv_layer(current_input, filters=64, kernal_size=3, i)
        conv_output = self._add_conv_layer(current_input, filters=2, kernal_size=1, params.NUM_CONV_LAYERS+1)

        flattened_output = tf.layers.flatten(conv_output)
        fc1 = tf.layers.dense(flattened_output,
            units = 512,
            activation = tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            name="fc1")

        self._output = tf.layers.dense(fc1,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            units = self._num_actions,
            activation=None)

        # Mask the output to only the Q value of the action taken
        pred_Q = tf.reduce_sum(tf.multiply(self.output, self._actions), axis=1)

        self._loss = tf.reduce_mean(tf.square(self.target_Q - pred_Q))
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        states = state.reshape(1, self._board_width, self._board_height, params.INPUT_DEPTH)
        return self.predict_batch(state, sess)

    def predict_batch(self, states, sess):
        return sess.run(self._output, feed_dict={ self._input_states: states })

    def train_batch(self, x_batch, y_batch, action_batch, sess):
        loss, _ = sess.run([self._loss, self._optimizer],
            feed_dict={ self._input_states: x_batch,
                        self._actions: action_batch,
                        self._target_Q: y_batch })
        return loss

def convert_board(board):
    
    state = np.zeros(self._board_width, self._board_width, params.INPUT_DEPTH)
    
    # If it's a terminal state we dont want that shit
    if (next_board.terminal_status() != -1):
        return state
    
    our_player = board.player_index
    opponent = 1 - our_player
    
    for i in range(board.rows):
        for j in range(board.cols):
            
            tile = board.grid[i][j]
      
            # the temporary storage for a tile
            temp = np.zeros(8)

            temp[0] = int(tile.type == TILE_OBSTACLE and not tile.is_city and not tile.is_mountain)
            temp[1] = int(tile.type != TILE_OBSTACLE and tile.type != TILE_FOG)
            temp[2] = int(tile.is_city)
            temp[3] = int(tile.is_mountain)
            temp[4] = int(tile.is_general)
            temp[5] = int(tile.type == opponent)
            temp[6] = int(tile.type == our_player)
            temp[7] = tile.army

            output[i,j,:] = temp
    
    return state
       
def convert_move(move):
    
    # TODO
    action = None
    return action






