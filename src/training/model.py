import tensorflow as tf

class Model:
    def __init__(self, eps=0, bw=None, bh=None):
        self._board_width = bw
        self._board_height = bh
        self._num_actions = self._board_width * self._board_height + 4
        self._num_conv_layers = 4

        self._states = None
        self._logits = None
        self._optimizer = None
        self.var_init = None

        self._create_model()

    def _add_conv_layer(self, input_tensor, filters, kernal_size, i):
        # Architecture found from alpha zero. Does a conv layer followed by batch normal folled by relu
        conv = tf.layers.conv2d(input_tensor, filters=filters, kernel_size=kernal_size,
                         strides=1, padding="SAME", name="conv" + str(i+1))
        norm = tf.layers.batch_normalization(conv, training=self.is_training, name="norm" + str(i+1))
        relu = tf.nn.relu(norm, name="relu" + str(i+1))
        return relu

    def _create_model(self):
        # Still need to decide input dimension exactly, 3 is a random number of layers of the board. TBD
        self._states = tf.placeholder(shape=[None, self._board_width, self._board_height, 3], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, _num_actions], dtype=tf.float32)

        current_input = self._states
        for i in range(self._num_conv_layers):
            current_input = self._add_conv_layer(current_input, filters=64, kernal_size=3, i)
        conv_output = self._add_conv_layer(current_input, filters=2, kernal_size=1, self._num_conv_layers+1)

        self._logits = tf.layers.dense(conv_output, self._num_actions, activation=tf.nn.relu)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: 
            state.reshape(1, self._board_width, self._board_height, 3)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})
