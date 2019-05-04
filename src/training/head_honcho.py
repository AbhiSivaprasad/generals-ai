import tensorflow as tf
from memory import Memory
from model import Model
from trainer import Trainer
import params

saver = tf.train.Saver()

with tf.Session() as sess:
    # Create the memory
    memory = Memory(params.MEMORY_SIZE)

    # Create the model object and initialize the model parameters
    model = Model(params.BOARD_WIDTH, params.BOARD_HEIGHT)
    model.var_init.eval()

    # Initialize the trainer
    trainer = Trainer(sess, model, memory)

    for episode in range(NUM_EPISODES):
        gm = trainer.gen_game(episode)
        gm.play(trainer)

        # Save model every 5 episodes
        if episode % 5 == 0:
            save_path = saver.save(sess, "./models/model.ckpt")
            print("Model Saved")