import tensorflow as tf
from memory import Memory
from model import Model
from trainer import Trainer
import params

# Create the model object and saver
model = Model(params.BOARD_WIDTH, params.BOARD_HEIGHT)
saver = tf.train.Saver()

with tf.Session() as sess:

    # Create the memory
    memory = Memory(params.MEMORY_SIZE)

    # Initialize the model parameters
    sess.run(model.var_init)

    # Initialize the trainer
    trainer = Trainer(sess, model, memory)

    for episode in range(params.NUM_EPISODES):
        gm = trainer.gen_game(episode)
        gm.play(trainer)

        # Save model every 5 episodes
        if episode % 5 == 0:
            save_path = saver.save(sess, "./models/model.ckpt")
            print("Model Saved")

        # Save game every 50 episodes
        if episode % 50 == 0:
            with open("../../resources/replays/{}.txt".format(episode), "w") as f:
                f.write(json.dumps(gm.logger.output()))