import tensorflow as tf
import numpy as np
from memory import Memory
from model import Model
from trainer import Trainer
import params
import json

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
    illegal_moves = []
    legal_moves = []

    for episode in range(params.NUM_EPISODES):
        print("Episode: {}".format(episode))
        gm = trainer.gen_game(episode)

        gm.play(trainer)
        legal_count, illegal_count = gm.players[0].legal_moves, gm.players[0].illegal_moves
        print("Illegal Moves: " + str(illegal_count))
        print("Legal Moves: " + str(legal_count))
        if (illegal_count != 0 or legal_count != 0):
            illegal_moves.append(illegal_count)
            legal_moves.append(legal_count)
        print("Number of Turns: {}".format(gm.turn))
        print("Episode Finished")
        print()

        # Save model every 50 episodes
        if (episode % 50 == 0):
            move_ratio = np.array(legal_moves) / (np.array(legal_moves) + np.array(illegal_moves))
            np.savetxt("legal_moves.csv", move_ratio, delimiter=",")
            save_path = saver.save(sess, "./models/model")
            print("Model Saved")
            print()

        # # Save game every 50 episodes
        # if episode % 100 == 0:
        #     with open("../../resources/replays/{}.txt".format(episode), "w") as f:
        #         f.write(json.dumps(gm.logger.output()))
        #     print ("Saved Replay")
        #     print()