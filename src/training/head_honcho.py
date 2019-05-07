import tensorflow as tf
import numpy as np
from memory import Memory
from model import Model, update_target_graph
from trainer import Trainer
import params
import json
import sys

tf.reset_default_graph()

# Create the model object and saver
model = Model(params.BOARD_WIDTH, params.BOARD_HEIGHT, "model")
target = Model(params.BOARD_WIDTH, params.BOARD_HEIGHT, "target")
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize the model parameters
    if (len(sys.argv) > 1 and sys.argv[1] == '-restore'):
        saver.restore(sess, "./models/model.ckpt")
        print("Model Restored")
        print()
    else:
        sess.run(tf.global_variables_initializer())

    # Create the memory
    memory = Memory(params.MEMORY_SIZE)
    # Initialize the trainer
    trainer = Trainer(sess, model, target, memory)
    # Set target vars equal to actual model vars
    sess.run(update_target_graph("model", "target"))

    # Initialize other vars
    tau = 0
    illegal_moves = []
    legal_moves = []

    for episode in range(params.NUM_EPISODES):
        print("Episode: {}".format(episode))
        gm = trainer.gen_game(episode)
        gm.play(trainer)
        # update tau step count
        tau += gm.turn

        # Logging
        legal_count, illegal_count = gm.players[0].legal_moves, gm.players[0].illegal_moves
        print("Illegal Moves: " + str(illegal_count))
        print("Legal Moves: " + str(legal_count))
        if (illegal_count != 0 or legal_count != 0):
            illegal_moves.append(illegal_count)
            legal_moves.append(legal_count)
        print("Number of Turns: {}".format(gm.turn))
        print("Episode Finished")

        if tau > params.MAX_TAU:
            # Update target vars equal to actual model vars
            sess.run(update_target_graph("model", "target"))
            tau = 0
            print("######## Target Model Updated ########")

        # Save model/memory/legal moves every 50 episodes
        if (episode % 50 == 0 and episode != 0):
            move_ratio = np.array(legal_moves) / (np.array(legal_moves) + np.array(illegal_moves))

            np.savetxt("legal_moves.csv", move_ratio, delimiter=",")
            saver.save(sess, "./models/model.ckpt")
            memory.save_memory()

            print("######## Model Saved ########")

        print()

        # # Save game every 50 episodes
        # if episode % 100 == 0:
        #     with open("../../resources/replays/{}.txt".format(episode), "w") as f:
        #         f.write(json.dumps(gm.logger.output()))
        #     print ("Saved Replay")
        #     print()