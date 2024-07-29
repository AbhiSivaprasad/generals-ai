import tensorflow as tf
from memory import Memory
from model import Model
from trainer import Trainer
import params
import json



with tf.Session() as sess:

  model = Model(params.BOARD_WIDTH, params.BOARD_HEIGHT)

  new_saver = tf.train.import_meta_graph('./models/model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

  # Create the memory
  memory = Memory(params.MEMORY_SIZE)

  # Initialize the model parameters
  sess.run(model.var_init)

  # Initialize the trainer
  trainer = Trainer(sess, model, memory)

  for episode in range(params.NUM_EPISODES):
    print("Episode: {}".format(episode))
    gm = trainer.gen_game(episode)

    gm.play(trainer)

    print("Number of Turns: {}".format(gm.turn))

    # Save model every 5 episodes
    if episode % 5 == 0:
      save_path = new_saver.save(sess, "./models/model")
      print("Model Saved")

      # Save game every 50 episodes
      if episode % 100 == 0:
        with open("../../resources/replays/{}.txt".format(episode), "w") as f:
          f.write(json.dumps(gm.logger.output()))
          print ("Saved Replay")