#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import *
from deeprl_hw2.policy import *


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int) #84 x 84 greyscale image from preprocessing step
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    input = Input(shape=input_shape, batch_shape=None, name='input', dtype=K.floatx(), sparse=False, tensor=None)
    with tf.name_scope('hidden1'):
        hidden1 = Dense(100, activation=None)(input)
    with tf.name_scope('output'):
        output = Dense(num_actions, activation='softmax')(hidden1)

    model = Model(inputs=input, outputs=output)
    print(model.summary())

    return model



def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on given game environment')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--target_update_freq', default=10000, type=int, help='interval between two updates of the target network')
    parser.add_argument('--num_burn_in', default=10, type=int, help='number of samples to be filled into the replay memory before updating the network')
    parser.add_argument('--train_freq', default=1, type=int, help='How often to update the Q-network')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--num_iterations', default=10000, type=int, help='num of iterations to run for the training')
    parser.add_argument('--max_episode_length', default=10000, type=int, help='max length of one episode')

    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)
    game_env = gym.make(args.env)

    num_actions = game_env.action_space.n
    input_shape=(84, 84)

    #setup model
    model = create_model(window=1, input_shape=input_shape, num_actions=num_actions, model_name='linear model')

    #setup optimizer
    optimizer = Adam(lr=0.001)

    #setup preprocessor
    atari_preprocessor = AtariPreprocessor(input_shape)
    history_preprocessor = HistoryPreprocessor(history_length=3)
    preprocessor = PreprocessorSequence([atari_preprocessor, history_preprocessor])

    #setup policy
    policy = UniformRandomPolicy(num_actions=num_actions)

    #setup DQN agent
    agent = DQNAgent(q_network=model, preprocessor=preprocessor, memory=None, policy=policy, gamma=args.gamma, target_update_freq=args.target_update_freq,
                     num_burn_in=args.num_burn_in, train_freq=args.train_freq, batch_size=args.batch_sizes)
    agent.compile(optimizer=optimizer, loss_func=mean_huber_loss)
    agent.fit(env=game_env, num_iterations=args.num_iterations, max_episode_length=args.max_episode_length)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
