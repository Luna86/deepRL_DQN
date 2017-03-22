"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor
from collections import deque


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=3):
        self.history_length = history_length
        self.initial_list = [np.zeros([84,84])] * history_length
        self.queue = deque(self.initial_list, maxlen=history_length)
        pass

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""

        all_states = list(self.queue) + [state]
        self.queue.append(state)
        processed_states = np.array(all_states, dtype=np.float32)
        processed_states = np.swapaxes(processed_states, 0, 2)
        processed_states = np.swapaxes(processed_states, 0, 1)
        return processed_states

    def reset(self):
        self.queue = deque(self.initial_list, maxlen=self.history_length)
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        return

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size
        pass

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        im = Image.fromarray(state,'RGB')
        im = im.convert('L')
        im = im.resize(self.new_size, Image.BILINEAR)
        processed_state = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[0], im.size[1])
        return processed_state

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        im = Image.fromarray(state,'RGB')
        im = im.convert('L')
        im = im.resize(self.new_size, Image.BILINEAR)
        processed_state = np.array(im.getdata(), dtype=np.float32).reshape(im.size[0], im.size[1])
        return processed_state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        processed_samples = []
        for i, sample in enumerate(samples):
            processed_sample = sample
            processed_sample.state = np.array(sample.state, dtype=np.float32)
            processed_sample.next_state = np.array(sample.next_state, dtype=np.float32)
            processed_samples.append(processed_sample)

        return processed_samples

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        processed_reward = 0
        if reward > 0:
            processed_reward = 1
        elif reward < 0:
            processed_reward = -1
        return processed_reward


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    #preprocessors: a list to preprocessors to call
    def __init__(self, preprocessors):
        self.atari = preprocessors[0]
        self.history = preprocessors[1]

    def reset(self):
        self.history.reset()

    def process_state_for_network(self, state):
        state = self.atari.process_state_for_network(state)
        return self.history.process_state_for_network(state)
    def process_reward(self, reward):
        return self.atari.process_reward(reward)