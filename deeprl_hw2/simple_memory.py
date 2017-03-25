from deeprl_hw2.core import ReplayMemory
import numpy as np

class Sample:
    def __init__(self, state, action, reward, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.is_terminal = is_terminal


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def tail_idx(self):
        return (self.start + self.length - 1) % self.maxlen

    def get(self, index):
        if index >= self.length:
            raise IndexError("")
        return self.data[(self.start+index) %  self.maxlen]

    def get_last_K_index(self, index, K):
        klist = [index]
        for i in range(min(K, self.length)-1):
            if klist[i] == self.start:
                break
            idx = (klist[i]-1)%self.maxlen
            klist.append(idx)
        return klist


class SimpleReplayMemory(ReplayMemory):
    def __init__(self, max_size, window_length):
        self.buffer = RingBuffer(max_size)
        self.window_length = window_length
        self.max_size = max_size

    def append(self, state, action, reward):
        self.buffer.append(Sample(state=state, action=action, reward=reward,  is_terminal=0))
        # self.is_terminal_buffer.append(False)

    def end_episode(self, final_state, is_terminal):
        # frame = final_state[:, :, -1]
        self.buffer.append(Sample(state=final_state, action=0, reward=0, is_terminal=1))
        # self.is_terminal_buffer.append(True)

    def sample(self, batch_size, indexes=None):
        batch = {'next_state': np.zeros([batch_size, 84, 84, self.window_length]),
                 'state': np.zeros([batch_size, 84, 84, self.window_length]),
                 'reward': np.zeros([batch_size,]),
                 'action': np.zeros([batch_size,], int),
                 'is_terminal': np.zeros([batch_size,])}
        # if indexes==None:
        indexes = np.random.randint(low=1, high=self.buffer.length-1, size=[batch_size,1])
        # print(indexes)
        for j in range(indexes.size):
            i = indexes[j][0]
            while i ==self.buffer.tail_idx() or self.buffer[i].is_terminal == 1:
                i = np.random.randint(low=1, high=self.buffer.length-1)
            batch["next_state"][j] = self.buffer[i+1].state
            batch["state"][j] = self.buffer[i].state
            batch["reward"][j] = self.buffer[i].action
            batch["action"][j] = self.buffer[i].reward
            batch["is_terminal"][j] = self.buffer[i+1].is_terminal

        return batch

    def clear(self):
        self.buffer = RingBuffer(self.max_size)






