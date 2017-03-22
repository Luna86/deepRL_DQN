from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.core import Sample
import numpy as np

class MemorySample:
    def __init__(self, frame, action, reward, is_terminal):
        self.frame = frame
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


class ExperienceReplayMemory(ReplayMemory):
    def __init__(self, max_size, window_length):
        self.buffer = RingBuffer(max_size)
        self.is_terminal_buffer = RingBuffer(max_size)
        self.window_length = window_length
        self.max_size = max_size

    def append(self, state, action, reward):
        frame = state[:,:,-1]
        self.buffer.append(MemorySample(frame=frame, action=action, reward=reward,  is_terminal = False))
        self.is_terminal_buffer.append(False)

    def end_episode(self, final_state, is_terminal):
        frame = final_state[:, :, -1]
        self.buffer.append(MemorySample(frame=frame, action=0, reward=0, is_terminal = is_terminal))
        self.is_terminal_buffer.append(True)

    def sample(self, batch_size, indexes=None):
        batch = {'next_state': np.zeros([batch_size, 84, 84, self.window_length]),
                 'state': np.zeros([batch_size, 84, 84, self.window_length]),
                 'reward': np.zeros([batch_size,]),
                 'action': np.zeros([batch_size,]),
                 'is_terminal': np.zeros([batch_size,])}
        # if indexes==None:
        indexes = np.random.randint(low=self.window_length, high=self.buffer.length, size=[batch_size,1])
        print(indexes)
        for j in range(indexes.size):
            i = indexes[j][0]
            print(i)
            m = self.buffer[i]
            action = m.action
            reward = m.reward
            # last_k_indexes = self.buffer.get_last_K_index(i,self.window_length)
            next_state = np.zeros([84,84,self.window_length])
            state = np.zeros([84,84,self.window_length])

            for k in range(self.window_length):
                index = i-k+1
                if index >=0 and index < self.buffer.length:
                    if self.is_terminal_buffer.get(index) and k != 0:
                        continue
                    next_state[:,:,-k-1] = (self.buffer.get(index).frame)
                else:
                    break

            for k in range(self.window_length):
                index = i-k
                if index >=0 and index < self.buffer.length:
                    if self.is_terminal_buffer.get(index) and k != 0:
                        continue
                    state[:,:,-k-1] = (self.buffer.get(index).frame)
                else:
                    break
            batch["next_state"][j] = next_state
            batch["state"][j] = state
            batch["reward"][j] = action
            batch["action"][j] = reward
            batch["is_terminal"][j] = m.is_terminal

            # s = Sample(state = state, action = action, reward=reward, next_state = next_state, is_terminal = m.is_terminal)
            # batch.append(s)
            # for k in range(len(last_k_indexes)):
            #     k_index = last_k_indexes[k]
            #     if self.is_terminal_buffer[k_index] and (k != 0):
            #         break
            #     state[:,:,-k-1] = self.buffer[k_index].frame
            # batch.append(Sample(state=state, ))
        return batch

    def clear(self):
        self.buffer = RingBuffer(self.max_size)
        self.is_terminal_buffer = RingBuffer(self.max_size)






