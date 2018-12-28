#!/usr/bin/env python

from collections import deque
import random


class ReplayBuffer:

    def __init__(self, max_size):

        self.max_size = max_size
        self.replay_buffer = deque(maxlen=max_size)
        self.size = 0

    # Length
    def __len__(self):

        return self.size

    # Get the content at a specific index
    def __getitem__(self, index):

        return self.replay_buffer[index]

    def add(self, content):

        # Appending the content to the right
        self.replay_buffer.append(content)

        # Updating the sizer of the replay buffer
        self.size = min(self.size + 1, self.max_size)

    def sample(self, sample_size):

        # Sampling contents from the buffer
        return random.sample(self.replay_buffer, min(sample_size, self.size))
