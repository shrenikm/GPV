#!/usr/bin/env python

import torch
import torch.nn as nn

from ddpg.utility import fanin_initialization


class Critic(nn.Module):

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim1=64,
                 hidden_dim2=64,
                 hidden_dim3=64,
                 e=0.003):

        super(Critic, self).__init__()

        # Set the device
        self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.e = e

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(
                self.hidden_dim1 + self.action_dim, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.fc4 = nn.Linear(self.hidden_dim3, 1)

        self.relu_activation = nn.ReLU()

        self.initialize_weights()

    def initialize_weights(self):

        # Fanin initialization for the first two layers
        self.fc1.weight.data = fanin_initialization(
                self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_initialization(
                self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_initialization(
                self.fc3.weight.data.size())

        # Uniform initialization for the final layer
        self.fc4.weight.data.uniform_(-self.e, self.e)
        self.fc4.bias.data.uniform_(-self.e, self.e)

    def forward(self, state, action):

        q = self.relu_activation(self.fc1(state))
        q = torch.cat((q, action), dim=1)
        q = self.relu_activation(self.fc2(q))
        q = self.relu_activation(self.fc3(q))
        q = self.fc4(q)

        return q
