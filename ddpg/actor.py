#!/usr/bin/env python

import torch
import torch.nn as nn

from utility import fanin_initialization


class Actor(nn.Module):

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim1=400,
                 hidden_dim2=300,
                 action_lim=1.0,
                 e=0.003):

        super(Actor, self).__init__()

        # Set the device
        self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.action_lim = action_lim
        self.e = e

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.action_dim)

        self.relu_activation = nn.ReLU()
        self.tanh_activation = nn.Tanh()

        self.initialize_weights()

    def initialize_weights(self):

        # Fanin initialization for the first two layers
        self.fc1.weight.data = fanin_initialization(
                self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_initialization(
                self.fc2.weight.data.size())

        # Uniform initialization for the final layer
        self.fc3.weight.data.uniform_(-self.e, self.e)
        self.fc3.bias.data.uniform_(-self.e, self.e)

    def forward(self, state):

        action = self.relu_activation(self.fc1(state))
        action = self.relu_activation(self.fc2(action))
        action = self.tanh_activation(self.fc3(action))

        # Scaling the action to match the limits
        action = action * torch.tensor(self.action_lim).to(self.device)

        return action
