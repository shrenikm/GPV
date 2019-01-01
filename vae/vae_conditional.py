#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.distributions as tdist


# Conditional VAE
class VAEConditional(nn.Module):

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim1=32,
                 hidden_dim2=64,
                 hidden_dim3=128,
                 hidden_dim4=64,
                 hidden_dim5=32):

        super(VAEConditional, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dim4 = hidden_dim4
        self.hidden_dim5 = hidden_dim5

        # Encoder
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)

        # Decoder
        self.fc4 = nn.Linear(
                self.hidden_dim3//2 + self.state_dim, self.hidden_dim4)
        self.fc5 = nn.Linear(self.hidden_dim4, self.hidden_dim5)
        self.fc6 = nn.Linear(self.hidden_dim5, self.action_dim)

        self.relu = nn.ReLU()

        # Normal distribution
        self.normal_dist = \
            tdist.MultivariateNormal(torch.zeros(self.hidden_dim3//2),
                                     torch.eye(self.hidden_dim3//2))

    def encoder(self, state):

        q = self.relu(self.fc1(state))
        q = self.relu(self.fc2(q))
        q = self.relu(self.fc3(q))

        mu = q[:, 0:self.hidden_dim3//2]
        log_sigma = q[:, self.hidden_dim3//2:]

        z = mu + torch.exp(log_sigma) * \
            self.normal_dist.sample().to(self.device)

        return mu, log_sigma, z

    def decoder(self, z, state):

        d = torch.cat((z, state), dim=1)
        d = self.relu(self.fc4(d))
        d = self.relu(self.fc5(d))
        d = self.fc6(d)

        return d
