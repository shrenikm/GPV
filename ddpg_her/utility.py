#!/usr/bin/env python

import torch
import numpy as np


def fanin_initialization(size):

    init = 1./np.sqrt(size[0])
    return torch.Tensor(size).uniform_(-init, init)


# Copying weights
def copy_weights(target_network, source_network):

    for target_params, source_params in zip(
            target_network.parameters(), source_network.parameters()):
        target_params.data.copy_(source_params.data)


# Updating target network weights
def update_target(target_network, source_network, tau):

    for target_params, source_params in zip(
            target_network.parameters(), source_network.parameters()):
        target_params.data.copy_(
                target_params.data * (1.0 - tau) + source_params.data * tau)


# Saving the model
def save_model(ddpg, path, model, episode_count=""):

    torch.save(ddpg.actor.state_dict(),
               path + model + '/' + 'actor' + str(episode_count) + '.pth')
    torch.save(ddpg.critic.state_dict(),
               path + model + '/' + 'critic' + str(episode_count) + '.pth')
    torch.save(ddpg.target_actor.state_dict(),
               path + model + '/' + 'target_actor' +
               str(episode_count) + '.pth')
    torch.save(ddpg.target_critic.state_dict(),
               path + model + '/' + 'target_critic' +
               str(episode_count) + '.pth')


# Ornstein Uhlenbeck noise
class OrnsteinUhlenbeckNoise:

    def __init__(self, size, mu=0, theta=0.15, sigma=0.2, dt=1e-2):

        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

        # Resetting to the mean
        self.reset()

    def reset(self):

        self.x = np.ones(self.size) * self.mu

    # Sampling
    def sample(self):

        dx = self.theta * (self.mu - self.x) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.x = self.x + dx
        return self.x


# Exploration noise
class ExplorationNoise:

    def __init__(self, action_space):

        # The following assumes a box action space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.range = self.action_high - self.action_low

    def sample(self, action):

        choice = np.random.choice(2, 1, p=[0.2, 0.8])

        if choice == 0:

            # Random action within the box
            action = self.action_space.sample()

        else:

            # Adding normal noise
            action += np.random.normal(0.0,
                                       (5.0/100)*self.range,
                                       self.action_dim)

        return action
