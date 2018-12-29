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
