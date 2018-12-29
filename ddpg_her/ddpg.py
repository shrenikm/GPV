#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

from ddpg.actor import Actor
from ddpg.critic import Critic
from ddpg.utility import OrnsteinUhlenbeckNoise
from ddpg.utility import copy_weights, update_target


class Ddpg:

    def __init__(self, state_dim, action_dim, action_lim, experience_replay):

        # Set the device
        self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

        # dimensions and replay buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.experience_replay = experience_replay

        # Hyperparameters
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.gamma = 0.99
        self.tau = 0.001
        self.minibatch_size = 128

        # Noise
        self.noise = OrnsteinUhlenbeckNoise(size=self.action_dim)

        # Actor
        self.actor = Actor(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           action_lim=self.action_lim)
        self.actor = self.actor.to(self.device)

        self.target_actor = Actor(state_dim=self.state_dim,
                                  action_dim=self.action_dim,
                                  action_lim=self.action_lim)
        self.target_actor = self.target_actor.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr)

        # Critic
        self.critic = Critic(state_dim=self.state_dim,
                             action_dim=self.action_dim)
        self.critic = self.critic.to(self.device)

        self.target_critic = Critic(state_dim=self.state_dim,
                                    action_dim=self.action_dim)
        self.target_critic = self.target_critic.to(self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr)

        self.critic_criterion = nn.MSELoss()

        # Making the network weights equal
        copy_weights(self.target_actor, self.actor)
        copy_weights(self.target_critic, self.critic)

    def train(self):

        # Get a batch from the experience replay
        content_batch = self.experience_replay.sample(self.minibatch_size)

        # Extracting information
        current_state = np.float32(
                [content[0] for content in content_batch])
        current_action = np.float32(
                [content[1] for content in content_batch])
        reward = np.float32(
                [content[2] for content in content_batch])
        next_state = np.float32(
                [content[3] for content in content_batch])

        current_state = torch.from_numpy(current_state).to(self.device)
        current_action = torch.from_numpy(current_action).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)

        target_action = self.target_actor.forward(next_state)
        q = torch.squeeze(
                self.target_critic.forward(next_state, target_action))
        y_target = reward + self.gamma * q
        y_predicted = torch.squeeze(
                self.critic.forward(current_state, current_action))

        # Optimize critic
        loss_critic = self.critic_criterion(y_predicted, y_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        predicted_action = self.actor.forward(current_state)
        loss_actor = -1*torch.sum(
                self.critic.forward(current_state, predicted_action))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Updating the target networks
        update_target(self.target_actor, self.actor, self.tau)
        update_target(self.target_critic, self.critic, self.tau)

    def noisy_action(self, state):

        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor.forward(state).cpu().detach().numpy()

        noise_action = action + self.noise.sample() * self.action_lim
        return noise_action
