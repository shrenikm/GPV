#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from vae.vae_conditional import VAEConditional
from ddpg.ddpg import Ddpg


class Stgpv(Ddpg):

    def __init__(self, state_dim, action_dim, action_lim, experience_replay):

        # Calling the DDPG base class constructor
        super(Stgpv, self).__init__(
                state_dim, action_dim, action_lim, experience_replay)

        # VAE hyperparameters
        self.vae_lr = 0.001
        self.alpha = 0.001

        # VAE
        self.vae = VAEConditional(state_dim=state_dim, action_dim=action_dim)
        self.vae = self.vae.to(self.device)
        self.vae_optimizer = optim.Adam(
                self.vae.parameters(), lr=self.vae_lr, betas=(0.5, 0.999))
        self.vae_decoder_criterion = nn.MSELoss()

    # DDPG training function
    def train_ddpg(self):

        # Calling the DDPG train function
        self.train()

    # VAE training function
    def train_vae(self):

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

        predicted_action = self.actor.forward(current_state)

        # Optimize
        self.vae_optimizer.zero_grad()
        mu, log_sigma, z = self.vae.encoder(current_state)
        d = self.vae.decoder(z, current_state)

        loss_decoder = self.vae_decoder_criterion(
                d, predicted_action)
        loss_encoder = 0.5 * torch.mean(mu ** 2 + torch.exp(log_sigma) ** 2 -
                                        log_sigma - 1)
        loss = self.alpha * loss_decoder + (1 - self.alpha) * loss_encoder
        loss.backward()
        self.vae_optimizer.step()
