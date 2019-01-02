#!/usr/bin/env python

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from common.data_manager import CreateDataset
from common.plot_manager import CustomFigure2D
from vae.vae_conditional import VAEConditional

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

if use_cuda:
    print('GPU available. Using {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU.')

# Environment and episode constants
ENV_RENDER = True
ENV_SEED = 1
MAX_EPISODES = 100

env_name = 'BipedalWalker-v2'
model_name = 'biped'
load_path = 'ddpg/data/'
save_path = 'vae/models/'
env = gym.make(env_name)

state_space = env.observation_space
action_space = env.action_space

state_dim = None
action_dim = None
action_low, action_high, action_lim = None, None, None

if type(state_space) is gym.spaces.box.Box:
    state_dim = state_space.shape[0]

if type(action_space) is gym.spaces.box.Box:

    action_dim = action_space.shape[0]
    action_low = action_space.low
    action_high = action_space.high

action_lim = action_high

if action_dim == 1:

    action_low, action_high, action_lim = \
            action_low[0], action_high[0], action_lim[0]

# Loading the data
data = np.load(load_path + model_name + '/' + 'data.npz')

x = data['arr_0']
y = data['arr_1']

# Creating the dataset
num_data = x.shape[0]
train_ratio = 0.8
num_train = int(num_data * train_ratio)
batch_size_train = 100
batch_size_test = 100

train_dataset = CreateDataset(x[0:num_train], y[0:num_train])
test_dataset = CreateDataset(x[num_train:], y[num_train:])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test)

# Learning parameters
learning_rate = 0.001
max_iter = 20000

iterations_per_epoch = int(num_train / batch_size_train)
print("Iterations per epoch: ", iterations_per_epoch)
num_epochs = int(max_iter / iterations_per_epoch)
print("Number of epochs: ", num_epochs)

vae = VAEConditional(state_dim=state_dim, action_dim=action_dim)
vae = vae.to(device)

# Loss
loss_function_decoder = nn.MSELoss()

# Alpha controls the weighing of both the losses
alpha = 0.05

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=learning_rate, betas=(0.5, 0.999))

vae.train()
iteration_count = 0
loss_list = []

for epoch in range(num_epochs):

    for i, batch in enumerate(train_loader):

        iteration_count += 1

        state = batch['x']
        action = batch['y']

        state = state.float().to(device)
        action = action.float().to(device)

        optimizer.zero_grad()

        mu, log_sigma, z = vae.encoder(state)
        d = vae.decoder(z, state)

        loss_decoder = loss_function_decoder(d, action)

        loss_encoder = 0.5*torch.mean(mu ** 2 + torch.exp(log_sigma) ** 2 -
                                      log_sigma - 1)

        # Weighing the losses
        loss = alpha * loss_decoder + (1 - alpha) * loss_encoder

        loss_list.append(loss.item())

        print('Loss (Iteration {0}): {1:.3f}'.format(
            iteration_count, loss.item()))

        loss.backward()
        optimizer.step()

# Saving
torch.save(vae.state_dict(), save_path + model_name + '/' + 'vae.pth')

# Plotting
figure = CustomFigure2D('Train loss Vs Iterations')
figure.set_axes_labels('Training iterations', 'Training loss')
figure.ax.plot(range(1, len(loss_list) + 1), loss_list)
plt.show()

# Test loss
iteration_count = 0
test_loss = 0
vae.eval()

with torch.no_grad():

    for i, batch in enumerate(test_loader):

        iteration_count += 1

        state = batch['x']
        action = batch['y']

        state = state.float().to(device)
        action = action.float().to(device)

        mu, log_sigma, z = vae.encoder(state)
        d = vae.decoder(z, state)

        loss_decoder = loss_function_decoder(d, action)
        loss_encoder = 0.5 * torch.mean(mu ** 2 + torch.exp(log_sigma) ** 2 -
                                        log_sigma - 1)

        loss = alpha * loss_decoder + (1 - alpha) * loss_encoder

        test_loss += loss.item()

print('Average test loss: {}'.format(test_loss/iteration_count))
