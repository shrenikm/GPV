#!/usr/bin/env python

import gym
import torch
import numpy as np

from ddpg.actor import Actor
from ddpg.critic import Critic

# Cuda indication
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

if use_cuda:
    print('GPU available. Using {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU')

# Environment and episode constants
ENV_RENDER = True
ENV_SEED = 1
MAX_EPISODES = 100
COLLECT_DATA = True

env_name = 'BipedalWalker-v2'
model_name = 'biped'
load_path = 'ddpg/models/'
save_path = 'ddpg/data/'
env = gym.make(env_name)

# Data to store
x = None
y = None

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

# Loading the actor
actor = Actor(state_dim=state_dim,
              action_dim=action_dim,
              action_lim=action_lim)
actor = actor.to(device)

# Loading network parameters
actor.load_state_dict(torch.load(load_path + model_name + '/' + 'actor.pth'))

# Evaluating the trained actor
for episode in range(MAX_EPISODES):

    print('Episode: {}'.format(episode + 1))

    # Setting the seed if required
    if ENV_SEED is not None:
        env.seed(ENV_SEED)

    observation = env.reset()

    while True:

        if ENV_RENDER:
            env.render()

        state = np.float32(observation)
        action = actor(torch.from_numpy(state).float().to(device))
        action = action.cpu().detach().numpy()

        # Storing data
        if COLLECT_DATA:

            if x is None:

                # Add the first collected data point
                x = state.reshape(1, state.shape[0])
                y = action.reshape(1, action.shape[0])

            else:

                # Append
                x = np.append(x, state.reshape(1, state.shape[0]), axis=0)
                y = np.append(y, action.reshape(1, action.shape[0]), axis=0)


        new_observation, reward, terminal, info = env.step(action)

        if terminal:
            next_state = None
        else:
            next_state = np.float32(new_observation)

        observation = new_observation

        if terminal:
            break

# Save data
if COLLECT_DATA:
    np.savez(save_path + model_name + '/' + 'data.npz', x, y)
