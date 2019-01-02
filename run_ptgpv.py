#!/usr/bin/env python

import gym
import torch
import numpy as np

from ddpg.actor import Actor
from vae.vae_conditional import VAEConditional

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

if use_cuda:
    print('GPU available. Using {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU.')

# Environment and episode constants
ENV_RENDER = True
ENV_SEED = 7
MAX_EPISODES = 100
RECORD_VIDEO = True

env_name = 'BipedalWalker-v2'
model_name = 'biped'
load_ddpg_path = 'ddpg/models/'
load_vae_path = 'vae/models/'
env = gym.make(env_name)

# Recording video if required
if RECORD_VIDEO:
    env = gym.wrappers.Monitor(env, "results")

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

# Initializing and loading the actor
actor = Actor(state_dim=state_dim,
              action_dim=action_dim,
              action_lim=action_lim)
actor = actor.to(device)
actor.load_state_dict(
        torch.load(load_ddpg_path + model_name + '/' + 'actor.pth'))

# Initializing and loading the vae
vae = VAEConditional(state_dim=state_dim, action_dim=action_dim)
vae = vae.to(device)
vae.load_state_dict(
        torch.load(load_vae_path + model_name + '/' + 'vae.pth'))

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
        action_ddpg = actor(torch.from_numpy(state).float().to(device))
        action_ddpg = action_ddpg.cpu().detach().numpy()

        action_vae = vae.generate(torch.from_numpy(state).float().to(device))
        action_vae = action_vae.squeeze(0)
        action_vae = action_vae.cpu().detach().numpy()

        # Setting the action
        action = action_vae

        new_observation, reward, terminal, info = env.step(action)

        if terminal:
            next_state = None
            break
        else:
            next_state = np.float32(new_observation)
            observation = new_observation
