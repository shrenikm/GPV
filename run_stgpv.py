#!/usr/bin/env python

import gym
import torch
import numpy as np

from stgpv.stgpv import Stgpv
from ddpg.replay_buffer import ReplayBuffer
from ddpg.utility import save_model

# Cuda indication
if torch.cuda.is_available():
    print('GPU available. Using {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU.')

# Environment and episode constants
ENV_RENDER = True
MAX_STEPS = None
MAX_BUFFER_SIZE = 1000000
MAX_EPISODES = 1000
SAVE_RATE = 20
CHECK_RATE = 5

env_name = 'Pendulum-v0'
model_name = 'pendulum'
# env_name = 'BipedalWalker-v2'
# model_name = 'biped'

save_path = 'stgpv/models/'
env = gym.make(env_name)

if MAX_STEPS is not None:
    env._max_episodes_steps = MAX_STEPS

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

experience_replay = ReplayBuffer(MAX_BUFFER_SIZE)

stgpv = Stgpv(state_dim, action_dim, action_lim, experience_replay)

# STGPV parameters

# Number of updates
k_ddpg = 1
k_vae = 10

for episode in range(MAX_EPISODES):

    print('Episode: {}'.format(episode + 1))
    print('-'*50)

    observation = env.reset()

    episode_reward = 0

    print('Updating networks')

    while True:

        state = np.float32(observation)

        # Exploring actions are achieved using the VAE output
        action = stgpv.vae.generate(
                torch.from_numpy(state).float().to(stgpv.device))
        # The action must be squeezed as the vae generate unsqueezes
        # the first dimension
        action = action.cpu().detach().numpy().squeeze(0)

        new_observation, reward, terminal, info = env.step(action)

        # Adding the reward
        episode_reward += reward

        if terminal:
            break
        else:
            next_state = np.float32(new_observation)

            stgpv.experience_replay.add((
                state, action, np.float32(reward), next_state))

        observation = new_observation

        # Training by running the ddpg and vae updates

        for k in range(k_ddpg):
            stgpv.train_ddpg()

        for k in range(k_vae):
            stgpv.train_vae()

        if episode % SAVE_RATE == 0:

            save_model(stgpv, save_path, model_name)

    # Running a separate episode with the target policy


    if episode % CHECK_RATE == 0:

        print('Checking agent')
        observation = env.reset()
        episode_reward = 0

        while True:

            if ENV_RENDER:
                env.render()

            state = np.float32(observation)

            # Exploring actions are achieved using the VAE output
            action = stgpv.target_actor(
                    torch.from_numpy(state).float().to(stgpv.device))
            action = action.cpu().detach().numpy()

            new_observation, reward, terminal, info = env.step(action)

            # Adding the reward
            episode_reward += reward

            if terminal:

                print('Total reward: {}'.format(episode_reward))
                break

            else:

                next_state = np.float32(new_observation)
