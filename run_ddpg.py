#!/usr/bin/env python

import gym
import torch
import numpy as np

from ddpg.ddpg import DDPG
from ddpg.replay_buffer import ReplayBuffer
from ddpg.utility import save_model

# Cuda indication
if torch.cuda.is_available():
    print('GPU available. Using {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU')

# Environment and episode constants
ENV_RENDER = True
MAX_STEPS = None
MAX_BUFFER_SIZE = 1000000
MAX_EPISODES = 5000
SAVE_RATE = 50

env_name = 'Pendulum-v0'
model_name = 'pendulum'
save_path = 'ddpg/models/'
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

# Initializing ddpg
ddpg = DDPG(state_dim, action_dim, action_lim, experience_replay)

for episode in range(MAX_EPISODES):

    observation = env.reset()

    print('Episode: {}'.format(episode + 1))

    episode_reward = 0

    while True:

        if ENV_RENDER:
            env.render()

        state = np.float32(observation)
        action = ddpg.noisy_action(state)

        new_observation, reward, terminal, info = env.step(action)

        # Adding the reward
        episode_reward += reward

        if terminal:
            next_state = None
        else:
            next_state = np.float32(new_observation)

            # Adding to the experience replay
            ddpg.experience_replay.add((
                state, action, np.float32(reward), next_state))

        observation = new_observation

        # Training the actor and critic networks
        ddpg.train()

        if terminal:

            print('Total reward: {}'.format(episode_reward))
            break

    if (episode % SAVE_RATE == 0) and (SAVE_RATE is not None):

        # Save the model
        save_model(ddpg, save_path, model_name)

