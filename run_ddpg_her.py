#!/usr/bin/env python

import gym
import torch
import numpy as np

from ddpg_her.ddpg import Ddpg
from ddpg_her.replay_buffer import ReplayBuffer
from ddpg_her.utility import save_model
from ddpg_her.her import Her

# Cuda indication
if torch.cuda.is_available():
    print('GPU available. Using {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU')

# Environment and episode constants
ENV_RENDER = True
MAX_STEPS = None
MAX_BUFFER_SIZE = 1000000
MAX_EPOCHS = 200
MAX_CYCLES = 50
EPISODES_PER_EPOCH = 16
SAVE_RATE = 5
OPTIMIZATION_STEPS = 40

env_name = 'FetchPush-v1'
model_name = 'fetch_push'
save_path = 'ddpg_her/models/'
env = gym.make(env_name)

if MAX_STEPS is not None:
    env._max_episodes_steps = MAX_STEPS

state_space = env.observation_space
action_space = env.action_space

action_low, action_high, action_lim = None, None, None

state_dim = state_space.spaces['observation'].shape[0]
goal_dim = state_space.spaces['desired_goal'].shape[0]

# We concatenate the observation and the desired goal for the state
state_dim += goal_dim

if type(action_space) is gym.spaces.box.Box:

    action_dim = action_space.shape[0]
    action_low = action_space.low
    action_high = action_space.high

action_lim = action_high[0]

experience_replay = ReplayBuffer(MAX_BUFFER_SIZE)

# Initializing ddpg
ddpg = Ddpg(state_dim, action_dim, action_lim, experience_replay)
her = Her(env, 'final')

for epoch in range(MAX_EPOCHS):

    print('Epoch: {}'.format(epoch + 1))
    print('-'*50)

    for cycle in range(MAX_CYCLES):

        for episode in range(EPISODES_PER_EPOCH):

            print('Episode: {}'.format(episode + 1))

            full_state = env.reset()
            # Extracting information from the full state
            observation = np.float32(full_state['observation'])
            desired_goal = np.float32(full_state['desired_goal'])
            achieved_goal = np.float32(full_state['achieved_goal'])

            episode_reward = 0

            # History to store all the transitions till the end of the episode
            history = []

            while True:

                if ENV_RENDER:
                    env.render()

                state = np.concatenate((observation, desired_goal), axis=0)
                action = ddpg.noisy_action(state)

                full_state, reward, terminal, info = env.step(action)
                observation = np.float32(full_state['observation'])

                episode_reward += reward

                if not terminal:
                    next_state = np.concatenate((observation, desired_goal),
                                                axis=0)

                    # Appending to the history
                    history.append((achieved_goal, state, action,
                                    np.float32(reward), next_state))

                # Updating the achieved goal. It is updated here as the previous
                # state achieved goal is required for the history
                achieved_goal = np.float32(full_state['achieved_goal'])

                if terminal:

                    print('Total reward: {}'.format(episode_reward))
                    break

            # Adding to the experience replay (Standard replay)
            for transition in history:

                # Adding the transition in the replay buffer.
                # Ignoring the achieved goal
                ddpg.experience_replay.add(transition[1::])

            # Adding HER transitions
            her_history = her.add(history)
            for transition in her_history:

                ddpg.experience_replay.add(transition)

            if (SAVE_RATE is not None) and (episode % SAVE_RATE == 0):

                # Save the model
                save_model(ddpg, save_path, model_name)

        print('Optimizing')
        print('-'*50)

        # Training
        for i in range(OPTIMIZATION_STEPS):
            ddpg.train()

        # Updating targets
        print('Optimizing targets')
        print('-'*50)
        ddpg.update_target()
