#!/usr/bin/env python

import gym
import torch
import torch.nn as nn
import numpy as np


class Her:

    def __init__(self, env, her_type='final'):

        self.env = env
        self.her_type = her_type

    def add(self, history):
    
        her_history = []

        if self.her_type == 'final':

            # Computing the new final goal
            # In this case it is the achieved goal at the final transition
            her_desired_goal = history[-1][0]

            for transition in history:

                achieved_goal = transition[0]
                state = transition[1]
                action = transition[2]
                reward = transition[3]
                next_state = transition[4]

                # Obtaining the new reward
                # The info parameter isnt being used and is set to None
                her_reward = self.env.compute_reward(
                        achieved_goal, her_desired_goal, None)

                # Changing the goals for the states
                state[-3:] = her_desired_goal
                next_state[-3:] = her_desired_goal

                # Adding to her replay
                her_history.append((state, action, her_reward, next_state))

        return her_history













