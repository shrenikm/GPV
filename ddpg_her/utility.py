#!/usr/bin/env python

import torch
import numpy as np


def fanin_initialization(size):

    init = 1./np.sqrt(size[0])
    return torch.Tensor(size).uniform_(-init, init)
