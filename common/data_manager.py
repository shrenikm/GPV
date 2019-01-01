#!/usr/bin/env python

from torch.utils.data import Dataset


# Class to load data
class CreateDataset(Dataset):

    def __init__(self, x, y):

        super(CreateDataset, self).__init__()

        assert x.shape[0] == y.shape[0], "x and y dimensions do not match"

        self.x = x
        self.y = y

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, index):

        sample = {'x': self.x[index, :].transpose(),
                  'y': self.y[index, :].transpose()}

        return sample
