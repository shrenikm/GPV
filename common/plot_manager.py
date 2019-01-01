#!/usr/bin/env python

import matplotlib.pyplot as plt


# Helper functions for plotting
class CustomFigure2D(object):

    def __init__(self, title):

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.ax.set_title(title)

    # Set axis limtis
    def set_axes_limits(self, x_lim, y_lim):

        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])

    # Set axis labels
    def set_axes_labels(self, x_label, y_label):

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    # Set figure title
    def set_figure_title(self, title):

        self.ax.set_title(title)


# Class for 3D figures
class CustomFigure3D(object):

    def __init__(self, title):

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_title(title)
