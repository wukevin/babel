"""
Misc. layers
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Warmup(object):  # Doesn't have to be nn.Module because it's not learned
    """
    Warmup layer similar to
    Sonderby 2016 - Linear deterministic warm-up
    """

    def __init__(self, inc: float = 5e-3, t_max: float = 1.0):
        self.t = 0.0
        self.t_max = t_max
        self.inc = inc
        self.counter = 0  # Track number of times called next

    def __iter__(self):
        return self

    def __next__(self):
        retval = self.t
        t_next = self.t + self.inc
        self.t = min(t_next, self.t_max)
        self.counter += 1
        return retval


class NullWarmup(Warmup):
    """
    No warmup - but provides a consistent API
    """

    def __init__(self, delay: int = 0, t_max: float = 1.0):
        self.val = t_max

    def __next__(self):
        return self.val


class DelayedLinearWarmup(object):
    """
    """

    def __init__(self, delay: int = 2000, inc: float = 5e-3, t_max: float = 1.0):
        self.t = 0.0
        self.t_max = t_max
        self.inc = inc
        self.delay = delay
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        retval = self.t
        if self.counter < self.delay:
            return retval
        self.t = min(self.t + self.inc, self.t_max)
        return retval


class SigmoidWarmup(object):
    """
    Sigmoid warmup
    Midpoints defines the number of iterations before we hit 0.5
    Scale determines how quickly we hit 1 after this point
    """

    def __init__(self, midpoint: int = 500, scale: float = 0.1, maximum: float = 1.0):
        self.midpoint = midpoint
        self.scale = scale
        self.maximum = maximum
        self.counter = 0
        self.t = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        retval = self.t
        t_next = 1.0 / (1.0 + np.exp(-self.scale * (self.counter - self.midpoint)))
        self.t = t_next
        self.counter += 1
        return self.maximum * retval


def plot_warmup(w, fname="plot.png"):
    values = []
    while True:
        if len(values) > 10 and values[-1] == values[-10] and values[-1] > 0:
            break
        values.append(next(w))
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(
        x=np.arange(len(values)), y=np.array(values),
    )
    ax.set(
        xlabel="Iterations", ylabel="Value",
    )
    fig.savefig(fname)


if __name__ == "__main__":
    x = SigmoidWarmup()
    plot_warmup(x)
