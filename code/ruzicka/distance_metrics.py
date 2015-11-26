#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numba import jit

@jit
def pairwise_minmax(x, y):
    """
    Simple implementation of the minmax distance metric
    for two (non-sparse) vectors.

    Parameters
    ----------
    x : array [n_features]
    y : array [n_features]
        Input vectors between which minmax will between
        calculate. Does not:
            * support sparse input.
            * check whether len(x) == len(y).

    Returns
    ----------
    The minmax distance between x and y.

    """
    mins, maxs = 0.0, 0.0
    for i in range(x.shape[0]):
        a, b = x[i], y[i]
        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a
    if maxs > 0.0:
        return 1.0 - (mins / maxs)
    return 0.0