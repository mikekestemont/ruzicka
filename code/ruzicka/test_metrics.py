#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

"""
Workhorse module, which contains the main distance functions
used (minmax, manhattan, and euclidean). By setting
`TARGET = gpu` below, the computation of the functions
can be accelerated on the GPU, if the `numbapro`
package is available:
    http://docs.continuum.io/numbapro/index
"""

from numbapro import autojit

TARGET = 'cpu'

@autojit(target=TARGET)
def minmax(x, y, rnd_feature_idxs):
    """
    Calculates the pairwise "minmax" distance between
    two vectors, but limited to the `rnd_feature_idxs`
    specified. Note that this function is symmetric,
    so that `minmax(x, y) = minmax(y, x)`.

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.
    rnd_feature_idxs: int array
        The list of indexes along which the distance
        has to be calculated (useful for bootstrapping).

    Returns
    ----------
    float: minmax(x, y)

    References:
    ----------
    - M. Koppel and Y. Winter (2014), Determining if Two
      Documents are by the Same Author, JASIST, 65(1): 178-187.
    - Cha SH. Comprehensive Survey on Distance/Similarity Measures
      between Probability Density Functions. International Journ.
      of Math. Models and Methods in Applied Sciences. 2007;
      1(4):300–307.
    """

    mins, maxs = 0.0, 0.0
    a, b = 0.0, 0.0

    for i in rnd_feature_idxs:
        a, b = x[i], y[i]
        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a

    return 1.0 - (mins / (maxs + 1e-6)) # avoid zero division


@autojit(target=TARGET)
def manhattan(x, y, rnd_feature_idxs):
    """
    Calculates the conventional pairwise Manhattan city
    block distance between two vectors, but limited to
    the `rnd_feature_idxs` specified.

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.
    rnd_feature_idxs: int array
        The list of indexes along which the distance
        has to be calculated (useful for bootstrapping).

    Returns
    ----------
    float: manhattan(x, y)

    References:
    ----------
    - Cha SH. Comprehensive Survey on Distance/Similarity Measures
      between Probability Density Functions. International Journ.
      of Math. Models and Methods in Applied Sciences. 2007;
      1(4):300–307.
    """

    diff, z = 0.0, 0.0

    for i in rnd_feature_idxs:
        z = x[i]-y[i]
        if z < 0.0:
            z = -z
        diff += z

    return diff


@autojit(target=TARGET)
def euclidean(x, y, rnd_feature_idxs):
    """
    Calculates the conventional pairwise Euclidean
    distance between two vectors, but limited to
    the `rnd_feature_idxs` specified.

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.
    rnd_feature_idxs: int array
        The list of indexes along which the distance
        has to be calculated (useful for bootstrapping).

    Returns
    ----------
    float: euclidean(x, y)

    References:
    ----------
    - Cha SH. Comprehensive Survey on Distance/Similarity Measures
      between Probability Density Functions. International Journ.
      of Math. Models and Methods in Applied Sciences. 2007;
      1(4):300–307.
    """
    diff, z = 0.0, 0.0

    for i in rnd_feature_idxs:
        z = x[i]-y[i]
        diff += (z * z)
    
    return math.sqrt(diff)



