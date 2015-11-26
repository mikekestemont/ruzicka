#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Note: Legacy module which is not actually used any more,
but which is potentially useful for future reference.
Offers theano-implementations of the distance functions used,
for calculating the pairwise distances between two vectors.

"""

import theano
import theano.tensor as T

def get_manhattan_fn():
    a = T.fvector('a')
    b = T.fvector('b')
    idxs = T.ivector('idxs')
    distance = T.sum(T.abs_(a[[idxs]] - b[[idxs]]), axis=1)
    return theano.function([a, b, idxs], distance, allow_input_downcast=True)

def get_euclidean_fn():
    a = T.fvector('a')
    b = T.fvector('b')
    idxs = T.ivector('idxs')
    distance = T.sqrt(T.sum(T.sqr(a[[idxs]] - b[[idxs]]), axis=1))
    return theano.function([a, b, idxs], distance, allow_input_downcast=True)

def get_minmax_fn():
    a = T.fvector('a')
    b = T.fvector('b')
    idxs = T.ivector('idxs')
    comb = T.stack([a[[idxs]], b[[idxs]]], axis=1)
    distance = 1.0 - ( T.sum(T.min(comb, axis=1)) / (T.sum(T.max(comb, axis=1)) + 1e-6) )
    return theano.function(inputs=[a, b, idxs], outputs=distance, allow_input_downcast=True)
