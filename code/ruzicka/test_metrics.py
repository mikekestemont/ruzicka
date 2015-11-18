# -*- coding: utf-8 -*-
from numbapro import autojit

TARGET = 'cpu'

@autojit(target=TARGET)
def minmax(x, y, rnd_feature_idxs='all'):
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
def manhattan(x, y, rnd_feature_idxs='all'):
    diff, z = 0.0, 0.0
    for i in rnd_feature_idxs:
        z = x[i]-y[i]
        if z < 0.0:
            z = -z
        diff += z
    return diff

@autojit(target=TARGET)
def euclidean(x, y, rnd_feature_idxs='all'):
    diff, z = 0.0, 0.0
    for i in rnd_feature_idxs:
        z = x[i]-y[i]
        diff += (z * z)
    return diff * diff



