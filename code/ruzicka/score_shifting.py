#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from itertools import permutations

import numpy as np

from evaluation import pan_metrics

def rescale(value, orig_min, orig_max, new_min, new_max):
    """

    Rescales a `value` in the old range defined by
    `orig_min` and `orig_max`, to the new range
    `new_min` and `new_max`. Assumes that
    `orig_min` <= value <= `orig_max`.

    Parameters
    ----------
    value: float, default=None
        The value to be rescaled.
    orig_min: float, default=None
        The minimum of the original range.
    orig_max: float, default=None
        The minimum of the original range.
    new_min: float, default=None
        The minimum of the new range.
    new_max: float, default=None
        The minimum of the new range.

    Returns
    ----------
    new_value: float
        The rescaled value.

    """

    orig_span = orig_max - orig_min
    new_span = new_max - new_min

    try:
        scaled_value = float(value - orig_min) / float(orig_span)
    except ZeroDivisionError:
        orig_span += 1e-6
        scaled_value = float(value - orig_min) / float(orig_span)

    return new_min + (scaled_value * new_span)


def correct_scores(scores, p1=.25, p2=.75):
    """

    Rescales a list of scores (originally between 0 and 1)
    to three new intervals:
    - a range between 0 and `p1`
    - a range between `p2`and 1
    - a range `p1` and `p2` (where every score = 0.5)

    Parameters
    ----------
    scores: array-like [nb_scores]
        The list of scores to be rescaled
    p1: float, default=.25
        The minimum of the original range.
    p2: float, default=.75
        The minimum of the original range.

    Returns
    ----------
    new_scores: floats, array-like [nb_scores]
        The rescaled scores.

    """
    new_scores = []
    for score in list(scores):
        if score <= p1:
            new_scores.append(rescale(score, min(scores), max(scores), 0.0, p1))
        elif score >= p2:
            new_scores.append(rescale(score, min(scores), max(scores), p2, 1.0))
        else:
            new_scores.append(0.5)
    return new_scores


class ScoreShifter:
    """

    An object to shifts the raw verification probabilities 
    outputted by a system, to better account for the PAN
    metrics, which have a strict attribution threshold at 0.5.

    The shifter can be fitted on a set of train scores, by
    optimizing the threshold parameters `p1` and `p2`
    with respect to AUC x c@1, using a simple grid search.

    """

    def __init__(self, step_size=0.01):
        """
        Contructor.

        Parameters
        ----------
        step_size: float, default=0.5
            The step size between the different values of
            `p1` and `p2` to be tested in the grid search.

        """
        self.optimal_p1 = None
        self.optimal_p2 = None
        self.step_size = step_size

    def fit(self, predicted_scores, ground_truth_scores):
        """
        Fits the score shifter on the (development) scores for
        a data set, by searching the optimal `p1` and `p2` (in terms
        of AUC x c@1) through a stepwise grid search.
        
        Parameters
        ----------
        prediction_scores : array [n_problems]
            The predictions outputted by a verification system.
            Assumes `0 >= prediction <=1`.

        ground_truth_scores : array [n_problems]
            The gold annotations provided for each problem.
            Will typically be `0` or `1`.

        """

        # define the grid to be searched:
        thresholds = np.arange(0.05, 1.0, self.step_size)
        nb_thresholds = thresholds.shape[0]

        # intialize score containers:
        both_scores = np.zeros((nb_thresholds, nb_thresholds))
        auc_scores = both_scores.copy()
        c_at_1_scores = both_scores.copy()

        # iterate over combinations:
        for i, j in permutations(range(nb_thresholds), 2):
            p1, p2 = thresholds[i], thresholds[j]

            if p1 <= p2: # ensure p1 <= p2!
                corrected_scores = correct_scores(predicted_scores, p1=p1, p2=p2)
                acc_score, auc_score, c_at_1_score = \
                    pan_metrics(prediction_scores=corrected_scores,
                                ground_truth_scores=ground_truth_scores)
                auc_scores[i][j] = auc_score
                c_at_1_scores[i][j] = c_at_1_score
                both_scores[i][j] = auc_score * c_at_1_score

        # find 2D optimum:
        opt_p1_idx, opt_p2_idx = np.unravel_index(both_scores.argmax(), both_scores.shape)
        self.optimal_p1 = thresholds[opt_p1_idx]
        self.optimal_p2 = thresholds[opt_p2_idx]
        
        # print some info:
        print('p1 for optimal combo:', self.optimal_p1)
        print('p2 for optimal combo:', self.optimal_p2)
        print('AUC for optimal combo:', auc_scores[opt_p1_idx][opt_p2_idx])
        print('c@1 for optimal combo:', c_at_1_scores[opt_p1_idx][opt_p2_idx])

        return self

    def transform(self, scores):
        """
        Shifts the probabilities of a (new) problem series, through
        applying the score_shifter with the previously set `p1` and `p2`.
        
        Parameters
        ----------
        scores : array [n_problems]
            The scores to be shifted

        Returns
        ----------
        shifted_scores: floats, array-like [nb_scores]
            The shifted scores.

        """

        return correct_scores(scores=scores,
                                p1=self.optimal_p1,
                                p2=self.optimal_p2)

    