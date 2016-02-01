#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offers implementation of a number of evaluation metrics
which are specific to the PAN competition's track on
authorship verification:
- accuracy
- AUC (Area Under the Curve)
- c@1

See for more details:
    E. Stamatatos, et al. Overview of the Author Identification
    Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
"""

from __future__ import print_function
from itertools import combinations

import seaborn as sb
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import recall_score, precision_score, f1_score


def accuracy(prediction_scores, ground_truth_scores):
    """
    Calculates the verification accuracy, assuming that every
    `score >= 0.5` represents an attribution.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will typically be `0` or `1`.

    Returns
    ----------
    acc = The number of correct attributions.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.

    """

    acc = 0.0
    assert len(ground_truth_scores) == len(prediction_scores)

    for gt_score, pred_score in zip(ground_truth_scores, prediction_scores):
        if (pred_score > 0.5) == (gt_score > 0.5):
            acc += 1.0
    return acc / float(len(prediction_scores))


def auc(prediction_scores, ground_truth_scores):
    """
    Calculates the AUC score (Area Under the Curve), a well-known
    scalar evaluation score for binary classifiers. This score
    also considers "unanswered" problem, where score = 0.5.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will typically be `0` or `1`.

    Returns
    ----------
    auc = the Area Under the Curve.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.

    """

    return roc_auc_score(ground_truth_scores, prediction_scores)


def c_at_1(prediction_scores, ground_truth_scores):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:

        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will always be `0` or `1`.

    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)


    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    """

    n = float(len(prediction_scores))
    nc, nu = 0.0, 0.0
    for gt_score, pred_score in zip(ground_truth_scores, prediction_scores):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    return (1 / n) * (nc + (nu * nc / n))


def pan_metrics(prediction_scores, ground_truth_scores):
    """
    Convenience function: calculates all three PAN evaluation measures for the
    given `prediction_scores` and `ground_truth_scores`.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will typically be `0` or `1`.

    Returns
    ----------
    (acc, auc, c@1) = a tuple with the 3 evaluation scores.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    
    """

    return accuracy(prediction_scores, ground_truth_scores), \
           auc(prediction_scores, ground_truth_scores), \
           c_at_1(prediction_scores, ground_truth_scores)
