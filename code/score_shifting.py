from __future__ import print_function

from itertools import permutations

import numpy as np

from evaluation import pan_metrics

def rescale(value, orig_min, orig_max, new_min, new_max):
    orig_span = orig_max - orig_min
    new_span = new_max - new_min
    scaled_value = float(value - orig_min) / float(orig_span)
    return new_min + (scaled_value * new_span)

def correct_scores(scores, p1=.25, p2=.75):
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

    def __init__(self, step_size=0.05):
        self.optimal_p1 = None
        self.optimal_p2 = None
        self.step_size = step_size

    def fit(self, scores, labels, ground_truth):
        thresholds = np.arange(0.0, 1.0, self.step_size)
        nb_thresholds = thresholds.shape[0]

        # intialize score containers:
        both_scores = np.zeros((nb_thresholds, nb_thresholds))
        auc_scores = both_scores.copy()
        c_at_1_scores = both_scores.copy()

        # iterate over combinations:
        for i, j in permutations(range(nb_thresholds), 2):
            p1, p2 = thresholds[i], thresholds[j]
            print(p1)
            print(p2)
            if p1 <= p2:
                corrected_scores = correct_scores(scores, p1=p1, p2=p2)
                acc_score, auc_score, c_at_1_score = \
                    pan_metrics(prediction_scores=corrected_scores,
                                labels=labels,
                                ground_truth=ground_truth)
                auc_scores[i][j] = auc_score
                c_at_1_scores[i][j] = c_at_1_score
                both_scores[i][j] = auc_score*c_at_1_score

        opt_p1_idx, opt_p2_idx = np.unravel_index(both_scores.argmax(), both_scores.shape)
        self.optimal_p1 = thresholds[opt_p1_idx]
        self.optimal_p2 = thresholds[opt_p2_idx]
        
        print('p1 for optimal combo:', self.optimal_p1)
        print('p2 for optimal combo:', self.optimal_p2)
        print('AUC for optimal combo:', auc_scores[opt_p1_idx][opt_p2_idx])
        print('c@1 for optimal combo:', c_at_1_scores[opt_p1_idx][opt_p2_idx])

    def transform(self, scores):
        return correct_scores(scores=scores,
                                p1=self.optimal_p1,
                                p2=self.optimal_p2)



    