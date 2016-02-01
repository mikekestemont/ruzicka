#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This module offers convenience wrappers for three type of
common routines in authorship experiments on a PAN data set:
    - leave-one-out attribution experiment
    - development verification experiment
    - development and test verification experiment
    - test verification experiment

"""

from __future__ import print_function
import os
from collections import Counter

import numpy as np
from sklearn import cross_validation
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from utilities import load_pan_dataset, load_ground_truth
from vectorization import Vectorizer
from score_shifting import ScoreShifter
from evaluation import pan_metrics
from Order1Verifier import Order1Verifier as Verifier

# we import the pairwise distance metrics:
from distance_metrics import pairwise_minmax
from scipy.spatial.distance import euclidean as pairwise_euclidean
from scipy.spatial.distance import cityblock as pairwise_manhattan

def attribution_experiment(corpus_dir, mfi, vector_space, min_df,
                           ngram_type, ngram_size, metric, base):
    """

    - Runs a naive authorship attribution experiment
      on the texts in the train folder under `corpus_dir`.
    - Returns the `accuracy` after a leave-one-text-out
      validation experiment.
    - Because of this setup, we only include texts
      from authors which have at least two texts
      in the training (excluding any `unknown.txt`).
    - For classification, we use the `neighbors` module
      from `sklearn`:
        + KNeighborsClassifier (instance-based, brute
          implementation of IB1-algorithm, with n=1).
        + NearestCentroid (`profile`, with mean
          centroids per author class and no shrinking).
    
    """

    METRICS = {'manhattan':pairwise_manhattan,
               'euclidean':pairwise_euclidean,
               'minmax':pairwise_minmax}
    pairwise_metric_fn = METRICS[metric]
    
    # extract the train data
    dev_train_data, _ = load_pan_dataset(corpus_dir+'train')
    dev_train_labels, dev_train_documents = zip(*dev_train_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi = mfi,
                            vector_space = vector_space,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size,
                            min_df = min_df)
    dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
    
    # encode problem/author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_train_labels)
    dev_train_y = label_encoder.transform(dev_train_labels)

    # only select texts from authors with more than 1 text:
    more_than_one = set(i for i,c in Counter(dev_train_y).items()\
                            if i > 1)
    train_X, train_y = [], []
    for i in range(dev_train_X.shape[0]):
        if dev_train_y[i] in more_than_one:
            train_X.append(dev_train_X[i])
            train_y.append(dev_train_y[i])
    dev_train_X = np.asarray(train_X)
    dev_train_y = np.asarray(train_y)

    # initiate the IB1-like classifier:
    clf = None
    if base == 'instance':
        # normal instance-based IB1:
        clf = KNeighborsClassifier(n_neighbors = 1,
                                   weights = 'uniform',
                                   algorithm = 'brute',
                                   metric = pairwise_metric_fn)
    elif base == 'profile':
        # mean centroid-based IB1:
        clf = NearestCentroid(shrink_threshold = None,
                              metric = pairwise_metric_fn)

    # simple leave-one-text-out validation:
    nb_samples = dev_train_X.shape[0]
    cv = cross_validation.LeaveOneOut(nb_samples)
    predictions = cross_validation.cross_val_predict(clf,
                               dev_train_X, dev_train_y, cv=cv)

    # return accuracy:
    return accuracy_score(y_true = dev_train_y,
                           y_pred = predictions)


def dev_experiment(corpus_dir, mfi, vector_space,
                   ngram_type, ngram_size, metric,
                   base, nb_bootstrap_iter, rnd_prop,
                   nb_imposters, min_df):
    """

    * Runs a development verification experiment on the
      train data folder under `corpus_dir`.
    * Calculates PAN metrics (auc, acc, c@1) on this data,
      given the *optimal* p1/p2 found by the score shifter.
    * Returns: auc, acc, c@1, p1, p2

    """
    
    # preprocess:
    dev_train_data, dev_test_data = load_pan_dataset(corpus_dir+'train')
    dev_train_labels, dev_train_documents = zip(*dev_train_data)
    dev_test_labels, dev_test_documents = zip(*dev_test_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi = mfi,
                            vector_space = vector_space,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size)

    # transform (and unsparsify):
    dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
    dev_test_X = vectorizer.transform(dev_test_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_train_labels+dev_test_labels)
    dev_train_y = label_encoder.transform(dev_train_labels)
    dev_test_y = label_encoder.transform(dev_test_labels)
    
    # fit the verifier:
    dev_verifier = Verifier(metric = metric,
                            base = base,
                            nb_bootstrap_iter = nb_bootstrap_iter,
                            rnd_prop = rnd_prop)
    dev_verifier.fit(dev_train_X, dev_train_y)
    dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                                 test_y = dev_test_y,
                                                 nb_imposters = nb_imposters)
    
    # load the ground truth:
    dev_gt_scores = load_ground_truth(
                        filepath=os.sep.join((corpus_dir, 'train', 'truth.txt')),
                        labels=dev_test_labels)

    # fit the score shifter
    shifter = ScoreShifter()
    shifter.fit(predicted_scores=dev_test_scores,
                ground_truth_scores=dev_gt_scores)
    dev_test_scores = shifter.transform(dev_test_scores)
    
    dev_acc_score, dev_auc_score, dev_c_at_1_score = \
            pan_metrics(prediction_scores=dev_test_scores,
                        ground_truth_scores=dev_gt_scores)

    return dev_auc_score, dev_acc_score, dev_c_at_1_score,\
           shifter.optimal_p1, shifter.optimal_p2


def dev_test_experiment(corpus_dir, mfi, vector_space,
               ngram_type, ngram_size, metric,
               base, nb_bootstrap_iter, rnd_prop,
               nb_imposters, min_df):
    """

    - Runs a verification experiment on the train data
      under `corpus_dir`, and, then, on the test data.
    - Returns PAN metrics (auc, acc, c@1) for the train
      data and for the test data, given the *optimal*
      p1/p2 found for the train data.
            
        + Note: apart from the hyperparameters specified,
          only p1/p2 are truly fitted on the development data, 
          because even e.g. the vectorizer is re-fitted
          on the test corpus.

    """

    ### TRAIN PHASE ########################################
    # preprocess:
    dev_train_data, dev_test_data = load_pan_dataset(corpus_dir+'train')
    dev_train_labels, dev_train_documents = zip(*dev_train_data)
    dev_test_labels, dev_test_documents = zip(*dev_test_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi=mfi,
                            vector_space=vector_space,
                            ngram_type=ngram_type,
                            ngram_size=ngram_size,
                            min_df=min_df)
    dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
    dev_test_X = vectorizer.transform(dev_test_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_train_labels+dev_test_labels)
    dev_train_y = label_encoder.transform(dev_train_labels)
    dev_test_y = label_encoder.transform(dev_test_labels)
    
    # fit development vectorizer
    dev_verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    dev_verifier.fit(dev_train_X, dev_train_y)
    dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                          test_y = dev_test_y,
                                          nb_imposters = nb_imposters)
    
    # fit the score shifter on the dev data:
    dev_gt_scores = load_ground_truth(
                        filepath=os.sep.join((corpus_dir, 'train', 'truth.txt')),
                        labels=dev_test_labels)
    shifter = ScoreShifter()
    shifter.fit(predicted_scores=dev_test_scores,
                ground_truth_scores=dev_gt_scores)
    dev_test_scores = shifter.transform(dev_test_scores)
    
    # calculate scores:
    dev_acc_score, dev_auc_score, dev_c_at_1_score = \
            pan_metrics(prediction_scores=dev_test_scores,
                        ground_truth_scores=dev_gt_scores)

    ### TEST PHASE ########################################
    # preprocess:
    train_data, test_data = load_pan_dataset(corpus_dir+'test')
    train_labels, train_documents = zip(*train_data)
    test_labels, test_documents = zip(*test_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi=mfi,
                            vector_space=vector_space,
                            ngram_type=ngram_type,
                            ngram_size=ngram_size)
    train_X = vectorizer.fit_transform(train_documents).toarray()
    test_X = vectorizer.transform(test_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels+test_labels)
    train_y = label_encoder.transform(train_labels)
    test_y = label_encoder.transform(test_labels)
    
    # fit and predict new verifier with test data:
    test_verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    test_verifier.fit(train_X, train_y)
    test_scores = test_verifier.predict_proba(test_X=test_X,
                                          test_y=test_y,
                                          nb_imposters = nb_imposters)
    
    # apply the optimzed score shifter:
    test_gt_scores = load_ground_truth(
                        filepath=os.sep.join((corpus_dir, 'test', 'truth.txt')),
                        labels=test_labels)
    test_scores = shifter.transform(test_scores)
    
    # calculate scores:
    test_acc_score, test_auc_score, test_c_at_1_score = \
            pan_metrics(prediction_scores=test_scores,
                        ground_truth_scores=test_gt_scores)

    print('::::: Test scores ::::::')
    print('Test AUC:', test_auc_score)
    print('Test acc:', test_acc_score)
    print('Test c@1:', test_c_at_1_score)
    
    return dev_auc_score, dev_acc_score, dev_c_at_1_score,\
           test_acc_score, test_auc_score, test_c_at_1_score


def test_experiment(corpus_dir, mfi, vector_space,
               ngram_type, ngram_size, metric,
               base, nb_bootstrap_iter, rnd_prop,
               nb_imposters, p1, p2, min_df):
    """

    - Runs verification experiment on the test folder under
      `corpus_dir`, given the hyperparameters specified,
      incl. p1 and p2.
    - Returns the auc, acc and c@1 for the test data, as well as
      the actual scores predicted and the associated ground truth.

    """

    # initialize score shifter:
    shifter = ScoreShifter()
    shifter.optimal_p1 = p1
    shifter.optimal_p2 = p2
    
    ### TEST PHASE ########################################
    # preprocess:
    train_data, test_data = load_pan_dataset(corpus_dir+'test')
    train_labels, train_documents = zip(*train_data)
    test_labels, test_documents = zip(*test_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi = mfi,
                            vector_space = vector_space,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size,
                            min_df = min_df)
    train_X = vectorizer.fit_transform(train_documents).toarray()
    test_X = vectorizer.transform(test_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels+test_labels)
    train_y = label_encoder.transform(train_labels)
    test_y = label_encoder.transform(test_labels)
    
    # fit and predict a verifier on the test data:
    test_verifier = Verifier(metric = metric,
                             base = base,
                             nb_bootstrap_iter = nb_bootstrap_iter,
                             rnd_prop = rnd_prop)
    test_verifier.fit(train_X, train_y)
    test_scores = test_verifier.predict_proba(test_X=test_X,
                                              test_y=test_y,
                                              nb_imposters = nb_imposters)
    
    # load the ground truth:
    test_gt_scores = load_ground_truth(
                        filepath=os.sep.join((corpus_dir, 'test', 'truth.txt')),
                        labels=test_labels)
    
    # apply the optimzed score shifter:
    test_scores = shifter.transform(test_scores)
    
    test_acc_score, test_auc_score, test_c_at_1_score = \
            pan_metrics(prediction_scores=test_scores,
                        ground_truth_scores=test_gt_scores)
    
    print('::::: Test scores ::::::')
    print('Test AUC:', test_auc_score)
    print('Test acc:', test_acc_score)
    print('Test c@1:', test_c_at_1_score)
    
    return test_auc_score, test_acc_score, test_c_at_1_score, \
           test_scores, test_gt_scores
    