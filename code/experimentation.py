from __future__ import print_function

import os
import time
from collections import Counter

import numpy as np

from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from utilities import load_pan_dataset, load_ground_truth
from vectorization import Vectorizer
from score_shifting import ScoreShifter
from evaluation import pan_metrics
from verification import Verifier

# import distance metrics
from distance_metrics import pairwise_minmax
from scipy.spatial.distance import euclidean as pairwise_euclidean
from scipy.spatial.distance import cityblock as pairwise_manhattan

def attribution_experiment(corpus_dir, mfi, vector_space,
                           ngram_type, ngram_size, metric, base):
    """
    * Runs a naive attribution experiment on the text
      in the train data.
    * Returns `accuracy` after a leave-one-text-out
      validation experiment.
    * Because of this setup, we only include texts
      from authors which have at least two texts
      in the training (excluding the unknown.txt).
    * As classifier, we use the `neighbors` module:
        + KNeighborsClassifier (instance-based, brute
          implementation of IB1-algorithm, with n=1).
        + NearestCentroid (`profile`, with mean
          centroids per author class and no shrinking).
    """
    start_time = time.time()
    METRICS = {'manhattan':pairwise_manhattan,
               'euclidean':pairwise_euclidean,
               'minmax':pairwise_minmax,
              }
    pairwise_metric_fn = METRICS[metric]
    
    ### TRAINING PHASE ########################################
    # preprocess:
    dev_train_data, _ = load_pan_dataset(corpus_dir+'train')
    dev_train_labels, dev_train_documents = zip(*dev_train_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi=mfi,
                            vector_space=vector_space,
                            ngram_type=ngram_type,
                            ngram_size=ngram_size)
    dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_train_labels)
    dev_train_y = label_encoder.transform(dev_train_labels)

    # only select texts from authors with more than 1 text:
    more_than_one = set(i for i,c in Counter(dev_train_y).items() if i > 1)
    train_X, train_y = [], []
    for i in range(dev_train_X.shape[0]):
        if dev_train_y[i] in more_than_one:
            train_X.append(dev_train_X[i])
            train_y.append(dev_train_y[i])

    dev_train_X = np.asarray(train_X)
    dev_train_y = np.asarray(train_y)

    if base == 'instance':
        clf = KNeighborsClassifier(n_neighbors=1,
                                   weights='uniform',
                                   algorithm='brute',
                                   metric=pairwise_metric_fn)
    elif base == 'profile':
        clf = NearestCentroid(shrink_threshold=None,
                              metric=pairwise_metric_fn)

    # simple LOO validation:
    nb_samples = dev_train_X.shape[0]
    cv = cross_validation.LeaveOneOut(nb_samples)
    predictions = cross_validation.cross_val_predict(clf,
                               dev_train_X, dev_train_y, cv=cv)
    weighted_f1 = f1_score(y_true=train_y,
                           y_pred=predictions,
                           average='weighted')
    print("F1 score: %0.3f " % (weighted_f1))
    # return accuracy:
    return weighted_f1


def dev_experiment(corpus_dir, mfi, vector_space,
               ngram_type, ngram_size, metric,
               base, nb_bootstrap_iter, rnd_prop,
               nb_imposters, method):
    """
    * Runs verification experiment on train data only.
    * Artifical: calculatesPAN metrics (auc, acc, c@1),
      given the *optimal* p1/p2 found by the score shifter.
    * Returns: auc, acc, c@1, p1, p2
    """
    start_time = time.time()
    ### TRAINING PHASE ########################################
    # preprocess:
    dev_train_data, dev_test_data = load_pan_dataset(corpus_dir+'train')
    dev_train_labels, dev_train_documents = zip(*dev_train_data)
    dev_test_labels, dev_test_documents = zip(*dev_test_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi=mfi,
                            vector_space=vector_space,
                            ngram_type=ngram_type,
                            ngram_size=ngram_size)
    dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
    dev_test_X = vectorizer.transform(dev_test_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_train_labels+dev_test_labels)
    dev_train_y = label_encoder.transform(dev_train_labels)
    dev_test_y = label_encoder.transform(dev_test_labels)
    
    dev_verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    dev_verifier.fit(dev_train_X, dev_train_y)
    dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                          test_y = dev_test_y,
                                          nb_imposters = nb_imposters,
                                          method = method)
    
    # fit the score shifter
    dev_gt_scores = load_ground_truth(
                        filepath=os.sep.join((corpus_dir, 'train', 'truth.txt')),
                        labels=dev_test_labels)
    
    shifter = ScoreShifter()
    shifter.fit(predicted_scores=dev_test_scores,
                ground_truth_scores=dev_gt_scores)
    dev_test_scores = shifter.transform(dev_test_scores)
    
    
    dev_acc_score, dev_auc_score, dev_c_at_1_score = \
            pan_metrics(prediction_scores=dev_test_scores,
                        ground_truth_scores=dev_gt_scores)
    return dev_auc_score, dev_acc_score, dev_c_at_1_score,\
           shifter.optimal_p1, shifter.optimal_p1


def dev_test_experiment(corpus_dir, mfi, vector_space,
               ngram_type, ngram_size, metric,
               base, nb_bootstrap_iter, rnd_prop,
               nb_imposters, method):
    """
    * Runs verification experiment on train corous,
      and, then, on the actual test data.
      Reports PAN metrics (auc, acc, c@1) for the test data,
      given the *optimal* p1/p2 for the train data.
    * Note: apart from the hyperparameters specified,
      only p1/p2 are truly fitted on the test data, 
      because even e.g. the vectorizer is re-fitted
      on the test corpus.
    """
    start_time = time.time()
    ### TRAIN PHASE ########################################
    # preprocess:
    dev_train_data, dev_test_data = load_pan_dataset(corpus_dir+'train')
    dev_train_labels, dev_train_documents = zip(*dev_train_data)
    dev_test_labels, dev_test_documents = zip(*dev_test_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi=mfi,
                            vector_space=vector_space,
                            ngram_type=ngram_type,
                            ngram_size=ngram_size)
    dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
    dev_test_X = vectorizer.transform(dev_test_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_train_labels+dev_test_labels)
    dev_train_y = label_encoder.transform(dev_train_labels)
    dev_test_y = label_encoder.transform(dev_test_labels)
    
    dev_verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    dev_verifier.fit(dev_train_X, dev_train_y)
    dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                          test_y = dev_test_y,
                                          nb_imposters = nb_imposters,
                                          method = method)
    
    # fit the score shifter
    dev_gt_scores = load_ground_truth(
                        filepath=os.sep.join((corpus_dir, 'train', 'truth.txt')),
                        labels=dev_test_labels)
    
    shifter = ScoreShifter()
    shifter.fit(predicted_scores=dev_test_scores,
                ground_truth_scores=dev_gt_scores)
    dev_test_scores = shifter.transform(dev_test_scores)
    
    
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
    
    
    test_verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    
    test_verifier.fit(train_X, train_y)
    
    test_scores = test_verifier.predict_proba(test_X=test_X,
                                          test_y=test_y,
                                          nb_imposters = nb_imposters,
                                          method = method)
    
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
    
    end_time = time.time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
    return dev_auc_score, dev_acc_score, dev_c_at_1_score,\
           test_acc_score, test_auc_score, test_c_at_1_score

def test_experiment(corpus_dir, mfi, vector_space,
               ngram_type, ngram_size, metric,
               base, nb_bootstrap_iter, rnd_prop,
               nb_imposters, method, p1, p2):
    """
    * Runs verification experiment on train data only.
    * Artifical: calculatesPAN metrics (auc, acc, c@1),
      given the *optimal* p1/p2 found by the score shifter.
    * Returns: auc, acc, c@1, p1, p2
    """
    start_time = time.time()

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
    
    
    test_verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    
    test_verifier.fit(train_X, train_y)
    
    test_scores = test_verifier.predict_proba(test_X=test_X,
                                          test_y=test_y,
                                          nb_imposters = nb_imposters,
                                          method = method)
    
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
    
    end_time = time.time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
    return test_auc_score, test_acc_score, test_c_at_1_score, test_scores


    