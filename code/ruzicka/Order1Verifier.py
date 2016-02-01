#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Module containing the Verifier-class to perform authorship verification
in the General Imposters (GI) framework. Using sklearn-conventions,
the Verifier-object offers a generic implementation of the algorithm
described in e.g.:
  - M. Koppel and Y. Winter (2014), Determining if Two Documents are
    by the Same Author, JASIST, 65(1): 178-187
  - Stover, J. A. , Y. Winter, M. Koppel, M. Kestemont (2015).
    Computational Authorship Verification Method Attributes New Work
    to Major 2nd Century African Author, JASIST,  doi: 10.1002/asi.23460.
  - ...

"""

from __future__ import print_function
import random
from itertools import combinations

import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# legacy: set metrics for using the theano functions
import tensor
GPU_METRICS = {'manhattan' : tensor.get_manhattan_fn(),
               'euclidean' : tensor.get_euclidean_fn(),
               'minmax' : tensor.get_minmax_fn()}

# import the pairwise distance functions:
from test_metrics import minmax, manhattan, euclidean, common_ngrams, cosine
CPU_METRICS = {'manhattan' : manhattan,
               'euclidean' : euclidean,
               'minmax' : minmax,
               'cng' : common_ngrams,
               'cosine' : cosine,
               }

class Order1Verifier:
    """

    Offers a generic implementation a generic implementation
    of the authorship verification algorithm described in e.g.:
      - M. Koppel and Y. Winter (2014), Determining if Two Documents are
        by the Same Author, JASIST, 65(1): 178-187
      - Stover, J. A. , Y. Winter, M. Koppel, M. Kestemont (2015).
        Computational Authorship Verification Method Attributes New Work
        to Major 2nd Century African Author, JASIST,  doi: 10.1002/asi.23460.
      - ...

    The object follow sklearn-like conventions, offering `fit()` and
    `predict_proba()`.

    """

    def __init__(self, metric='manhattan', base='profile',
                 random_state=1066, device='cpu'):
        """
        Constructor.

        Parameters
        ----------
        metric: str
            The distance metric used; should be one of:
                + minmax
                + manhattan
                + euclidean

        base, str
            Indicates whether to use an instance-based or
            profile-based approach for each author; should
            be 'profile' or 'instance'.

        nb_bootstrap_iter: int, default=100
            Indicates the number of bootstrap iterations to
            be used (e.g. 100). If this evaluates to False,
            we run a naive version of the imposter algorithm
            without bootstrapping; i.e. we simply check once
            whether the target author appears to be a test
            document's nearest neighbour among the imposters).
    
        random_seed: int, default=1066
            Integer used for seeding the random streams.

        rnd_prop: scalar, default=.5
            Float specifying the number of features to be
            randomly sampled in each iteration.

        device: str, default='cpu'
            Indicating whether we use the theano- or JIT-
            accelerated distance computations. (For the 
            paper, we eventually used the numba-version
            throughout.)
        
        """

        # some sanity checks:
        if base != 'profile':
            raise NotImplementedError
        self.base = base

        # set rnd seeds:
        random.seed(1066)
        self.rnd = np.random.RandomState(random_state)

        # check with we use JIT- or theano-metrics:
        if device == 'cpu':
            self.metric_fn = CPU_METRICS[metric]
        elif device == 'gpu':
            self.metric_fn = GPU_METRICS[metric]


    def fit(self, X, y):
        """
        Runs very light, memory-based like fitting Method
        which primarily stores `X` and `y` in memory. In the
        case of profile-based verifier, we store a single,
        mean centroid per author in memory.

        Parameters
        ----------
        X: floats, array-like [nb_documents, nb_features]
            The 2D matrix representing the training instance-based
            to be memorized.

        y, array of ints [nb_documents]
            An int-encoded representation of the correct authorship
            for each training documents.

        References
        ----------
        - Daelemans, W. & van den Bosch, A. (2005). Memory-Based
          Language Processing. Cambridge University Press.
        - M. Koppel and S. Seidman (2013), Automatically
          Identifying Pseudepigraphic Texts, EMNLP-13: 1449-1454.

        """

        self.train_X = NearestCentroid().fit(X, y).centroids_ # mean centroids
        self.train_y = np.array(range(self.train_X.shape[0]))

        nb_items = self.train_X.shape[0]

        # calculate all pairwise distances in data set:
        distances = []
        idxs = range(self.train_X.shape[0])
        for i, j in combinations(range(nb_items), 2):
            distances.append(self.metric_fn(self.train_X[i],
                                            self.train_X[j],
                                            idxs))

        # fit a 0-1 scaler on the distances:
        distances = np.array(distances, dtype='float32').transpose()
        distances = distances[~np.isnan(distances)]
        self.distance_scaler1 = StandardScaler().fit(distances)
        distances = self.distance_scaler1.transform(distances.transpose())
        self.distance_scaler2 = MinMaxScaler().fit(distances)


    def dist_closest_target(self, test_vector, target_int,
                            rnd_feature_idxs='all'):
        """

        Given a `test_vector` and an integer representing a target
        authors (`target_int`), we retrieve the distance to the
        nearest document by the target author in the training data.
        In the distance calculation, we only take into account
        the feature values specified in `rnd_feature_idxs` (if
        the latter parameter is specified); else, we use the
        entire feature space.

        Parameters
        ----------
        test_vector : array of floats [nb_features]
            The 1D vector representing the vectorized test document
            for a particular problems.

        target_int : int
            An int encoding the target_author against which the
            authorship of `test_vector` has to be verified. This
            author is assumed to have at least one document in the
            training data.

        rnd_feature_idxs : list of ints, default='all'
            Integer list, specifying the indices of the feature
            values which are should in the distance calculation.
            If unspecified, we use the entire feature space.

        Returns
        ----------
        dist : float
            The actual distance to the nearest document vector
            in memory by the target author.

        """

        # use entire feature space if necessary:
        if rnd_feature_idxs == 'all': # use entire feature space
            rnd_feature_idxs = range(test_vector.shape[0])

        # calculate distance to nearest neighbour for the
        # target author (which potentially has only 1 item):
        distances = []
        for idx in range(len(self.train_y)):
            if self.train_y[idx] == target_int:
                distances.append(self.metric_fn(self.train_X[idx],
                                                test_vector,
                                                rnd_feature_idxs))
        return min(distances)


    def predict_proba(self, test_X, test_y):
        """

        Given a `test_vector` and an integer representing a target
        authors (`target_int`), we retrieve the distance to the
        nearest document in the training data, which is NOT authored
        by the target author. In the distance calculation, we only
        take into account the feature values specified in
        `rnd_feature_idxs` (if the latter parameter is specified);
        else, we use the entire feature space. Note that we each time
        sample a random number of imposters from the available training 
        documents, the number of which is specified by `nb_imposters`.

        Two routines are distinguished:
        - If self.nb_bootstrap_iter evaluates to `False`, we run a
          naive version of the imposter algorithm without boot-
          strapping; i.e. we simply check once whether the target
          author appears to be a test document's nearest neighbour
          among all imposters, using all features available. In this
          case, all probabilities returned will be `0` or `1`.
        - Else, we apply the normal verification method, using
          self.nb_bootstrap_iter iterations. In this case, the re-
          turned probabilities represent the proportions of bootstraps
          in which the target_author yieled a closer neighbour, than
          any other of the randomly sampled imposter authors.

        Parameters
        ----------
        test_X : floats, array-like [nb_test_problems, nb_features]
            A 2D matrix representing the test documents in vectorized
            format. Should not contain documents that are also present
            in the training data.

        test_y : list of ints [nb_test_problems]
            An int encoding the target_authors for each test problem.
            These are not necessarily the correct author for the test
            documents; only the authors against which the authorship
            of the individual test_documents has to be verified. All
            authors in test_y *must* be present in the training data.

        nb_imposters : int, default=None
            Specifies the number of imposter or distractor documents
            which are randomly sampled from the training documents
            which were not written by the target author.

        Returns
        ----------
        probas : list of floats, array-like [nb_problems]
            A score assigned to each individual verificatio
            problem, indicating the likelihood with which
            the verifier would attrribute `test_X[i]` to
            candidate author `test[i]`.

        Note
        ----------
        It is unwise to directly evaluate the probabilities
        returned by `predict_proba()` using the PAN evaluation
        metrics, since these probabilities do not account for 
        the strict 0.5 cutoff which is used by these metric.
        Use the `ScoreShifter()` in `score_shifting.py` to
        obtain a more sensible estimate in this respect.

        """

        distances = []
        for test_vector, target_int in zip(test_X, test_y):
            target_dist = self.dist_closest_target(test_vector=test_vector,
                                                   target_int=target_int,
                                                   rnd_feature_idxs='all')
            distances.append(target_dist)
        
        distances = np.array(distances, dtype='float32').transpose()
        return 1.0 - self.distance_scaler2.transform(self.distance_scaler1.transform(distances))
