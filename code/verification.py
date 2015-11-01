from __future__ import print_function

import numpy as np
import tensor

from sklearn.neighbors import NearestCentroid

METRICS = {'manhattan' : tensor.get_manhattan_fn(),
           'euclidean' : tensor.get_euclidean_fn(),
           'minmax' : tensor.get_minmax_fn()}

class Verifier:
    def __init__(self, metric='manhattan', base=False,
                 nb_bootstrap_iter=0, random_state=1066,
                 rnd_prop=.5):

        # some sanity checks:
        assert base in ('profile', 'instance')
        assert (rnd_prop > 0.0) and (rnd_prop < 1.0)
        assert metric in METRICS

        self.base = base
        self.nb_bootstrap_iter = nb_bootstrap_iter
        self.rnd = np.random.RandomState(random_state)
        self.rnd_prop = rnd_prop
        self.metric_fn = METRICS[metric]

    def fit(self, X, y):
        if self.base == 'instance':
            self.train_X = np.asarray(X)
            self.train_y = np.asarray(y)

        elif self.base == 'profile':
            self.train_X = NearestCentroid().fit(X, y).centroids_ # mean centroids
            self.train_y = np.array(range(self.train_X.shape[0]))

    def dist_closest_target(self, test_vector, target_int, train_X='NA'):
        if not isinstance(train_X, np.ndarray):
            train_X = self.train_X
        target_X = []
        for class_int, vector in zip(self.train_y, train_X):
            if class_int == target_int:
                target_X.append(vector)
        target_X = np.asarray(target_X, dtype='float32')
        distances = self.metric_fn(target_X, test_vector)
        min_idx = distances.argmin()
        return distances[min_idx]

    def dist_closest_non_target(self, test_vector, target_int, train_X='NA', nb_imposters=1):
        if not isinstance(train_X, np.ndarray):
            train_X = self.train_X
        target_X = []
        for class_int, vector in zip(self.train_y, train_X):
            if class_int != target_int:
                target_X.append(vector)
        target_X = np.asarray(target_X, dtype='float32')

        # select random imposters:
        rnd_imposters_idxs = self.rnd.randint(target_X.shape[0],
                            size=nb_imposters)
        target_X = target_X[rnd_imposters_idxs, :]
        # calculate distance to the imposters:
        distances = self.metric_fn(target_X, test_vector)
        min_idx = distances.argmin()
        return distances[min_idx]


    def predict_proba(self, test_X, test_y, nb_imposters=0):
        distances = []
        if not self.nb_bootstrap_iter: # naive verification:
            for test_vector, target_int in zip(test_X, test_y):
                    target_dist = self.dist_closest_target(test_vector=test_vector,
                                                           target_int=target_int)
                    non_target_dist = self.dist_closest_non_target(test_vector=test_vector,
                                                           target_int=target_int)
                    if target_dist < non_target_dist:
                        distances.append(1.0)
                    else:
                        distances.append(0.0)

        else: # bootstrapped, imposter-based, verification:
            cnt = 0
            for test_vector, target_int in zip(test_X, test_y):

                cnt += 1
                if cnt % 10 == 0:
                    print('\t - # test documents processed:', cnt, 'out of', len(test_y))

                bootstrap_score = 0.0
                for i in range(self.nb_bootstrap_iter):
                    # select random features:
                    rnd_feature_idxs = self.rnd.randint(self.train_X.shape[1],
                                                     size=int(self.train_X.shape[1] * self.rnd_prop))
                    impaired_test_vector = test_vector[rnd_feature_idxs]
                    impaired_train_X = self.train_X[:, rnd_feature_idxs]

                    # get distance to closest (non) target:
                    target_dist = self.dist_closest_target(test_vector=impaired_test_vector,
                                                           target_int=target_int,
                                                           train_X=impaired_train_X)
                    non_target_dist = self.dist_closest_non_target(test_vector=impaired_test_vector,
                                                                   target_int=target_int,
                                                                   train_X=impaired_train_X,
                                                                   nb_imposters=nb_imposters)
                    if target_dist < non_target_dist:
                        bootstrap_score += (1.0 / self.nb_bootstrap_iter)
                distances.append(bootstrap_score)

        return np.array(distances, dtype='float32')
