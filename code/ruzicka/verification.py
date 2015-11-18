from __future__ import print_function

import numpy as np
import tensor
import random
random.seed(1066)

from sklearn.neighbors import NearestCentroid


GPU_METRICS = {'manhattan' : tensor.get_manhattan_fn(),
           'euclidean' : tensor.get_euclidean_fn(),
           'minmax' : tensor.get_minmax_fn()}

from test_metrics import minmax, manhattan, euclidean

CPU_METRICS = {'manhattan' : manhattan,
           'euclidean' : euclidean,
           'minmax' : minmax}

class Verifier:
    def __init__(self, metric='manhattan', base=False,
                 nb_bootstrap_iter=0, random_state=1066,
                 rnd_prop=.5, device='cpu'):

        # some sanity checks:
        assert base in ('profile', 'instance')
        assert (rnd_prop > 0.0) and (rnd_prop < 1.0)
        
        self.base = base
        self.nb_bootstrap_iter = nb_bootstrap_iter
        self.rnd = np.random.RandomState(random_state)
        self.rnd_prop = rnd_prop
        if device == 'cpu':
            self.metric_fn = CPU_METRICS[metric]
        elif device == 'gpu':
            self.metric_fn = GPU_METRICS[metric]

    def fit(self, X, y):
        if self.base == 'instance':
            self.train_X = X
            self.train_y = y

        elif self.base == 'profile':
            self.train_X = NearestCentroid().fit(X, y).centroids_ # mean centroids
            self.train_y = np.array(range(self.train_X.shape[0]))

    def dist_closest_target(self, test_vector, target_int, train_X='NA', rnd_feature_idxs='all'):
        if rnd_feature_idxs == 'all':
            rnd_feature_idxs = range(test_vector.shape[0])
        distances = []
        for idx in range(len(self.train_y)):
            if self.train_y[idx] == target_int:
                distances.append(self.metric_fn(self.train_X[idx],
                                        test_vector,
                                        rnd_feature_idxs))
        return min(distances)

    def dist_closest_non_target(self, test_vector, target_int, train_X='NA', rnd_feature_idxs='all', nb_imposters=None):
        distances = []
        if rnd_feature_idxs == 'all':
            rnd_feature_idxs = range(test_vector.shape[0])
        non_target_idxs = [i for i in range(len(self.train_y)) if self.train_y[i] != target_int]
        random.shuffle(non_target_idxs)
        for idx in non_target_idxs[:nb_imposters]:
            distances.append(self.metric_fn(self.train_X[idx],
                                    test_vector,
                                    rnd_feature_idxs))
        return min(distances)

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
                    target_dist = self.dist_closest_target(test_vector=test_vector,
                                                           target_int=target_int,
                                                           rnd_feature_idxs=rnd_feature_idxs)
                    non_target_dist = self.dist_closest_non_target(test_vector=test_vector,
                                                                   target_int=target_int,
                                                                   nb_imposters=nb_imposters,
                                                                   rnd_feature_idxs=rnd_feature_idxs)
                    if target_dist < non_target_dist:
                        bootstrap_score += (1.0 / self.nb_bootstrap_iter)
                distances.append(bootstrap_score)

        return np.array(distances, dtype='float32')
