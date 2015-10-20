from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist
from distance_metrics import minmax
import tensor

METRICS = {'manhattan' : tensor.get_manhattan_fn(),
           'euclidean' : tensor.get_euclidean_fn(),
           'minmax' : tensor.get_minmax_fn()}

class Verifier:
    def __init__(self, metric='manhattan', base=False,
                 nb_bootstrap_iter=0, random_state=1066,
                 rnd_prop=.5):
        assert base in ('profile', 'instance')
        self.base = base
        self.nb_bootstrap_iter = nb_bootstrap_iter
        self.rnd = np.random.RandomState(random_state)
        self.rnd_prop = rnd_prop
        self.metric_fn = METRICS[metric]

    def fit(self, X, y):
        print('--- fitting ---')
        if self.base == 'profile':
            vecs_per_centroid = {}
            for vector, label in zip(X, y):
                try:
                    vecs_per_centroid[label].append(vector)
                except KeyError:
                    vecs_per_centroid[label] = [vector]
            for label, vecs in vecs_per_centroid.items():
                vecs_per_centroid[label] = np.asarray(vecs).median(axis=0)
                np.asarray(vecs).mean(axis=0).shape
            self.train_X, self.train_y = [], []
            for label, c in vecs_per_centroid.items():
                self.train_X.append(c)
                self.train_y.append(label)
            self.train_X = np.asarray(self.train_X, dtype='float32')
            self.train_y = np.asarray(self.train_y, dtype='int8')
        elif self.base == 'instance':
            self.train_X = X
            self.train_y = y

    def distance_to_target_class(self, test_vector, target_int, train_X='NA'):
        if not isinstance(train_X, np.ndarray):
            train_X = self.train_X
        target_X = []
        for class_int, vector in zip(self.train_y, train_X):
            if class_int == target_int:
                target_X.append(vector)
        target_X = np.asarray(target_X, dtype='float32')
        distances = self.metric_fn(target_X, test_vector)
        #distances = cdist(target_X, [test_vector], self.metric)
        if distances.shape[0] > 1:
            distance = np.mean(distances)
        else:
            distance = distances[0][0]
        return distance

    def distance_to_closest_target_class_instance(self, test_vector, target_int, train_X='NA'):
        if not isinstance(train_X, np.ndarray):
            train_X = self.train_X
        target_X = []
        for class_int, vector in zip(self.train_y, train_X):
            if class_int == target_int:
                target_X.append(vector)
        target_X = np.asarray(target_X, dtype='float32')
        distances = self.metric_fn(target_X, test_vector)
        #distances = cdist(target_X, [test_vector], self.metric)
        #min_idx = distances.argmin(axis=1)[0]
        min_idx = distances.argmin()
        return distances[min_idx], target_X[min_idx, :]

    def distance_to_closest_non_target_class_instance(self, test_vector, target_int, train_X='NA', nb_imposters=1):
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
        #distances = cdist(target_X, [test_vector], self.metric)
        #min_idx = distances.argmin(axis=1)[0]
        min_idx = distances.argmin()
        return distances[min_idx], target_X[min_idx, :]

    def distance_to_other_class(self, test_vector, target_int, train_X='NA', nb_imposters=0):
        if not isinstance(train_X, np.ndarray):
            train_X = self.train_X
        target_X = []
        for class_int, vector in zip(self.train_y, train_X):
            if class_int != target_int:
                target_X.append(vector)
        target_X = np.asarray(target_X, dtype='float32')
        if nb_imposters:
            imposter_idxs = self.rnd.randint(train_X.shape[0],
                                        size=nb_imposters)
            train_X = train_X[imposter_idxs,:]
        distances = self.metric_fn(target_X, test_vector)
        #distances = cdist(target_X, [test_vector], self.metric)
        if distances.shape[0] > 1:
            distance = np.mean(distances)
        else:
            distance = distances[0][0]
        return distance

    def predict_proba(self, test_X, test_y, nb_imposters=0, method='m1'):
        if not self.nb_bootstrap_iter:
            distances = []
            for test_vector, target_int in zip(test_X, test_y):
                target_dist = self.distance_to_target_class(test_vector=test_vector,
                                                               target_int=target_int)
                if weigh:
                    non_target_dist = self.distance_to_other_class(test_vector=test_vector,
                                                               target_int=target_int,
                                                               nb_imposters=nb_imposters)
                    weighed_dist = target_dist / non_target_dist
                    distances.append(weighed_dist)
                else:
                    distances.append(target_dist)
            return self.distances_to_probabilities(distances=np.asarray(distances),
                                                   extremize=extremize)
        else:
            distances = []
            cnt = 0
            for test_vector, target_int in zip(test_X, test_y):
                cnt += 1
                print('iter:', cnt)
                bootstrap_score = 0
                for i in range(self.nb_bootstrap_iter):
                    # select random features:
                    rnd_feature_idxs = self.rnd.randint(self.train_X.shape[1],
                                                     size=int(self.train_X.shape[1] * self.rnd_prop))
                    impaired_test_vector = test_vector[rnd_feature_idxs]
                    impaired_train_X = self.train_X[:, rnd_feature_idxs]
                    # get distance to closest item from target author:
                    target_dist, closest_target = self.distance_to_closest_target_class_instance(test_vector=impaired_test_vector,
                                                               target_int=target_int,
                                                               train_X=impaired_train_X)
                    non_target_dist, closest_non_target = self.distance_to_closest_non_target_class_instance(test_vector=impaired_test_vector,
                                                                   target_int=target_int,
                                                                   train_X=impaired_train_X,
                                                                   nb_imposters=nb_imposters)
                    _1 = self.metric_fn(np.asarray([closest_target]), closest_non_target)
                    _2 = self.metric_fn(np.asarray([impaired_test_vector]), closest_non_target)
                    if method == 'm1':
                        if target_dist < non_target_dist:
                            bootstrap_score += (1.0 / self.nb_bootstrap_iter)
                    elif method == 'm2':
                        if (target_dist**2) < (_1 * _2):
                            bootstrap_score += (1.0 / self.nb_bootstrap_iter)
                #dist = 1.0 - bootstrap_score # convert similarity to distance
                distances.append(bootstrap_score)
            return np.array(distances)
