from __future__ import print_function

import os
from scipy.spatial.distance import pdist, squareform

import numpy as np
from utilities import load_pan_dataset, load_ground_truth, train_dev_split
from vectorization import Vectorizer
from score_shifting import ScoreShifter
from distance_metrics import minmax
from evaluation import SADPs_DADPs, distributions, pan_metrics
from visualization import pca, tsne

from verification import Verifier
from sklearn.preprocessing import LabelEncoder

# preprocess:
corpus_dir = '../data/2014/du_essays'
train_data, test_data = load_pan_dataset(corpus_dir)
train_labels, train_documents = zip(*train_data)
test_labels, test_documents = zip(*test_data)

# vectorize:
vectorizer = Vectorizer(mfi=100000, vector_space='tf_std', ngram_type='char', ngram_size=4)
train_X = vectorizer.fit_transform(train_documents).toarray()
test_X = vectorizer.transform(test_documents).toarray()

# encode author labels:
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_labels)
test_y = label_encoder.transform(test_labels)

# create a train and dev set from the training data
# (bit verbose, due to specific PAN evaluation framework):
tmp_train_X, tmp_train_y, \
dev_X, dev_y, dev_labels, dev_ground_truth = \
                    train_dev_split(train_X=train_X,
                                    train_y=train_y)

dev_verifier = Verifier(metric='minmax',
                    base='instance', # or 'profile'
                    nb_bootstrap_iter=100,
                    rnd_prop=0.5)
dev_verifier.fit(tmp_train_X, tmp_train_y)
prediction_scores = dev_verifier.predict_proba(test_X=dev_X,
                    test_y=dev_y,
                    nb_imposters=30,
                    method='m1')

shifter = ScoreShifter()
shifter.fit(scores=prediction_scores,
                    labels=dev_labels,
                    ground_truth=dev_ground_truth)

test_verifier = Verifier(metric='minmax',
                    base='instance', # or 'profile'
                    nb_bootstrap_iter=100,
                    rnd_prop=0.5)
test_verifier.fit(train_X, train_y)

test_scores = test_verifier.predict_proba(test_X=test_X,
                                      test_y=test_y,
                                      nb_imposters=30,
                                      method='m1')
test_scores = shifter.transform(test_scores)

test_ground_truth = load_ground_truth(filepath=os.sep.join((corpus_dir, 'truth.txt')))

acc_score, auc_score, c_at_1_score = \
        pan_metrics(prediction_scores=test_scores,
                    labels=test_labels,
                    ground_truth=test_ground_truth)
print('Test AUC:', auc_score)
print('Test acc:', acc_score)
print('Test c@1:', c_at_1_score)


"""
# explore train data:
train_dm = squareform(pdist(train_X, minmax))
SADPs, DADPs = SADPs_DADPs(train_dm, train_labels, trim_DADPs=True)
DPs = np.asarray(list(SADPs)+list(DADPs))
min_dist, max_dist = np.min(DPs), np.max(DPs)
SADPs = (SADPs-min_dist) / (max_dist - min_dist)
DADPs = (DADPs-min_dist) / (max_dist - min_dist)
print(min_dist, max_dist)
distributions(SADPs, DADPs)
"""

#F1, D = evaluate(train_data)
#pca(X=train_dm, labels=train_labels, ints=train_y, feature_names=vectorizer.feature_names)
#tsne(X=train_dm, labels=train_labels, ints=train_y)
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=1, metric='cityblock')
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean', algorithm='brute', leaf_size=1, p=1)





