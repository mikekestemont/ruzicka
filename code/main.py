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


# hyperparams:
CORPUS_DIR = '../data/2014/du_essays/'
MFI = 1000
VECTOR_SPACE = 'tf_std'
NGRAM_TYPE = 'char'
NGRAM_SIZE = 4
METRIC = 'minmax'
BASE = 'instance'
NB_BOOTSTRAP_ITER = 100
RND_PROP = 0.5
NB_IMPOSTERS = 30
METHOD = 'm1'

### TRAINING PHASE ########################################
# preprocess:
dev_train_data, dev_test_data = load_pan_dataset(CORPUS_DIR+'train')
dev_train_labels, dev_train_documents = zip(*dev_train_data)
dev_test_labels, dev_test_documents = zip(*dev_test_data)

# vectorize:
vectorizer = Vectorizer(mfi=MFI,
                        vector_space=VECTOR_SPACE,
                        ngram_type=NGRAM_TYPE,
                        ngram_size=NGRAM_SIZE)
dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
dev_test_X = vectorizer.transform(dev_test_documents).toarray()

# encode author labels:
label_encoder = LabelEncoder()
dev_train_y = label_encoder.fit_transform(dev_train_labels)
dev_test_y = label_encoder.transform(dev_test_labels)

dev_verifier = Verifier(metric = METRIC,
                    base = BASE,
                    nb_bootstrap_iter = NB_BOOTSTRAP_ITER,
                    rnd_prop = RND_PROP)
dev_verifier.fit(dev_train_X, dev_train_y)
dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                      test_y = dev_test_y,
                                      nb_imposters = NB_IMPOSTERS,
                                      method = METHOD)

# fit the score shifter
dev_gt_scores = load_ground_truth(
                    filepath=os.sep.join((CORPUS_DIR, 'train', 'truth.txt')),
                    labels=dev_test_labels)

shifter = ScoreShifter()
shifter.fit(predicted_scores=dev_test_scores,
            ground_truth_scores=dev_gt_scores)
dev_test_scores = shifter.transform(dev_test_scores)


dev_acc_score, dev_auc_score, dev_c_at_1_score = \
        pan_metrics(prediction_scores=dev_test_scores,
                    ground_truth_scores=dev_gt_scores)
print('::::: Dev scores ::::::')
print('Dev AUC:', dev_auc_score)
print('Dev acc:', dev_acc_score)
print('Dev c@1:', dev_c_at_1_score)

### TEST PHASE ########################################
# preprocess:
train_data, test_data = load_pan_dataset(CORPUS_DIR+'test')
train_labels, train_documents = zip(*train_data)
test_labels, test_documents = zip(*test_data)

# vectorize:
vectorizer = Vectorizer(mfi=MFI,
                        vector_space=VECTOR_SPACE,
                        ngram_type=NGRAM_TYPE,
                        ngram_size=NGRAM_SIZE)
train_X = vectorizer.fit_transform(train_documents).toarray()
test_X = vectorizer.transform(test_documents).toarray()

# encode author labels:
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_labels)
test_y = label_encoder.transform(test_labels)

test_verifier = Verifier(metric = METRIC,
                    base = BASE,
                    nb_bootstrap_iter = NB_BOOTSTRAP_ITER,
                    rnd_prop = RND_PROP)
test_verifier.fit(train_X, train_y)
test_scores = test_verifier.predict_proba(test_X=test_X,
                                      test_y=test_y,
                                      nb_imposters = NB_IMPOSTERS,
                                      method = METHOD)

test_gt_scores = load_ground_truth(
                    filepath=os.sep.join((CORPUS_DIR, 'test', 'truth.txt')),
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





