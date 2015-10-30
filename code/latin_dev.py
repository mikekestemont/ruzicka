from __future__ import print_function

import os
import time
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb

font = {'family' : 'arial', 'size' : 6}
sb.plt.rc('font', **font)

import numpy as np
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from utilities import load_pan_dataset
from vectorization import Vectorizer
from sklearn.preprocessing import LabelEncoder
from utilities import train_dev_split
from utilities import get_vocab_size
from score_shifting import ScoreShifter
from evaluation import pan_metrics
from verification import Verifier

corpus_dirs = ['../data/latin/dev',
               #'../data/2014/du_reviews',
               #'../data/2014/en_essays',
               #'../data/2014/en_novels',
              ]
nb_experiments = 5
ngram_type = 'char'
metric = 'minmax'
ngram_size = 4
base = 'instance'
nb_bootstrap_iter = 100
rnd_prop = 0.5
nb_imposters = 30
mfi = 10000
vector_space = 'tf'
min_df = 2

# create a dict, where we store the
# optimal settings for each metric 
# and space pair:
best_mfi = {}

for corpus_dir in corpus_dirs:
    best_mfi[corpus_dir] = {}
    print('>>> corpus:', corpus_dir)
    # preprocess:
    dev_train_data, _ = load_pan_dataset(corpus_dir)
    dev_train_labels, dev_train_documents = zip(*dev_train_data)
    
    # vectorize:
    vectorizer = Vectorizer(mfi = mfi,
                            vector_space = vector_space,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size)
    dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
    
    # encode author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_train_labels)
    dev_train_y = label_encoder.transform(dev_train_labels)

    train_X, train_y, dev_X, dev_y, dev_gt_scores= train_dev_split(dev_train_X, dev_train_y)
    
    dev_verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    dev_verifier.fit(train_X, train_y)
    dev_test_scores = dev_verifier.predict_proba(test_X = dev_X,
                                          test_y = dev_y,
                                          nb_imposters = nb_imposters)
    
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

# dump best settings for reuse during testing:
with open('../output/best_train_params.json', 'w') as fp:
    json.dump(best_mfi, fp)
