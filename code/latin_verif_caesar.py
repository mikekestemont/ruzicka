from __future__ import print_function

import os
import time
import json
import pickle
from itertools import product, combinations

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from ruzicka.utilities import binarize
from ruzicka.vectorization import Vectorizer
from ruzicka.utilities import load_pan_dataset, train_dev_split, get_vocab_size
from sklearn.cross_validation import train_test_split
from ruzicka.score_shifting import ScoreShifter
from ruzicka.evaluation import pan_metrics
from ruzicka.verification import Verifier
import ruzicka.art as art

# run script for top-5 metrics
ngram_type = 'word'
ngram_size = 1
base = 'profile'
vector_space = 'tf'
metric = 'minmax'
nb_bootstrap_iter = 100
rnd_prop = 0.5
nb_imposters = 30
mfi = 10000
min_df = 2

# get imposter data:
train_data, _ = load_pan_dataset('../data/latin/dev') # ignore unknown documents
train_labels, train_documents = zip(*train_data)

# get test data:
test_data, _ = load_pan_dataset('../data/latin/test') # ignore unknown documents
test_labels, test_documents = zip(*test_data)

# fit encoder for author labels:
label_encoder = LabelEncoder()
label_encoder.fit(train_labels+test_labels)
train_ints = label_encoder.transform(train_labels)
test_ints = label_encoder.transform(test_labels)

# fit vectorizer:
vectorizer = Vectorizer(mfi = mfi,
                        vector_space = vector_space,
                        ngram_type = ngram_type,
                        ngram_size = ngram_size)
vectorizer.fit(train_documents+test_documents)
train_X = vectorizer.transform(train_documents).toarray()
test_X = vectorizer.transform(test_documents).toarray()

cols = ['label']
for test_author in sorted(set(test_ints)):
    auth_label = label_encoder.inverse_transform([test_author])[0]
    cols.append(auth_label)

proba_df = pd.DataFrame(columns=cols)

for idx in range(len(test_documents)):
    target_auth = test_ints[idx]
    target_docu = test_X[idx]
    non_target_test_ints = np.array([test_ints[i] for i in range(len(test_ints)) if i != idx])
    non_target_test_X = np.array([test_X[i] for i in range(len(test_ints)) if i != idx])
    tmp_train_X = np.vstack((train_X, non_target_test_X))
    tmp_train_y = np.hstack((train_ints, non_target_test_ints))
    
    tmp_test_X, tmp_test_y = [], []
    for t_auth in sorted(set(test_ints)):
        tmp_test_X.append(target_docu)
        tmp_test_y.append(t_auth)

    # fit the verifier:
    verifier = Verifier(metric = metric,
                        base = base,
                        nb_bootstrap_iter = nb_bootstrap_iter,
                        rnd_prop = rnd_prop)
    verifier.fit(tmp_train_X, tmp_train_y)
    probas = verifier.predict_proba(test_X = tmp_test_X,
                        test_y = tmp_test_y,
                        nb_imposters = nb_imposters)
    
    row = [label_encoder.inverse_transform([target_auth])[0]] # author label
    row += list(probas)
    print(row)
    proba_df.loc[len(proba_df)] = row

proba_df = proba_df.set_index('label')

# write away score tables:
table_dir = '../output/tables/'
if not os.path.isdir(table_dir):
    os.mkdir(table_dir)
proba_df.to_csv(table_dir+'lat_proba_'+metric+'_'+vector_space+'.csv')

cm = sns.clustermap(proba_df)
ax = cm.ax_heatmap
# ylabels:
for idx, label in enumerate(ax.get_yticklabels()):
    label.set_rotation('horizontal')
    label.set_fontname('Arial')
    label.set_fontsize(5)
cm.savefig('../output/caesar_verif_heatmap.pdf')
