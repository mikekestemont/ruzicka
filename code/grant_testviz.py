from __future__ import print_function
import os
import time
import json
import pickle

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.spatial.distance import squareform, pdist
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ruzicka.distance_metrics import pairwise_minmax
from ruzicka.visualization import clustermap, tree
from ruzicka.utilities import load_pan_dataset
from ruzicka.vectorization import Vectorizer

ngram_type = 'word'
ngram_size = 1
mfi = 300
vector_space = 'tf_std'

test_data, _ = load_pan_dataset('../data/grant/foreground/')
test_labels, test_documents = zip(*test_data)

# fit vectorizer on dev data:
vectorizer = Vectorizer(mfi=mfi,
                        max_df = .30,
                        vector_space=vector_space,
                        ngram_type=ngram_type,
                        ngram_size=ngram_size)

X = vectorizer.fit_transform(test_documents).toarray()
dm = squareform(pdist(X, pairwise_minmax))
tree(dm, test_labels)

# scale distance matrix:
nonzeroes = dm[dm.nonzero()]
max_ = nonzeroes.max()
min_ = nonzeroes.min()
dm = (dm-min_) / (max_ - min_)
np.fill_diagonal(dm, 0.0)

import glob
directory = '../data/grant/foreground'
labels = []
for author in sorted(os.listdir(directory)):
    path = os.sep.join((directory, author))
    if os.path.isdir(path):
        for filepath in sorted(glob.glob(path+'/*.txt')):
            name = os.path.splitext(os.path.basename(filepath))[0]
            labels.append((author+'-'+name))

clustermap(dm, labels)




