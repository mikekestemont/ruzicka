from __future__ import print_function

import os
import time
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from distance_metrics import pairwise_minmax
from scipy.spatial.distance import squareform, pdist
from visualization import tsne, clustermap, tree


font = {'family' : 'arial', 'size' : 6}
sb.plt.rc('font', **font)

import numpy as np
from utilities import load_pan_dataset
from vectorization import Vectorizer
from sklearn.preprocessing import LabelEncoder

ngram_type = 'word'
ngram_size = 1
mfi = 10000
vector_space = 'tf'

dev_data, _ = load_pan_dataset('../data/latin/dev/')
dev_labels, dev_documents = zip(*dev_data)

# fit vectorizer on dev data:
vectorizer = Vectorizer(mfi=mfi,
                        vector_space=vector_space,
                        ngram_type=ngram_type,
                        ngram_size=ngram_size)
vectorizer.fit(dev_documents)

test_data, _ = load_pan_dataset('../data/latin/test/')
test_labels, test_documents = zip(*test_data)
vectorizer.fit_transform(test_documents)

X = vectorizer.transform(test_documents).toarray()
dm = squareform(pdist(X, pairwise_minmax))
tree(dm, test_labels)

# scale distance matrix:
nonzeroes = dm[dm.nonzero()]
max_ = nonzeroes.max()
min_ = nonzeroes.min()
dm = (dm-min_) / (max_ - min_)
np.fill_diagonal(dm, 0.0)

clustermap(dm, test_labels)




