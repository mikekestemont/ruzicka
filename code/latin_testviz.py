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
from visualization import tsne, clustermap


font = {'family' : 'arial', 'size' : 6}
sb.plt.rc('font', **font)

import numpy as np
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from utilities import load_pan_dataset
from vectorization import Vectorizer
from sklearn.preprocessing import LabelEncoder

corpus_dir = '../data/latin/test'
ngram_type = 'word'
ngram_size = 1
mfi = 10000
vector_space = 'tf_std'

print('>>> corpus:', corpus_dir)
# preprocess:
test_data, _ = load_pan_dataset(corpus_dir)
labels, documents = zip(*test_data)

# vectorize:
vectorizer = Vectorizer(mfi=mfi,
                        vector_space=vector_space,
                        ngram_type=ngram_type,
                        ngram_size=ngram_size)
X = vectorizer.fit_transform(documents).toarray()
dm = squareform(pdist(X, pairwise_minmax))
# scale distance matrix:
max_ = X.max()
min_ = X.min()
dm = (dm-min_) / (max_ - min_)

clustermap(dm, labels)


