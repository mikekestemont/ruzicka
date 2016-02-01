from __future__ import print_function

import os
import time
import json
import pickle
import glob
from itertools import product, combinations

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist

from sklearn.preprocessing import LabelEncoder

from ruzicka.utilities import binarize
from ruzicka.vectorization import Vectorizer
from ruzicka.utilities import load_pan_dataset, train_dev_split, get_vocab_size
from sklearn.cross_validation import train_test_split
from ruzicka.score_shifting import ScoreShifter
from ruzicka.evaluation import pan_metrics
from ruzicka.verification import Verifier
import ruzicka.art as art
from ruzicka.distance_metrics import pairwise_minmax
from ruzicka.visualization import clustermap, tree

from pystyl.analysis import distance_matrix
from pystyl.analysis import vnc_clustering

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


ha_directory = '../data/latin/ha_test'
# get test data:
data, _ = load_pan_dataset(ha_directory) # ignore unknown documents
data = sorted(data)
labels, documents = zip(*data)

print(labels)


# for ha: get labels
labels = []
for author in sorted(os.listdir(ha_directory)):
    path = os.sep.join((ha_directory, author))
    if os.path.isdir(path):
        for filepath in sorted(glob.glob(path+'/*.txt')):
            name = os.path.splitext(os.path.basename(filepath))[0]
            labels.append((author+'-'+name))

ordered = """
spart-hadrian
spart-aelius
capit-antonin
capit-marcaur
capit-ver
vulcac-avid
lamprid-commod            
capit-pertinax
capit-didius        
spart-sever
spart-pescenn
capit-clodalb
spart-caracall
spart-antget
capit-macrin
lamprid-diadum
lamprid-helioga
lamprid-alexsev
capit-maximinii
capit-gordianiii
capit-maxbalb
trebell-valerianii
trebell-galienii
trebell-tyrxxx
trebell-claud
vopisc-aurelian
vopisc-tacit
vopisc-prob
vopisc-firmsat
vopisc-carus
""".split('\n')
ordered = [s.strip() for s in ordered if s.strip()]

ordered_documents, ordered_labels = [], []

for o in ordered:
    #o = o.lower()
    print(o)
    cnt = 0
    for label, (_, document) in zip(labels, data):
        label = label.lower()
        #print('\t', label)
        if label.startswith(o):
            ordered_documents.append(document)
            ordered_labels.append(label)
            cnt += 1
    #print(o, cnt)

#print(ordered_labels)

from pystyl.corpus import Corpus

corpus = Corpus(language='other')
corpus.add_texts(texts=ordered_documents, titles=ordered_labels, target_names=ordered_labels)
corpus.preprocess(alpha_only=True, lowercase=True)
corpus.tokenize()
corpus.vectorize(ngram_type = 'word',
                 ngram_size = 1,
                 vector_space = 'tf',
                 mfi = 10000,
                 min_df = 2)

"""
print('Generating VNC...')
# VNC Analysis:

dm = distance_matrix(corpus, 'minmax')
tree = vnc_clustering(dm, linkage='ward')

from pystyl.visualization import scipy_dendrogram

scipy_dendrogram(corpus=corpus, tree=tree, outputfile='~/Desktop/ha_vnc.pdf',\
                 fontsize=5, color_leafs=False, show=False, save=True, return_svg=False)
"""

"""
print('Generating clustermap...')
# heatmap:
X = corpus.vectorizer.X.toarray()
dm = squareform(pdist(X, pairwise_minmax))
tree(dm, ordered_labels)


# scale distance matrix:
nonzeroes = dm[dm.nonzero()]
max_ = nonzeroes.max()
min_ = nonzeroes.min()
dm = (dm-min_) / (max_ - min_)
np.fill_diagonal(dm, 0.0)

clustermap(dm, ordered_labels)
"""

print("Generate heatmap:")
X = corpus.vectorizer.X.toarray()

dm = squareform(pdist(X, pairwise_minmax))
tree(dm, ordered_labels)


# scale distance matrix:
nonzeroes = dm[dm.nonzero()]
max_ = nonzeroes.max()
min_ = nonzeroes.min()
dm = (dm-min_) / (max_ - min_)
np.fill_diagonal(dm, 0.0)

plt.clf()
# convert X to a pandas dataframe:
df = pd.DataFrame(data=dm, columns=labels)
# extract correlations for a scaled version of X:
df = df.applymap(lambda x:int(x*100000)).corr()
# clustermap plotting:

fig, ax = plt.subplots()

ax = sns.heatmap(df)



fontsize=4
outputfile='../output/heatmap.pdf'
# some aesthetic interventions:
# xlabels:
for idx, label in enumerate(ax.get_xticklabels()):
    label.set_rotation('vertical')
    label.set_fontname('Arial')
    label.set_fontsize(fontsize)
# ylabels:
for idx, label in enumerate(ax.get_yticklabels()):
    label.set_rotation('horizontal')
    label.set_fontname('Arial')
    label.set_fontsize(fontsize)

fig.savefig(outputfile)






