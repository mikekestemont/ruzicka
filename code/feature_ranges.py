from __future__ import print_function

import os

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from experimentation import dev_experiment, test_experiment

corpus_dirs = ['../data/2014/du_essays',
               #'../data/2014/du_reviews',
               #'../data/2014/en_essays',
               #'../data/2014/en_novels',
              ]
feature_ranges = range(100, 1000, 500)

for corpus_dir in corpus_dirs:
    for vector_space in ('tf_std', 'tf_idf'):
        # new plot:
        sb.plt.clf()
        f, ax = plt.subplots(1,1)
        sb.set_style("darkgrid")
        ax.set_ylim(.5, 1)
        for metric in ('minmax', 'euclidean'):
            scores = []
            for nb_feats in feature_ranges:
                print('\t- nb feats:', nb_feats)
                dev_auc_score, dev_acc_score, dev_c_at_1_score = \
                                    dev_experiment(corpus_dir = corpus_dir+'/',
                                                   mfi = nb_feats,
                                                   vector_space = vector_space,
                                                   ngram_type = 'char',
                                                   ngram_size = 4,
                                                   metric = metric,
                                                   base = 'profile',
                                                   nb_bootstrap_iter = 100,
                                                   rnd_prop = 0.5,
                                                   nb_imposters = 30,
                                                   method = 'm1')
                print('::::: Dev scores ::::::')
                print('Dev AUC:', dev_auc_score)
                print('Dev acc:', dev_acc_score)
                print('Dev c@1:', dev_c_at_1_score)
                scores.append(dev_acc_score * dev_c_at_1_score)        
            maxi_score = format(np.max(np.asarray(scores)), '.1f')
            maxi_pos = feature_ranges[np.argmax(scores)]
            an1 = ax.annotate(metric+"\nmax: "+str(maxi_score),
                      xy=(maxi_pos, maxi_score), xycoords="data",
                      va="center", ha="left", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.6", fc="w"))
            std = format(np.std(np.asarray(scores)), '.2f')
            mean = format(np.mean(np.asarray(scores)), '.2f')
            l = metric+'($\sigma$:'+std+', $\mu$:'+mean+')'
            sb.plt.plot(feature_ranges, scores, label=l)
        sb.plt.title(vector_space)
        sb.plt.xlabel('# MFI')
        sb.plt.ylabel('AUC x c@1')
        sb.plt.legend(loc='best')
        c = os.path.basename(corpus_dir)
        sb.plt.savefig('../output/'+c+'_'+vector_space+'.pdf')

"""
dev_auc_score, dev_acc_score, dev_c_at_1_score,\
test_acc_score, test_auc_score, test_c_at_1_score = \
                   test_experiment(corpus_dir = '../data/2014/du_essays/',
                                   mfi = 1000,
                                   vector_space = 'tf_std',
                                   ngram_type = 'char',
                                   ngram_size = 4,
                                   metric = 'minmax',
                                   base = 'instance',
                                   nb_bootstrap_iter = 100,
                                   rnd_prop = 0.5,
                                   nb_imposters = 30,
                                   method = 'm1')
print('::::: Dev scores ::::::')
print('Dev AUC:', dev_auc_score)
print('Dev acc:', dev_acc_score)
print('Dev c@1:', dev_c_at_1_score)
print('::::: Test scores ::::::')
print('Test AUC:', test_auc_score)
print('Test acc:', test_acc_score)
print('Test c@1:', test_c_at_1_score)
"""

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





