from __future__ import print_function

import os

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from experimentation import attribution_experiment

corpus_dirs = ['../data/2014/du_essays',
               #'../data/2014/du_reviews',
               #'../data/2014/en_essays',
               #'../data/2014/en_novels',
              ]
feature_ranges = range(1000, 100000, 5000)

for corpus_dir in corpus_dirs:
    for vector_space in ('tf_std', 'tf_idf'):
        # new plot:
        sb.plt.clf()
        f, ax = plt.subplots(1,1)
        sb.set_style("darkgrid")
        ax.set_ylim(.0, 1)
        for metric in ('minmax', 'euclidean'):
            scores = []
            for nb_feats in feature_ranges:
                print('\t- nb feats:', nb_feats)
                acc_score = attribution_experiment(corpus_dir = '../data/2014/du_essays/',
                                                   mfi = nb_feats,
                                                   vector_space = vector_space,
                                                   ngram_type = 'char',
                                                   ngram_size = 4,
                                                   metric = metric,
                                                   base = 'profile')
                print('::::: Dev score ::::::')
                print('F1:', acc_score)
                scores.append(acc_score)
            maxi_score = format(np.max(np.asarray(scores)), '.1f')
            maxi_pos = feature_ranges[np.argmax(scores)]
            an1 = ax.annotate(metric+"\nmax: "+str(maxi_score),
                      xy=(maxi_pos, maxi_score), xycoords="data",
                      va="center", ha="left", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.6", fc="w"))
            cv = np.std(scores)/np.mean(scores)
            cv = format(cv, '.3f')
            l = metric+' ($\sigma$/$\mu$ :'+cv+')'
            sb.plt.plot(feature_ranges, scores, label=l)
        sb.plt.title(vector_space.replace('_', '-'))
        sb.plt.xlabel('# MFI')
        sb.plt.ylabel('Weighted F1')
        sb.plt.legend(loc='best')
        c = os.path.basename(corpus_dir)
        sb.plt.savefig('../output/'+c+'_'+vector_space+'_attr.pdf')







