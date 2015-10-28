from __future__ import print_function

import os
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb

font = {'family' : 'arial', 'size' : 6}
sb.plt.rc('font', **font)

import numpy as np
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from experimentation import dev_experiment

corpus_dirs = ['../data/2014/du_essays',
               #'../data/2014/du_reviews',
               #'../data/2014/en_essays',
               #'../data/2014/en_novels',
              ]
feature_ranges = range(1000, 200000, 5000)

best_mfi = {}

for corpus_dir in corpus_dirs:
    best_mfi[corpus_dir] = {}
    for vector_space in ('tf_std', 'tf_idf'):
        best_mfi[corpus_dir][vector_space] = {}
        # new plot:
        sb.plt.clf()
        f, ax = plt.subplots(1,1)
        sb.set_style("darkgrid")
        ax.set_ylim(.5, 1)
        for metric in ('minmax', 'euclidean'):
            scores, p1s, p2s = [], [], []
            for nb_feats in feature_ranges:
                print('\t- nb feats:', nb_feats)
                dev_auc_score, dev_acc_score, dev_c_at_1_score, opt_p1, opt_p2 = \
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
                p1s.append(opt_p1)
                p2s.append(opt_p2)

            # determine position of optimal AUC x c@1:
            opt_idx = np.argmax(scores)
            # collect best settings:
            opt_mfi = feature_ranges[opt_idx]
            opt_score = scores[opt_idx]
            opt_p1 = p1s[opt_idx]
            opt_p2 = p2s[opt_idx]
            best = {'score':opt_score, 'mfi':opt_mfi, 'p1':opt_p1, 'p2':opt_p2}
            best_mfi[corpus_dir][vector_space][metric] = best

            # annotate optimum:
            an1 = ax.annotate(metric+"\nmax: "+format(opt_score, '.3f'),
                      xy=(opt_mfi, format(opt_score, '.3f')), xycoords="data",
                      va="center", ha="left", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.6", fc="w"))

            # prepare legend label:
            cv = np.std(scores)/np.mean(scores)
            cv = format(cv, '.3f')
            l = metric+' ($\sigma$/$\mu$ :'+cv+')'

            # plot results:
            sb.plt.plot(feature_ranges, scores, label=l)

        sb.plt.title(vector_space.replace('_', '-'))
        sb.plt.xlabel('# MFI')
        sb.plt.ylabel('AUC $\cdot$ c@1')
        sb.plt.legend(loc='best')
        c = os.path.basename(corpus_dir)
        sb.plt.savefig('../output/'+c+'_'+vector_space+'.pdf')

# dump best settings for reuse during testing:
with open('../output/best_train_params.json', 'w') as fp:
    json.dump(best_mfi, fp)
