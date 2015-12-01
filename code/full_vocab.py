"""
For each dataset, this script runs a series of experiments
on the train corpus, using the Imposters Framework.
For all metrics and vector space combinations,
we collect PAN scores for a range of MFI (auc, c@1, etc.).
For each experiment, we score the score shifter's
p1 and p2, which yielded the optimal AUC x c@1.
"""

from __future__ import print_function
import os
import json
import pickle
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from ruzicka.experimentation import dev_experiment
from ruzicka.utilities import get_vocab_size

# set hyperparameters:
corpus_dirs = ['../data/2014/du_essays/',
               '../data/2014/gr_articles/',
               '../data/2014/sp_articles/',
               '../data/2014/du_reviews/',
               '../data/2014/en_essays/',
               '../data/2014/en_novels/']

base = 'instance'
nb_bootstrap_iter = 100
rnd_prop = 0.5
min_df = 1
nb_imposters = 30

feature_types = {'char 3-gram': ('char', 3),
                 'char 4-gram': ('char', 4),
                 'word 1-gram': ('word', 1)}

for corpus_dir in corpus_dirs:
    print('>>> corpus:', corpus_dir)
    
    df = pd.DataFrame(columns=['feature type']+sorted(feature_types))

    for metric in ('manhattan', 'minmax', 'euclidean'):
        print('\t\t>>>', metric)
        
        for vector_space in ('tf_std', 'tf_idf', 'tf'):
            
            print('\t\t\t +++', vector_space)
            combo = metric+' - '+vector_space
            print(combo)

            row = [combo]

            for feature_type in sorted(feature_types):
                # get max nb of features for current feature family:
                ngram_type, ngram_size = feature_types[feature_type]
                max_vocab_size = get_vocab_size(corpus_dir = corpus_dir,
                                    ngram_type = ngram_type,
                                    ngram_size = ngram_size,
                                    min_df = min_df,
                                    phase = 'train')

                dev_auc_score, dev_acc_score, dev_c_at_1_score, opt_p1, opt_p2 = \
                                    dev_experiment(corpus_dir = corpus_dir+'/',
                                                   mfi = max_vocab_size,
                                                   vector_space = vector_space,
                                                   ngram_type = ngram_type,
                                                   ngram_size = ngram_size,
                                                   metric = metric,
                                                   base = base,
                                                   min_df = min_df,
                                                   nb_bootstrap_iter = nb_bootstrap_iter,
                                                   rnd_prop = rnd_prop,
                                                   nb_imposters = nb_imposters)
                row.append(dev_auc_score * dev_c_at_1_score)
            
            df.loc[len(df)] = row

    df.set_index('feature type', inplace=True, drop=True)
    c = os.path.basename(corpus_dir[:-1])
    df.to_excel('../output/'+c+'_'+base+'_full_voc.xlsx')

