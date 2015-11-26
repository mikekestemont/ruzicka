"""
This script collects the optimal settings for each space-metric
combinations and runs a test experiment with these settings on
the test part of the corresponding dataset. Apart from creating
results tables for each data set separately (auc, c@1, etc.),
we carry out a significance test on the predictions returned
by each verifier, using approximate randomization.
"""
from __future__ import print_function

from itertools import product, combinations
import json
import os
import pandas as pd
from ruzicka.experimentation import test_experiment
from ruzicka.utilities import binarize
import ruzicka.art as art

def stringify(i):
    return '+'.join(i[::-1]).replace('_', '-')

settings = json.load(open('../output/best_train_params.json'))

# init result tables:
index, columns = set(), set()
for corpus_dir in settings:
    for vsm in settings[corpus_dir]:
        index.add(vsm)
        for metric in settings[corpus_dir][vsm]:
            columns.add(metric)


for corpus_dir in settings:
    print('\t- '+corpus_dir)
    # initialize score tables:
    auc_df = pd.DataFrame(index=sorted(index),
                      columns=sorted(columns))
    acc_df = pd.DataFrame(index=sorted(index),
                      columns=sorted(columns))
    c_at_1_df = pd.DataFrame(index=sorted(index),
                      columns=sorted(columns))
    score_df = pd.DataFrame(index=sorted(index),
                      columns=sorted(columns))
    
    test_results = {} # create an object to store predicted scores in
    test_gt = {} # create an object to store ground thruth scores in
    for vsm in settings[corpus_dir]:
        print('\t\t+'+vsm)
        if vsm not in test_results:
            test_results[vsm] = {}
            test_gt[vsm] = {}
        for metric in settings[corpus_dir][vsm]:
            print('\t\t\t* '+metric)

            # load best settings:
            p1 = settings[corpus_dir][vsm][metric]['p1']
            p2 = settings[corpus_dir][vsm][metric]['p2']
            mfi = settings[corpus_dir][vsm][metric]['mfi']

            # run the test experiment
            dev_auc_score, dev_acc_score, dev_c_at_1_score, \
            test_scores, test_gt_scores = \
                                    test_experiment(corpus_dir = corpus_dir,
                                                   mfi = mfi,
                                                   vector_space = vsm,
                                                   ngram_type = 'word',
                                                   ngram_size = 1,
                                                   metric = metric,
                                                   base = 'instance',
                                                   nb_bootstrap_iter = 100,
                                                   rnd_prop = 0.5,
                                                   nb_imposters = 30,
                                                   p1 = p1,
                                                   p2 = p2,
                                                   min_df=1)
            auc_df.ix[vsm][metric] = dev_auc_score
            acc_df.ix[vsm][metric] = dev_acc_score
            c_at_1_df.ix[vsm][metric] = dev_c_at_1_score
            score_df.ix[vsm][metric] = dev_auc_score * dev_c_at_1_score
            # store the actual scores
            test_results[vsm][metric] = test_scores
            test_gt[vsm][metric] = test_gt_scores

    # write away score tables:
    table_dir = '../output/tables/'
    if not os.path.isdir(table_dir):
        os.mkdir(table_dir)

    corpus_name = os.path.basename(corpus_dir[:-1]) # remove trailing slash
    auc_df.to_csv(table_dir+corpus_name+'_auc.csv')
    acc_df.to_csv(table_dir+corpus_name+'_acc.csv')
    c_at_1_df.to_csv(table_dir+corpus_name+'_c_at_1.csv')
    score_df.to_csv(table_dir+corpus_name+'_score.csv')

    # now significance testing:
    combs = sorted([i for i in product(index, columns)])
    str_combs = sorted([stringify(i) for i in combs])
    signif_df = pd.DataFrame(index=str_combs, columns=str_combs).fillna('')
    comb_combs = sorted([i for i in combinations(combs, 2)])

    for comb1, comb2 in comb_combs:
        system1 = binarize(test_results[comb1[0]][comb1[1]])
        system2 = binarize(test_results[comb2[0]][comb2[1]])
        gold = binarize(test_gt[comb1[0]][comb1[1]])
        
        system1, system2, gold, common, common_gold = \
                art.getdifference(system1, system2, gold)
        s = art.labelingsignificance(gold, system1, system2, N=100000,
                common=common, common_gold=common_gold,
                verbose=False, training=None)
        diff_f1 = s['accuracy']
        s = ''
        if diff_f1 < 0.001:
            s = '***'
        elif diff_f1 < 0.01:
            s = '**'
        elif diff_f1 < 0.05:
            s = '*'
        else:
            s = '='
        k1, k2 = stringify(comb1), stringify(comb2)
        signif_df[k1][k2] = s
        signif_df[k2][k1] = s
    
    signif_df.to_csv(table_dir+corpus_name+'_signif.csv')



            

