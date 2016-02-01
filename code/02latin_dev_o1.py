from __future__ import print_function

import os
import time
import json
import pickle
import sys
from itertools import product, combinations

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

c1, c2, c3, c4 = sb.color_palette("Set1")[:4]

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve

from ruzicka.utilities import binarize
from ruzicka.vectorization import Vectorizer
from ruzicka.utilities import load_pan_dataset, train_dev_split, get_vocab_size
from sklearn.cross_validation import train_test_split
from ruzicka.score_shifting import ScoreShifter
from ruzicka.evaluation import pan_metrics
from ruzicka.Order1Verifier import Order1Verifier as Verifier
import ruzicka.art as art

corpus_dir = '../data/latin/dev'
ngram_type = 'word'
ngram_size = 1
base = 'profile'
mfi = sys.maxint
min_df = 1

# preprocess:
all_data, _ = load_pan_dataset(corpus_dir)
all_labels, all_documents = zip(*all_data)

# make split at author level:
random_state = 1066
dev_documents, dev_labels = [], []
test_documents, test_labels = [], []
authors = list(set(all_labels))
np.random.RandomState(random_state).shuffle(authors)
uni_dev_authors, uni_test_authors = authors[:int(len(authors)/2.0)], authors[int(len(authors)/2.0):]
for i, author in enumerate(all_labels):
    if author in uni_dev_authors:
        dev_documents.append(all_documents[i])
        dev_labels.append(all_labels[i])
    elif author in uni_test_authors:
        test_documents.append(all_documents[i])
        test_labels.append(all_labels[i])

fig = sb.plt.figure()
sb.plt.xlabel("recall", fontsize=10)
sb.plt.ylabel("precision", fontsize=10)
sb.plt.xlim(0.45, 1)
sb.plt.ylim(0.45, 1.05)

vector_spaces = ['tf_std', 'tf_idf', 'tf']
dist_metrics = ['cng', 'manhattan', 'minmax', 'cosine']
combs = sorted([i for i in product(vector_spaces, dist_metrics)])
score_df = pd.DataFrame(index=sorted(combs),
                      columns=sorted(combs))
# collect scores for significance testing
holder = {}
for vector_space in vector_spaces:
    holder[vector_space] = {}
    for metric in dist_metrics:
        holder[vector_space][metric] = "NA"

f_results = open('../output/tables/latin_dev_o1_scores.csv', 'w')
f_results.write('metric,vsm,c@1,AUC,c@1xAUC\n')
for vector_space, metric in combs:
    print('\t +++', vector_space, '+', metric)

    ####### development phase ##################################################################################
    vectorizer = Vectorizer(mfi = mfi,
                            vector_space = vector_space,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size,
                            min_df=min_df)
    dev_X = vectorizer.fit_transform(dev_documents).toarray()
    label_encoder = LabelEncoder()
    label_encoder.fit(dev_labels)
    dev_y = label_encoder.transform(dev_labels)
    dev_train_X, dev_train_y, dev_test_X, \
        dev_test_y, dev_gt_scores = train_dev_split(dev_X, dev_y)
    dev_verifier = Verifier(metric = metric,
                            base = base)
    dev_verifier.fit(dev_train_X, dev_train_y)
    dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                          test_y = dev_test_y)
    shifter = ScoreShifter()
    shifter.fit(predicted_scores=dev_test_scores,
                ground_truth_scores=dev_gt_scores)
    dev_test_scores = shifter.transform(dev_test_scores)
    dev_acc_score, dev_auc_score, dev_c_at_1_score = \
            pan_metrics(prediction_scores=dev_test_scores,
                        ground_truth_scores=dev_gt_scores)
    print('::::: Development scores ::::::')
    print('Dev AUC:', dev_auc_score)
    print('Dev acc:', dev_acc_score)
    print('Dev c@1:', dev_c_at_1_score)

    ####### test phase ##################################################################################
    vectorizer = Vectorizer(mfi = mfi,
                            vector_space = vector_space,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size,
                            min_df=min_df)
    test_X = vectorizer.fit_transform(test_documents).toarray()
    label_encoder = LabelEncoder()
    label_encoder.fit(test_labels)
    test_y = label_encoder.transform(test_labels)
    test_train_X, test_train_y, test_test_X, \
        test_test_y, test_gt_scores = train_dev_split(test_X, test_y)
    test_verifier = Verifier(metric = metric,
                            base = base)
    test_verifier.fit(test_train_X, test_train_y)
    test_test_scores = test_verifier.predict_proba(test_X = test_test_X,
                                          test_y = test_test_y)
    # apply trained shifter:
    test_test_scores = shifter.transform(test_test_scores)
    test_acc_score, test_auc_score, test_c_at_1_score = \
            pan_metrics(prediction_scores=test_test_scores,
                        ground_truth_scores=test_gt_scores)
    print('::::: Test scores ::::::')
    print('Test AUC:', test_auc_score)
    print('Test acc:', test_acc_score)
    print('Test c@1:', test_c_at_1_score)

    f_results.write(','.join([str(i) for i in [metric, vector_space,\
                                test_c_at_1_score, test_auc_score,\
                                test_c_at_1_score * test_auc_score]])+'\n')

    holder[vector_space][metric] = (test_c_at_1_score, test_gt_scores, test_test_scores)
    precisions, recalls, thresholds = precision_recall_curve(test_gt_scores, test_test_scores)
    if metric == 'cng':
        c = c1
    elif metric == 'manhattan':
        c = c2
    elif metric == 'minmax':
        c = c3
    elif metric == 'cosine':
        c = c4
    if vector_space == 'tf':
        ls = 'solid'
    elif vector_space == 'tf_std':
        ls = 'dotted'
    elif vector_space == 'tf_idf':
        ls = 'dashed'
    sb.plt.plot(recalls, precisions, linestyle=ls, color=c, \
        label=vector_space+'+'+metric+': '+"{:.2f}".format(test_c_at_1_score*100))


    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 'xx-small')
    #lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=3, mode="expand", borderaxespad=0., fontsize = 'xx-small')
    sb.plt.gca().set_aspect('equal', adjustable='box')
    sb.plt.savefig("../output/latin_prec_rec_o1.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

f_results.close()

# significance testing:
comb_combs = sorted([i for i in combinations(combs, 2)])
signif_df = pd.DataFrame(index=combs, columns=combs).fillna('')

for comb1, comb2 in comb_combs:
    print('\t-Testing significance for:', comb1, 'vs', comb2)

    # get results from holder:
    system1 = holder[comb1[0]][comb1[1]][-1]
    system2 = holder[comb2[0]][comb2[1]][-1]
    gt_scores = holder[comb1[0]][comb1[1]][1]

    # binarize:
    system1 = binarize(system1)
    system2 = binarize(system2)
    gold = binarize(gt_scores)
        
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

    # update table:
    signif_df[comb1][comb2] = s
    signif_df[comb2][comb1] = s

# fill in c@1 score in main diagonal:
for comb in combs:
    c_at_1 = holder[comb[0]][comb[1]][0]
    signif_df[comb][comb] = c_at_1

# normalize labels:
signif_df.columns = ['+'.join(i).replace('_', '-') for i in signif_df.columns]
signif_df.index = ['+'.join(i).replace('_', '-') for i in signif_df.index]

signif_df.to_csv('../output/tables/latin_dev_o1_signif.csv')

