from __future__ import print_function
import os
import json
import pickle
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ruzicka.utilities import *
from ruzicka.vectorization import Vectorizer
from ruzicka.score_shifting import ScoreShifter
from ruzicka.evaluation import pan_metrics
from ruzicka.Order1Verifier import Order1Verifier
from ruzicka.Order2Verifier import Order2Verifier
from ruzicka import art

# corpora:
corpus_dirs = ['../data/2014/en_essays/',
               '../data/2014/du_essays/',
               '../data/2014/gr_articles/',
               '../data/2014/sp_articles/',
               '../data/2014/du_reviews/',
               '../data/2014/en_novels/',
               '../data/2014/en_novels/',]

# set hyperparameters:
base = 'profile'
nb_bootstrap_iter = 100
rnd_prop = 0.5
min_df = 1
nb_imposters = 30
feature_types = {'char 3-gram': ('char', 3),
                 'char 4-gram': ('char', 4),
                 'word 1-gram': ('word', 1)}


for corpus_dir in corpus_dirs:
    print('>>> corpus:', corpus_dir)
    
    # results for 1st order method:
    df_1st_order_dev = pd.DataFrame(columns=['feature type']+sorted(feature_types))
    df_1st_order_test = pd.DataFrame(columns=['feature type']+sorted(feature_types))
    
    # results for 2nd order method:
    df_2nd_order_dev = pd.DataFrame(columns=['feature type']+sorted(feature_types))
    df_2nd_order_test = pd.DataFrame(columns=['feature type']+sorted(feature_types))
    
    # significance:
    df_proba = pd.DataFrame(columns=['feature type']+sorted(feature_types))

    # loop over metrics:
    for metric in ('cng', 'cosine', 'minmax', 'manhattan'):
        print('\t\t>>>', metric)
        
        for vector_space in ('tf_std', 'tf_idf', 'tf'):
            
            print('\t\t\t +++', vector_space)
            combo = metric+' - '+vector_space

            dev_ord1_row = [combo]
            test_ord1_row = [combo]
            dev_ord2_row = [combo]
            test_ord2_row = [combo]
            proba_row = [combo]

            for feature_type in sorted(feature_types):

                ########################################################################
                ### 1st order: DEVEL PART ##############################################
                ########################################################################

                # get max nb of features for current feature family:
                ngram_type, ngram_size = feature_types[feature_type]
                max_vocab_size = get_vocab_size(corpus_dir = corpus_dir,
                                    ngram_type = ngram_type,
                                    ngram_size = ngram_size,
                                    min_df = min_df,
                                    phase = 'train')

                # preprocess:
                dev_train_data, dev_test_data = load_pan_dataset(corpus_dir+'train')
                dev_train_labels, dev_train_documents = zip(*dev_train_data)
                dev_test_labels, dev_test_documents = zip(*dev_test_data)
                
                # vectorize:
                vectorizer = Vectorizer(mfi = max_vocab_size,
                                        vector_space = vector_space,
                                        ngram_type = ngram_type,
                                        ngram_size = ngram_size)
            
                # transform (and unsparsify):
                dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
                dev_test_X = vectorizer.transform(dev_test_documents).toarray()
                
                # encode author labels:
                label_encoder = LabelEncoder()
                label_encoder.fit(dev_train_labels+dev_test_labels)
                dev_train_y = label_encoder.transform(dev_train_labels)
                dev_test_y = label_encoder.transform(dev_test_labels)
                
                # fit the verifier:
                dev_verifier = Order1Verifier(metric = metric,
                                              base = base)
                dev_verifier.fit(dev_train_X, dev_train_y)
                dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                                             test_y = dev_test_y)
                
                # load the ground truth:
                dev_gt_scores = load_ground_truth(
                        filepath=os.sep.join((corpus_dir, 'train', 'truth.txt')),
                        labels=dev_test_labels)
                
                # fit the score shifter
                shifter = ScoreShifter()
                shifter.fit(predicted_scores=dev_test_scores,
                            ground_truth_scores=dev_gt_scores)
                dev_test_scores = shifter.transform(dev_test_scores)
                
                dev_acc_score, dev_auc_score, dev_c_at_1_score = \
                        pan_metrics(prediction_scores=dev_test_scores,
                                    ground_truth_scores=dev_gt_scores)
                print(dev_acc_score)

                dev_ord1_row.append(dev_auc_score * dev_c_at_1_score)


                ########################################################################
                ### 1st order: TEST PART ###############################################
                ########################################################################

                max_vocab_size = get_vocab_size(corpus_dir = corpus_dir,
                                                ngram_type = ngram_type,
                                                ngram_size = ngram_size,
                                                min_df = min_df,
                                                phase = 'test')
                # preprocess:
                train_data, test_data = load_pan_dataset(corpus_dir+'test')
                train_labels, train_documents = zip(*train_data)
                test_labels, test_documents = zip(*test_data)
                
                # vectorize:
                vectorizer = Vectorizer(mfi = max_vocab_size,
                                        vector_space = vector_space,
                                        ngram_type = ngram_type,
                                        ngram_size = ngram_size,
                                        min_df = min_df)
                train_X = vectorizer.fit_transform(train_documents).toarray()
                test_X = vectorizer.transform(test_documents).toarray()
                
                # encode author labels:
                label_encoder = LabelEncoder()
                label_encoder.fit(train_labels+test_labels)
                train_y = label_encoder.transform(train_labels)
                test_y = label_encoder.transform(test_labels)
                
                # fit and predict a verifier on the test data:
                test_verifier = Order1Verifier(metric = metric,
                                         base = base)
                test_verifier.fit(train_X, train_y)
                test_scores = test_verifier.predict_proba(test_X=test_X,
                                                          test_y=test_y)
                
                # load the ground truth:
                test_gt_scores = load_ground_truth(
                                    filepath=os.sep.join((corpus_dir, 'test', 'truth.txt')),
                                    labels=test_labels)
                
                # apply the optimzed score shifter:
                ord1_test_scores = shifter.transform(test_scores)
                
                test_acc_score, test_auc_score, test_c_at_1_score = \
                        pan_metrics(prediction_scores=ord1_test_scores,
                                    ground_truth_scores=test_gt_scores)

                test_ord1_row.append(test_auc_score * test_c_at_1_score)

                ########################################################################
                ### 2nd order: DEVEL PART ##############################################
                ########################################################################

                # determine mfi:
                max_vocab_size = get_vocab_size(corpus_dir = corpus_dir,
                                    ngram_type = ngram_type,
                                    ngram_size = ngram_size,
                                    min_df = min_df,
                                    phase = 'train')

                # preprocess:
                dev_train_data, dev_test_data = load_pan_dataset(corpus_dir+'train')
                dev_train_labels, dev_train_documents = zip(*dev_train_data)
                dev_test_labels, dev_test_documents = zip(*dev_test_data)
                
                # vectorize:
                vectorizer = Vectorizer(mfi = max_vocab_size,
                                        vector_space = vector_space,
                                        ngram_type = ngram_type,
                                        ngram_size = ngram_size)
                
                # transform (and unsparsify):
                dev_train_X = vectorizer.fit_transform(dev_train_documents).toarray()
                dev_test_X = vectorizer.transform(dev_test_documents).toarray()
                
                # encode author labels:
                label_encoder = LabelEncoder()
                label_encoder.fit(dev_train_labels+dev_test_labels)
                dev_train_y = label_encoder.transform(dev_train_labels)
                dev_test_y = label_encoder.transform(dev_test_labels)
                
                # fit the verifier:
                dev_verifier = Order2Verifier(metric = metric,
                                        base = base,
                                        nb_bootstrap_iter = nb_bootstrap_iter,
                                        rnd_prop = rnd_prop)
                dev_verifier.fit(dev_train_X, dev_train_y)
                dev_test_scores = dev_verifier.predict_proba(test_X = dev_test_X,
                                                             test_y = dev_test_y,
                                                             nb_imposters = nb_imposters)
                    
                # load the ground truth:
                dev_gt_scores = load_ground_truth(
                                    filepath=os.sep.join((corpus_dir, 'train', 'truth.txt')),
                                    labels=dev_test_labels)
                
                # fit the score shifter
                shifter = ScoreShifter()
                shifter.fit(predicted_scores=dev_test_scores,
                            ground_truth_scores=dev_gt_scores)
                dev_test_scores = shifter.transform(dev_test_scores)
                
                dev_acc_score, dev_auc_score, dev_c_at_1_score = \
                        pan_metrics(prediction_scores=dev_test_scores,
                                    ground_truth_scores=dev_gt_scores)
                
                dev_ord2_row.append(dev_auc_score * dev_c_at_1_score)

                ########################################################################
                ### 2nd order: TEST PART ###############################################
                ########################################################################

                # determine mfi:
                max_vocab_size = get_vocab_size(corpus_dir = corpus_dir,
                                    ngram_type = ngram_type,
                                    ngram_size = ngram_size,
                                    min_df = min_df,
                                    phase = 'test')

                # preprocess:
                train_data, test_data = load_pan_dataset(corpus_dir+'test')
                train_labels, train_documents = zip(*train_data)
                test_labels, test_documents = zip(*test_data)
                
                # vectorize:
                vectorizer = Vectorizer(mfi = max_vocab_size,
                                        vector_space = vector_space,
                                        ngram_type = ngram_type,
                                        ngram_size = ngram_size,
                                        min_df = min_df)
                train_X = vectorizer.fit_transform(train_documents).toarray()
                test_X = vectorizer.transform(test_documents).toarray()
                
                # encode author labels:
                label_encoder = LabelEncoder()
                label_encoder.fit(train_labels+test_labels)
                train_y = label_encoder.transform(train_labels)
                test_y = label_encoder.transform(test_labels)
                
                # fit and predict a verifier on the test data:
                test_verifier = Order2Verifier(metric = metric,
                                         base = base,
                                         nb_bootstrap_iter = nb_bootstrap_iter,
                                         rnd_prop = rnd_prop)
                test_verifier.fit(train_X, train_y)
                test_scores = test_verifier.predict_proba(test_X=test_X,
                                                          test_y=test_y,
                                                          nb_imposters = nb_imposters)
                
                # load the ground truth:
                test_gt_scores = load_ground_truth(
                                    filepath=os.sep.join((corpus_dir, 'test', 'truth.txt')),
                                    labels=test_labels)
                
                # apply the optimized score shifter:
                ord2_test_scores = shifter.transform(test_scores)
                
                test_acc_score, test_auc_score, test_c_at_1_score = \
                        pan_metrics(prediction_scores=ord2_test_scores,
                                    ground_truth_scores=test_gt_scores)
                        
                test_ord2_row.append(test_auc_score * test_c_at_1_score)


                ########################################################################
                ### SIGNIFICANCE #######################################################
                ########################################################################
                system1 = binarize(ord1_test_scores)
                system2 = binarize(ord2_test_scores)

                gold = binarize(test_gt_scores)
                
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
                
                proba_row.append(s)            
            
            df_1st_order_dev.loc[len(df_1st_order_dev)] = dev_ord1_row
            df_1st_order_test.loc[len(df_1st_order_test)] = test_ord1_row

            df_2nd_order_dev.loc[len(df_2nd_order_dev)] = dev_ord2_row
            df_2nd_order_test.loc[len(df_2nd_order_test)] = test_ord2_row

            df_proba.loc[len(df_proba)] = proba_row

    # get corpus basename:
    c = os.path.basename(corpus_dir[:-1])

    # save tables:
    df_1st_order_dev.set_index('feature type', inplace=True, drop=True)
    df_1st_order_dev.to_excel('../output/'+c+'_dev_1st_order.xlsx')

    df_1st_order_test.set_index('feature type', inplace=True, drop=True)
    df_1st_order_test.to_excel('../output/'+c+'_test_1st_order.xlsx')

    df_2nd_order_dev.set_index('feature type', inplace=True, drop=True)
    df_2nd_order_dev.to_excel('../output/'+c+'_dev_2nd_order.xlsx')

    df_2nd_order_test.set_index('feature type', inplace=True, drop=True)
    df_2nd_order_test.to_excel('../output/'+c+'_test_2nd_order.xlsx')

    df_proba.set_index('feature type', inplace=True, drop=True)
    df_proba.to_excel('../output/'+c+'_significance.xlsx')
