from itertools import product
import json
import os
import pandas as pd
from experimentation import test_experiment

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
    
    test_results = {} # create an object to store actual scores in
    for vsm in settings[corpus_dir]:
        print('\t\t+'+vsm)
        if vsm not in test_results:
            test_results[vsm] = {}
        for metric in settings[corpus_dir][vsm]:
            print('\t\t\t* '+metric)

            # load best settings:
            p1 = settings[corpus_dir][vsm][metric]['p1']
            p2 = settings[corpus_dir][vsm][metric]['p2']
            mfi = settings[corpus_dir][vsm][metric]['mfi']

            # run the test experiment
            dev_auc_score, dev_acc_score, dev_c_at_1_score, test_scores = \
                                    test_experiment(corpus_dir = corpus_dir+'/',
                                                   mfi = mfi,
                                                   vector_space = vsm,
                                                   ngram_type = 'char',
                                                   ngram_size = 4,
                                                   metric = metric,
                                                   base = 'profile',
                                                   nb_bootstrap_iter = 100,
                                                   rnd_prop = 0.5,
                                                   nb_imposters = 30,
                                                   method = 'm1',
                                                   p1 = p1,
                                                   p2 = p2)
            auc_df.ix[vsm][metric] = dev_auc_score
            acc_df.ix[vsm][metric] = dev_acc_score
            c_at_1_df.ix[vsm][metric] = dev_c_at_1_score
            score_df.ix[vsm][metric] = dev_auc_score * dev_c_at_1_score
            # store the actual scores
            test_results[vsm][metric] = test_scores

    # write away score tables:
    table_dir = '../output/tables/'
    if not os.path.isdir(table_dir):
        os.mkdir(table_dir)

    corpus_name = os.path.basename(corpus_dir)
    auc_df.to_csv(table_dir+corpus_name+'_auc.csv')
    acc_df.to_csv(table_dir+corpus_name+'_acc.csv')
    c_at_1_df.to_csv(table_dir+corpus_name+'_c_at_1.csv')
    score_df.to_csv(table_dir+corpus_name+'_score.csv')

    # now significance testing:
    combs = sorted(['+'.join(i).replace('_', '-') for i in \
                    product(index, columns)])
    signif_df = pd.DataFrame(index=combs, columns=combs).fillna('')

    # fill tabele with significance tests:
    
    print(signif_df)
    signif_df.to_csv(table_dir+corpus_name+'_signif.csv')



            

