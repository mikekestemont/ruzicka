import codecs
import glob
import os
from collections import Counter

import numpy as np

def load_pan_dataset(directory, ext='txt', encoding='utf8'):
    train_data, test_data = [], []

    for author in os.listdir(directory):
        path = os.sep.join((directory, author))
        if os.path.isdir(path):
            for filepath in glob.glob(path+'/*.'+ext):
                text = codecs.open(filepath, mode='r').read()
                name = os.path.splitext(os.path.basename(filepath))[0]
                if name == 'unknown':
                    test_data.append    ((author, text))
                elif name.startswith('known'):
                    train_data.append((author, text))
    return train_data, test_data

def load_ground_truth(filepath):
    ground_truth = {}
    for line in open(filepath).readlines():
        problem_id, outcome = line.strip().split()
        if outcome == 'Y':
            outcome = 1.0
        elif outcome == 'N':
            outcome = 0.0
        ground_truth[problem_id] = outcome
    return ground_truth

def train_dev_split(train_X, train_y):
    # set rnd seed:
    np.random.seed(train_X.shape[0])
    # select authors for which we have more than one training documents:
    samplable_authors = [i for i, v in Counter(train_y).items() if v > 1]
    # collect indexes of dev items which we will remove from train:
    dev_idxs = []
    for author in samplable_authors:
        # select the index of a random text:
        available_idxs = [i for i in range(len(train_y))
                            if train_y[i] == author]
        dev_idxs.append(np.random.choice(available_idxs))
    
    # tmp remove selected dev docs from training data:
    tmp_train_X = np.array([train_X[i] for i in range(train_X.shape[0]) if i not in dev_idxs])
    tmp_train_y = np.array([train_y[i] for i in range(train_X.shape[0]) if i not in dev_idxs])
    
    dev_X, dev_y, dev_labels, dev_ground_truth = [], [], [], {}
    for cnt, dev_idx in enumerate(dev_idxs):
        # create first same author-problem:
        dev_X.append(train_X[dev_idx])
        dev_y.append(train_y[dev_idx])
        dev_label = 'same_'+str(cnt+1) # assign unique label for evaluation purposes
        dev_labels.append(dev_label)
        dev_ground_truth[dev_label] = 1.0
        # now create different-author problem with the same doc:
        dev_X.append(train_X[dev_idx])
        # select random other author:
        other_authors = [y for y in train_y if y != train_y[dev_idx]]
        dev_y.append(np.random.choice(other_authors))
        dev_label = 'diff_'+str(cnt+1)
        dev_labels.append(dev_label)
        dev_ground_truth[dev_label] = 0.0
    dev_X = np.array(dev_X)
    
    return tmp_train_X, tmp_train_y, \
           dev_X, dev_y, dev_labels, dev_ground_truth
                         
    