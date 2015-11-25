import codecs
import glob
import os
import sys
from collections import Counter
import numpy as np
from vectorization import Vectorizer

from sklearn.cross_validation import train_test_split

def load_pan_dataset(directory, ext='txt', encoding='utf8'):
    """
    Loads the data from `directory`, which should hold subdirs
    for each "problem"/author in a dataset. As with the official
    PAN datasets, each `unknown` instance for an author is included
    as test data.
    Returns `train_data` and `test_data` which are both lists of
    (author, text) tuples.
    """
    train_data, test_data = [], []
    for author in sorted(os.listdir(directory)):
        path = os.sep.join((directory, author))
        if os.path.isdir(path):
            for filepath in sorted(glob.glob(path+'/*.'+ext)):
                text = codecs.open(filepath, mode='r').read()
                name = os.path.splitext(os.path.basename(filepath))[0]
                if name == 'unknown':
                    test_data.append((author, text))
                else:
                    train_data.append((author, text))
    return train_data, test_data

def get_vocab_size(corpus_dir,
                  ngram_type,
                  ngram_size,
                  min_df=0.0,
                  phase='train'):
    """
    Fits a vectorizer on the train data under `corpus_dir`,
    i.e. excluding the 'unknown' texts. Returns the max number 
    of features available, given `ngram_type` and `ngram_size`.
    """
    train_data, _ = load_pan_dataset(corpus_dir+'/'+phase)
    train_labels, train_documents = zip(*train_data)
    
    # vectorize maximum nb of features:
    vectorizer = Vectorizer(mfi = sys.maxint,
                            ngram_type = ngram_type,
                            ngram_size = ngram_size,
                            min_df = min_df)
    vectorizer.fit(train_documents)
    return len(vectorizer.feature_names)


def load_ground_truth(filepath, labels):
    """
    Loads the ground truth for a given dataset (truth.txt`).
    Needs the problems labels in the correct order,
    to ensure that scores and labels are properly
    aligned.
    """
    ground_truth = {}
    for line in open(filepath).readlines():
        problem_id, outcome = line.strip().split()
        if outcome == 'Y':
            outcome = 1.0
        elif outcome == 'N':
            outcome = 0.0
        ground_truth[problem_id] = outcome
    return [ground_truth[l] for l in labels]

def train_dev_split(train_X, train_y, random_state=1027):
    X_dev, X_test, y_dev, y_test = train_test_split(train_X, train_y,
                                        test_size=.5,
                                        random_state=random_state,
                                        stratify=train_y)

    test_gt_scores = []

    # randomly select 1/2 of the idxs:
    np.random.seed(random_state)
    author_options = set(train_y)
    rnd_idxs = np.random.choice(len(y_test), int(len(y_test)/2))

    for idx, y in enumerate(y_test):
        if idx in rnd_idxs:
            # pick random other author:
            real_author = y_test[idx]
            other_authors = [a for a in author_options if a != real_author]
            fake_author = np.random.choice(other_authors, 1)[0]
            y_test[idx] = fake_author
            # indicate it's a negative example:
            test_gt_scores.append(0.0)
        else:
            test_gt_scores.append(1.0)

    return X_dev, y_dev,\
           X_test, y_test, test_gt_scores


def binarize(scores):
    scs = []
    for sc in scores:
        if sc == 0.5:
            scs.append('X')
        elif sc < 0.5:
            scs.append('N')
        elif sc > 0.5:
            scs.append('Y')
    return scs

def stringify(i):
    return '+'.join(i[::-1]).replace('_', '-')

                         
    